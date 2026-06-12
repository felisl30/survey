#!/usr/bin/env python3
"""
flare_controller_s3.py

Controlador principal para S3: FLARE-like / FLARE-inspired active retrieval.

Este archivo ejecuta el loop para UNA pregunta. Luego se puede importar desde
run_s3_flare_like.py para correr todo el CSV.

Flujo:
    pregunta
        -> generar próxima oración / claim candidato
        -> decidir si esa oración necesita retrieval
        -> recuperar evidencia si hace falta
        -> regenerar/corregir la oración con evidencia
        -> acumular respuesta parcial
        -> repetir
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_CODE_DIR = Path(__file__).resolve().parent
S2_CODE_DIR = PROJECT_ROOT / "s2_model_code"
S1_CODE_DIR = PROJECT_ROOT / "s1_model_code"

for path in [PROJECT_ROOT, S3_CODE_DIR, S2_CODE_DIR, S1_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from direct_llm import ask_direct_llm_with_metadata
    from project_paths import S2_INDEX_DIR
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar direct_llm.py. Ejecutá desde la raíz del proyecto."
    ) from exc

try:
    from retriever_s1 import S1Retriever, clean_text as s1_clean_text
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar retriever_s1.py. S3 reutiliza el retriever de S1/S2."
    ) from exc

try:
    from prompts_s3 import (
        S3_CANDIDATE_SYSTEM_PROMPT,
        S3_REGENERATE_SYSTEM_PROMPT,
        build_candidate_prompt,
        build_regenerate_prompt,
    )
    from active_retrieval_policy_s3 import (
        ABSTENTION_MARKERS,
        CLARIFICATION_MARKERS,
        needs_retrieval,
    )
except ModuleNotFoundError:
    from s3_model_code.prompts_s3 import (
        S3_CANDIDATE_SYSTEM_PROMPT,
        S3_REGENERATE_SYSTEM_PROMPT,
        build_candidate_prompt,
        build_regenerate_prompt,
    )
    from s3_model_code.active_retrieval_policy_s3 import (
        ABSTENTION_MARKERS,
        CLARIFICATION_MARKERS,
        needs_retrieval,
    )


DEFAULT_INDEX_DIR = S2_INDEX_DIR
DEFAULT_MAX_STEPS = 4
DEFAULT_TOP_K_PER_STEP = 2
DEFAULT_MAX_RETRIEVAL_STEPS = 3
DEFAULT_MAX_TOTAL_CHUNKS = 6
DEFAULT_MAX_CHARS_PER_CHUNK = 900


def clean_text(value: Any) -> str:
    try:
        return s1_clean_text(value)
    except Exception:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def normalize_for_rules(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", ".")
            if text.endswith("%"):
                number = float(text[:-1]) / 100.0
            else:
                number = float(text)
        else:
            number = float(value)
    except (TypeError, ValueError):
        return default

    if 1.0 < number <= 100.0:
        number = number / 100.0

    return min(max(number, 0.0), 1.0)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)

    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def strip_markdown_fence(text: str) -> str:
    stripped = clean_text(text)
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> str | None:
    text = strip_markdown_fence(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]

    return None


def load_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    json_text = extract_first_json_object(text)
    if not json_text:
        return None, "No se encontró objeto JSON."

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON inválido: {exc}"

    if not isinstance(parsed, dict):
        return None, "El JSON parseado no es un objeto."

    return parsed, ""


def clean_sentence(text: Any) -> str:
    sentence = normalize_space(text)
    sentence = re.sub(r"^[-*•]\s*", "", sentence).strip()
    sentence = sentence.strip('"').strip("'").strip()

    if len(sentence) > 700:
        sentence = sentence[:700].rstrip() + " [...]"

    return sentence


def contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def call_llm_json(
    *,
    prompt: str,
    system_prompt: str,
    model: str | None,
    max_retries: int,
    dry_run: bool,
    dry_run_payload: dict[str, Any],
) -> dict[str, Any]:
    if dry_run:
        raw_output = json.dumps(dry_run_payload, ensure_ascii=False)
        return {
            "parsed": dry_run_payload,
            "raw_output": raw_output,
            "parse_error": "",
            "model": "dry_run",
            "usage_json": "{}",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "dry_run": True,
        }

    result = ask_direct_llm_with_metadata(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_retries=max_retries,
    )

    raw_output = clean_text(result.get("raw_output", ""))
    parsed, parse_error = load_json_object(raw_output)

    return {
        "parsed": parsed or {},
        "raw_output": raw_output,
        "parse_error": parse_error,
        "model": result.get("model", model or ""),
        "usage_json": result.get("usage_json", ""),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "total_tokens": result.get("total_tokens"),
        "latency_seconds": result.get("latency_seconds"),
        "dry_run": False,
    }


def validate_candidate_payload(obj: dict[str, Any]) -> dict[str, Any]:
    done = coerce_bool(obj.get("done"), default=False)
    candidate_sentence = clean_sentence(obj.get("candidate_sentence", ""))
    confidence = safe_float(obj.get("confidence", 0.0), default=0.0)

    if not done and not candidate_sentence:
        done = True

    return {
        "done": done,
        "candidate_sentence": candidate_sentence,
        "confidence": confidence,
    }


def generate_candidate_sentence(
    *,
    question: str,
    partial_answer: str,
    step: int,
    max_steps: int,
    task_type: str = "open_qa",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    prompt = build_candidate_prompt(
        question=question,
        partial_answer=partial_answer,
        step=step,
        max_steps=max_steps,
        task_type=task_type,
    )

    if partial_answer.strip():
        dry_payload = {
            "done": True,
            "candidate_sentence": "",
            "confidence": 0.80,
        }
    else:
        dry_payload = {
            "done": False,
            "candidate_sentence": "This question requires checking the available evidence before giving a final answer.",
            "confidence": 0.55,
        }

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S3_CANDIDATE_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    parsed = validate_candidate_payload(call["parsed"])

    return {
        **parsed,
        "candidate_raw_output": call["raw_output"],
        "candidate_parse_error": call["parse_error"],
        "candidate_model": call["model"],
        "candidate_input_tokens": call["input_tokens"],
        "candidate_output_tokens": call["output_tokens"],
        "candidate_total_tokens": call["total_tokens"],
        "candidate_latency_seconds": call["latency_seconds"],
    }


def sanitize_chunk(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": item.get("rank", ""),
        "chunk_id": clean_text(item.get("chunk_id", "")),
        "doc_id": clean_text(item.get("doc_id", "")),
        "title": clean_text(item.get("title", "")),
        "score": item.get("score", ""),
        "text": clean_text(item.get("text", "")),
        "source": clean_text(item.get("source", "")),
        "source_split": clean_text(item.get("source_split", "")),
        "topic": clean_text(item.get("topic", "")),
        "question_id": clean_text(item.get("question_id", "")),
        "original_hotpotqa_id": clean_text(item.get("original_hotpotqa_id", "")),
        "paragraph_index": item.get("paragraph_index", ""),
        "is_gold_evidence": item.get("is_gold_evidence", ""),
    }


def get_chunk_id(item: dict[str, Any]) -> str:
    return clean_text(item.get("chunk_id", ""))


def summarize_evidence_ids(chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in chunks:
        cid = get_chunk_id(item)
        if cid and cid not in ids:
            ids.append(cid)
    return ids


def create_retriever(index_dir: Path = DEFAULT_INDEX_DIR) -> S1Retriever:
    if not index_dir.exists():
        raise FileNotFoundError(f"No existe index_dir: {index_dir}")
    return S1Retriever(index_dir=index_dir)


def build_retrieval_query(
    *,
    question: str,
    candidate_sentence: str,
    partial_answer: str = "",
) -> str:
    question = normalize_space(question)
    candidate_sentence = normalize_space(candidate_sentence)

    if candidate_sentence:
        return f"{question} {candidate_sentence}".strip()

    return question


def retrieve_for_candidate(
    *,
    retriever: S1Retriever,
    retrieval_query: str,
    top_k: int,
    already_seen_chunk_ids: set[str] | None = None,
    max_total_chunks_remaining: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    already_seen_chunk_ids = already_seen_chunk_ids or set()
    effective_top_k = top_k + len(already_seen_chunk_ids)

    start = time.time()
    raw_chunks = retriever.retrieve(retrieval_query, top_k=effective_top_k)
    latency = time.time() - start

    selected: list[dict[str, Any]] = []
    for item in raw_chunks:
        safe = sanitize_chunk(item)
        cid = get_chunk_id(safe)

        if cid and cid in already_seen_chunk_ids:
            continue

        selected.append(safe)

        if len(selected) >= top_k:
            break

    if max_total_chunks_remaining is not None and max_total_chunks_remaining >= 0:
        selected = selected[:max_total_chunks_remaining]

    return selected, round(latency, 3)


def validate_regeneration_payload(obj: dict[str, Any]) -> dict[str, Any]:
    final_sentence = clean_sentence(obj.get("final_sentence", ""))
    used_evidence = coerce_bool(obj.get("used_evidence"), default=True)
    confidence = safe_float(obj.get("confidence", 0.0), default=0.0)
    abstain_sentence = coerce_bool(obj.get("abstain_sentence"), default=False)

    support_status = clean_text(obj.get("support_status", "")).lower()
    if support_status not in {"supported", "corrected", "not_enough_info", "refuted"}:
        support_status = "not_enough_info" if abstain_sentence else "corrected"

    final_sentence_lower = normalize_for_rules(final_sentence)

    if contains_any(final_sentence_lower, ABSTENTION_MARKERS):
        abstain_sentence = True
        if support_status not in {"not_enough_info", "refuted"}:
            support_status = "not_enough_info"

    if support_status in {"not_enough_info", "refuted"}:
        abstain_sentence = True

    if not final_sentence:
        final_sentence = "No hay información suficiente para sostener esta afirmación con la evidencia recuperada."
        abstain_sentence = True
        support_status = "not_enough_info"

    return {
        "final_sentence": final_sentence,
        "used_evidence": used_evidence,
        "confidence": confidence,
        "abstain_sentence": abstain_sentence,
        "support_status": support_status,
    }


def regenerate_sentence_with_evidence(
    *,
    question: str,
    partial_answer: str,
    candidate_sentence: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str | None = None,
    max_retries: int = 2,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    dry_run: bool = False,
) -> dict[str, Any]:
    prompt = build_regenerate_prompt(
        question=question,
        partial_answer=partial_answer,
        candidate_sentence=candidate_sentence,
        retrieved_chunks=retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    dry_payload = {
        "final_sentence": candidate_sentence if retrieved_chunks else "No hay información suficiente para sostener esta afirmación con la evidencia disponible.",
        "used_evidence": True,
        "confidence": 0.70 if retrieved_chunks else 0.40,
        "abstain_sentence": False if retrieved_chunks else True,
        "support_status": "supported" if retrieved_chunks else "not_enough_info",
    }

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S3_REGENERATE_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    parsed = validate_regeneration_payload(call["parsed"])

    return {
        **parsed,
        "regenerate_raw_output": call["raw_output"],
        "regenerate_parse_error": call["parse_error"],
        "regenerate_model": call["model"],
        "regenerate_input_tokens": call["input_tokens"],
        "regenerate_output_tokens": call["output_tokens"],
        "regenerate_total_tokens": call["total_tokens"],
        "regenerate_latency_seconds": call["latency_seconds"],
    }


def append_sentence(partial_answer: str, sentence: str) -> str:
    partial = clean_text(partial_answer)
    sentence = clean_sentence(sentence)

    if not sentence:
        return partial
    if not partial:
        return sentence
    if partial.endswith((".", "!", "?")):
        return f"{partial} {sentence}"
    return f"{partial}. {sentence}"


def should_stop_after_supported_retrieval(
    *,
    task_type: str,
    needs_retrieval: bool,
    final_sentence: str,
    abstain_sentence: bool,
    support_status: str,
    sentence_confidence: float,
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    """
    Corta el loop FLARE-like cuando una oración generada con retrieval ya quedó
    suficientemente verificada.

    Motivación:
    - En preguntas open_retrieval, una primera oración supported/corrected suele
      ser ya una respuesta completa.
    - Si seguimos generando, el modelo puede producir una oración redundante
      tipo "Therefore..." y forzar una nueva recuperación innecesaria.
    - Esa segunda recuperación puede traer chunks no pertinentes y terminar en
      abstención, aunque la primera respuesta ya estuviera bien soportada.
    """
    if task_type != "open_retrieval":
        return False

    if not needs_retrieval:
        return False

    if not retrieved_chunks:
        return False

    if abstain_sentence:
        return False

    support_status = clean_text(support_status).lower()
    if support_status not in {"supported", "corrected"}:
        return False

    if safe_float(sentence_confidence, default=0.0) < 0.55:
        return False

    final_lower = normalize_for_rules(final_sentence)
    if contains_any(final_lower, ABSTENTION_MARKERS):
        return False

    if contains_any(final_lower, CLARIFICATION_MARKERS):
        return False

    return True


def should_stop_after_sentence(
    *,
    final_sentence: str,
    abstain_sentence: bool,
    step: int,
    max_steps: int,
) -> bool:
    lower = normalize_for_rules(final_sentence)

    if step >= max_steps:
        return True
    if abstain_sentence:
        return True
    if contains_any(lower, ABSTENTION_MARKERS):
        return True
    if contains_any(lower, CLARIFICATION_MARKERS):
        return True

    return False


def merge_usage_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    latency = 0.0

    found_input = False
    found_output = False
    found_total = False
    found_latency = False

    for item in items:
        value = item.get("input_tokens")
        if value is not None and clean_text(value) != "":
            input_tokens += int(float(value))
            found_input = True

        value = item.get("output_tokens")
        if value is not None and clean_text(value) != "":
            output_tokens += int(float(value))
            found_output = True

        value = item.get("total_tokens")
        if value is not None and clean_text(value) != "":
            total_tokens += int(float(value))
            found_total = True

        value = item.get("latency_seconds")
        if value is not None and clean_text(value) != "":
            latency += float(value)
            found_latency = True

    return {
        "input_tokens": input_tokens if found_input else None,
        "output_tokens": output_tokens if found_output else None,
        "total_tokens": total_tokens if found_total else None,
        "latency_seconds": round(latency, 3) if found_latency else None,
    }


def infer_final_abstention(
    *,
    answer: str,
    trace: list[dict[str, Any]],
) -> bool:
    """
    Determina si la respuesta final debe marcarse como abstención.

    Si cualquier paso del trace marcó abstain_sentence=true o
    support_status not_enough_info/refuted, la salida final debe tener
    abstained=true.
    """
    answer_lower = normalize_for_rules(answer)

    if contains_any(answer_lower, ABSTENTION_MARKERS):
        return True

    if contains_any(answer_lower, CLARIFICATION_MARKERS):
        return True

    for step in trace:
        if bool(step.get("abstain_sentence")):
            return True

        support_status = clean_text(step.get("support_status", "")).lower()
        if support_status in {"not_enough_info", "refuted"}:
            return True

        final_sentence = normalize_for_rules(step.get("final_sentence", ""))
        if contains_any(final_sentence, ABSTENTION_MARKERS):
            return True

    return False


def build_final_payload(
    *,
    answer: str,
    confidence_values: list[float],
    abstained: bool,
    evidence_ids: list[str],
    trace: list[dict[str, Any]],
    usage_items: list[dict[str, Any]],
    error: str = "",
) -> dict[str, Any]:
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    usage = merge_usage_dicts(usage_items)

    return {
        "answer": clean_text(answer),
        "confidence": round(float(confidence), 4),
        "abstained": bool(abstained),
        "retrieval_mode": "active",
        "num_generation_steps": len(trace),
        "num_retrieval_steps": int(sum(1 for step in trace if step.get("needs_retrieval"))),
        "num_chunks_retrieved_total": int(sum(len(step.get("retrieved_chunk_ids", [])) for step in trace)),
        "evidence_ids": evidence_ids,
        "flare_trace": trace,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_seconds": usage["latency_seconds"],
        "error": clean_text(error),
    }


def run_flare_controller_for_question(
    *,
    question: str,
    retriever: S1Retriever | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    top_k_per_step: int = DEFAULT_TOP_K_PER_STEP,
    max_retrieval_steps: int = DEFAULT_MAX_RETRIEVAL_STEPS,
    max_total_chunks: int = DEFAULT_MAX_TOTAL_CHUNKS,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    retrieval_strategy: str = "rules",
    task_type: str = "open_qa",
    retrieval_bias: str = "balanced",
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    question = normalize_space(question)
    if not question:
        raise ValueError("La pregunta está vacía.")

    if max_steps <= 0:
        raise ValueError("max_steps debe ser mayor que 0.")
    if top_k_per_step <= 0:
        raise ValueError("top_k_per_step debe ser mayor que 0.")
    if max_retrieval_steps < 0:
        raise ValueError("max_retrieval_steps no puede ser negativo.")
    if max_total_chunks < 0:
        raise ValueError("max_total_chunks no puede ser negativo.")

    if retriever is None and not dry_run:
        retriever = create_retriever(index_dir)

    partial_answer = ""
    trace: list[dict[str, Any]] = []
    evidence_used: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    confidence_values: list[float] = []
    usage_items: list[dict[str, Any]] = []
    retrieval_steps = 0
    error = ""

    try:
        for step in range(1, max_steps + 1):
            candidate = generate_candidate_sentence(
                question=question,
                partial_answer=partial_answer,
                step=step,
                max_steps=max_steps,
                task_type=task_type,
                model=model,
                max_retries=max_retries,
                dry_run=dry_run,
            )

            usage_items.append({
                "input_tokens": candidate.get("candidate_input_tokens"),
                "output_tokens": candidate.get("candidate_output_tokens"),
                "total_tokens": candidate.get("candidate_total_tokens"),
                "latency_seconds": candidate.get("candidate_latency_seconds"),
            })

            if candidate["done"]:
                break

            candidate_sentence = clean_sentence(candidate["candidate_sentence"])
            if not candidate_sentence:
                break

            decision = needs_retrieval(
                question=question,
                candidate_sentence=candidate_sentence,
                partial_answer=partial_answer,
                candidate_confidence=candidate.get("confidence"),
                task_type=task_type,
                retrieval_bias=retrieval_bias,
                strategy=retrieval_strategy,
                model=model,
                max_retries=max_retries,
                dry_run=dry_run,
            )

            usage_items.append({
                "input_tokens": decision.get("decision_input_tokens"),
                "output_tokens": decision.get("decision_output_tokens"),
                "total_tokens": decision.get("decision_total_tokens"),
                "latency_seconds": decision.get("decision_latency_seconds"),
            })

            needs_ret = bool(decision["needs_retrieval"])
            retrieval_query = ""
            retrieved_chunks: list[dict[str, Any]] = []
            retrieval_latency_seconds = 0.0
            retrieval_skipped_reason = ""

            if needs_ret:
                if retrieval_steps >= max_retrieval_steps:
                    retrieval_skipped_reason = "max_retrieval_steps_reached"
                    needs_ret = False
                elif len(seen_chunk_ids) >= max_total_chunks:
                    retrieval_skipped_reason = "max_total_chunks_reached"
                    needs_ret = False
                else:
                    retrieval_query = build_retrieval_query(
                        question=question,
                        candidate_sentence=candidate_sentence,
                        partial_answer=partial_answer,
                    )

                    remaining = max_total_chunks - len(seen_chunk_ids)

                    if dry_run:
                        retrieved_chunks = []
                        retrieval_latency_seconds = 0.0
                    else:
                        if retriever is None:
                            raise ValueError("retriever=None pero dry_run=False.")

                        retrieved_chunks, retrieval_latency_seconds = retrieve_for_candidate(
                            retriever=retriever,
                            retrieval_query=retrieval_query,
                            top_k=top_k_per_step,
                            already_seen_chunk_ids=seen_chunk_ids,
                            max_total_chunks_remaining=remaining,
                        )

                    retrieval_steps += 1

                    for item in retrieved_chunks:
                        cid = get_chunk_id(item)
                        if cid:
                            seen_chunk_ids.add(cid)
                        evidence_used.append(item)

            if needs_ret:
                regenerated = regenerate_sentence_with_evidence(
                    question=question,
                    partial_answer=partial_answer,
                    candidate_sentence=candidate_sentence,
                    retrieved_chunks=retrieved_chunks,
                    model=model,
                    max_retries=max_retries,
                    max_chars_per_chunk=max_chars_per_chunk,
                    dry_run=dry_run,
                )

                usage_items.append({
                    "input_tokens": regenerated.get("regenerate_input_tokens"),
                    "output_tokens": regenerated.get("regenerate_output_tokens"),
                    "total_tokens": regenerated.get("regenerate_total_tokens"),
                    "latency_seconds": regenerated.get("regenerate_latency_seconds"),
                })

                final_sentence = clean_sentence(regenerated["final_sentence"])
                sentence_confidence = safe_float(regenerated["confidence"], default=0.0)
                abstain_sentence = bool(regenerated["abstain_sentence"])
                support_status = clean_text(regenerated["support_status"])
                regeneration_parse_error = clean_text(regenerated.get("regenerate_parse_error", ""))
            else:
                final_sentence = candidate_sentence
                sentence_confidence = safe_float(candidate.get("confidence"), default=0.0)
                abstain_sentence = contains_any(normalize_for_rules(final_sentence), ABSTENTION_MARKERS)
                support_status = "not_checked"
                regeneration_parse_error = ""

            confidence_values.append(sentence_confidence)
            partial_answer = append_sentence(partial_answer, final_sentence)

            trace.append({
                "step": step,
                "candidate_sentence": candidate_sentence,
                "candidate_confidence": candidate.get("confidence"),
                "candidate_parse_error": clean_text(candidate.get("candidate_parse_error", "")),
                "needs_retrieval": bool(needs_ret),
                "retrieval_decision_reason": clean_text(decision.get("reason", "")),
                "retrieval_decision_confidence": decision.get("confidence"),
                "retrieval_policy_source": clean_text(decision.get("policy_source", "")),
                "retrieval_rule_name": clean_text(decision.get("rule_name", "")),
                "task_type": task_type,
                "retrieval_bias": retrieval_bias,
                "retrieval_skipped_reason": retrieval_skipped_reason,
                "retrieval_query": retrieval_query,
                "retrieved_chunk_ids": summarize_evidence_ids(retrieved_chunks),
                "retrieval_latency_seconds": retrieval_latency_seconds,
                "final_sentence": final_sentence,
                "sentence_confidence": sentence_confidence,
                "abstain_sentence": bool(abstain_sentence),
                "support_status": support_status,
                "regeneration_parse_error": regeneration_parse_error,
            })

            if should_stop_after_supported_retrieval(
                task_type=task_type,
                needs_retrieval=bool(needs_ret),
                final_sentence=final_sentence,
                abstain_sentence=abstain_sentence,
                support_status=support_status,
                sentence_confidence=sentence_confidence,
                retrieved_chunks=retrieved_chunks,
            ):
                break

            if should_stop_after_sentence(
                final_sentence=final_sentence,
                abstain_sentence=abstain_sentence,
                step=step,
                max_steps=max_steps,
            ):
                break

    except Exception as exc:
        error = str(exc)

    abstained = infer_final_abstention(
        answer=partial_answer,
        trace=trace,
    )

    if not partial_answer:
        partial_answer = "No hay información suficiente para responder de forma confiable."
        abstained = True

    return build_final_payload(
        answer=partial_answer,
        confidence_values=confidence_values,
        abstained=abstained,
        evidence_ids=summarize_evidence_ids(evidence_used),
        trace=trace,
        usage_items=usage_items,
        error=error,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test del controlador S3 FLARE-like para una pregunta."
    )

    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--top-k-per-step", type=int, default=DEFAULT_TOP_K_PER_STEP)
    parser.add_argument("--max-retrieval-steps", type=int, default=DEFAULT_MAX_RETRIEVAL_STEPS)
    parser.add_argument("--max-total-chunks", type=int, default=DEFAULT_MAX_TOTAL_CHUNKS)
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    parser.add_argument("--retrieval-strategy", choices=["rules", "llm", "hybrid"], default="rules")
    parser.add_argument(
        "--task-type",
        choices=["open_qa", "open_direct", "open_retrieval", "multiple_choice", "ambiguous"],
        default="open_qa",
    )
    parser.add_argument(
        "--retrieval-bias",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    result = run_flare_controller_for_question(
        question=args.question,
        index_dir=args.index_dir,
        model=args.model,
        max_steps=args.max_steps,
        top_k_per_step=args.top_k_per_step,
        max_retrieval_steps=args.max_retrieval_steps,
        max_total_chunks=args.max_total_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        retrieval_strategy=args.retrieval_strategy,
        task_type=args.task_type,
        retrieval_bias=args.retrieval_bias,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fire_controller_s4.py

Controlador principal para S4: FIRE-like / FIRE-inspired claim verification.

Objetivo
--------
Ejecutar el loop FIRE-like para UNA pregunta y UNA respuesta inicial.

Flujo:
    pregunta + respuesta inicial
        -> extraer claims
        -> para cada claim:
            -> verificar contra evidencia inicial
            -> si not_enough_info y needs_more_evidence:
                -> recuperar más chunks
                -> verificar de nuevo
            -> guardar trace claim-by-claim
        -> reparar/decidir respuesta final
        -> devolver JSON completo listo para run_s4_fire_like.py

Este archivo integra:
- claim_extractor_s4.py
- fire_verifier_s4.py
- retriever_s1.py / índice S2
- prompts_s4.py para reparación final opcional con LLM

No recorre CSV completo. Eso va en:
- run_s4_fire_like.py

Uso rápido
----------
Self-test sin API ni índice real:
python s4_model_code/fire_controller_s4.py --self-test

Ejemplo sin índice, con dry-run:
python s4_model_code/fire_controller_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --initial-answer "La traviata was composed by Giuseppe Verdi." \
  --dry-run

Ejemplo con índice real:
python s4_model_code/fire_controller_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --initial-answer "La traviata was composed by Giuseppe Verdi." \
  --index-dir indexes/s2/adaptive_rag \
  --claim-strategy rules \
  --verification-strategy rules \
  --repair-strategy rules
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Paths e imports del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S4_CODE_DIR = Path(__file__).resolve().parent
S2_CODE_DIR = PROJECT_ROOT / "s2_model_code"
S1_CODE_DIR = PROJECT_ROOT / "s1_model_code"

for path in [PROJECT_ROOT, S4_CODE_DIR, S2_CODE_DIR, S1_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from claim_extractor_s4 import extract_claims
    from fire_verifier_s4 import (
        build_search_query_rules,
        generate_search_query_llm,
        sanitize_chunks,
        verify_claim_once,
    )
    from prompts_s4 import (
        DEFAULT_MAX_CLAIMS,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        S4_FINAL_REPAIR_SYSTEM_PROMPT,
        build_final_repair_prompt,
    )
except ModuleNotFoundError:
    from s4_model_code.claim_extractor_s4 import extract_claims
    from s4_model_code.fire_verifier_s4 import (
        build_search_query_rules,
        generate_search_query_llm,
        sanitize_chunks,
        verify_claim_once,
    )
    from s4_model_code.prompts_s4 import (
        DEFAULT_MAX_CLAIMS,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        S4_FINAL_REPAIR_SYSTEM_PROMPT,
        build_final_repair_prompt,
    )


DEFAULT_INDEX_DIR = Path("indexes/s2/adaptive_rag")
DEFAULT_MAX_ROUNDS_PER_CLAIM = 2
DEFAULT_TOP_K_PER_ROUND = 2
DEFAULT_MAX_TOTAL_RETRIEVALS = 6
DEFAULT_MAX_TOTAL_CHUNKS = 10

ClaimStrategy = Literal["rules", "llm", "hybrid"]
VerificationStrategy = Literal["rules", "llm", "hybrid"]
RepairStrategy = Literal["rules", "llm", "none"]
QueryStrategy = Literal["rules", "llm", "hybrid"]

VALID_REPAIR_STRATEGIES = {"rules", "llm", "none"}
VALID_QUERY_STRATEGIES = {"rules", "llm", "hybrid"}

FINAL_DECISIONS = {
    "unchanged",
    "corrected",
    "abstained",
    "clarification_kept",
    "no_claims",
}


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
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


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def safe_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, indent=indent)


def unique_preserve_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


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


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def create_retriever(index_dir: Path = DEFAULT_INDEX_DIR) -> Any:
    """
    Crea el retriever usado por S1/S2/S3.

    Requiere que exista s1_model_code/retriever_s1.py y el índice:
    - chunks.csv
    - embeddings.npy
    - metadata.json
    """
    try:
        from retriever_s1 import S1Retriever
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se pudo importar retriever_s1.py. "
            "Ejecutá desde la raíz del proyecto y verificá s1_model_code/."
        ) from exc

    if not index_dir.exists():
        raise FileNotFoundError(f"No existe index_dir: {index_dir}")

    return S1Retriever(index_dir=index_dir)


def sanitize_chunk(item: dict[str, Any], *, fallback_rank: int = 1) -> dict[str, Any]:
    return {
        "rank": item.get("rank", fallback_rank),
        "chunk_id": clean_text(item.get("chunk_id", f"chunk_{fallback_rank}")),
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


def sanitize_chunk_list(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for i, item in enumerate(chunks or [], start=1):
        if isinstance(item, dict):
            safe = sanitize_chunk(item, fallback_rank=i)
            if safe["text"]:
                output.append(safe)
    return output


def get_chunk_id(chunk: dict[str, Any]) -> str:
    return clean_text(chunk.get("chunk_id", ""))


def merge_evidence_chunks(
    current_chunks: list[dict[str, Any]],
    new_chunks: list[dict[str, Any]],
    *,
    max_total_chunks: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in current_chunks + new_chunks:
        safe = sanitize_chunk(item, fallback_rank=len(merged) + 1)
        cid = get_chunk_id(safe) or f"__idx_{len(merged)}"
        if cid in seen:
            continue
        seen.add(cid)
        merged.append(safe)
        if len(merged) >= max_total_chunks:
            break

    return merged


def retrieve_chunks(
    *,
    retriever: Any,
    query: str,
    top_k: int,
    already_seen_chunk_ids: set[str] | None = None,
    max_total_chunks_remaining: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    if retriever is None:
        return [], 0.0

    if top_k <= 0:
        return [], 0.0

    already_seen_chunk_ids = already_seen_chunk_ids or set()
    effective_top_k = max(top_k, top_k + len(already_seen_chunk_ids))

    start = time.time()
    raw_chunks = retriever.retrieve(query, top_k=effective_top_k)
    latency = time.time() - start

    selected: list[dict[str, Any]] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue

        safe = sanitize_chunk(item, fallback_rank=len(selected) + 1)
        cid = get_chunk_id(safe)

        if cid and cid in already_seen_chunk_ids:
            continue

        selected.append(safe)

        if len(selected) >= top_k:
            break

    if max_total_chunks_remaining is not None and max_total_chunks_remaining >= 0:
        selected = selected[:max_total_chunks_remaining]

    return selected, round(latency, 3)


# ---------------------------------------------------------------------------
# LLM call for final repair
# ---------------------------------------------------------------------------

def import_llm_client():
    try:
        from direct_llm import ask_direct_llm_with_metadata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se pudo importar direct_llm.py. Ejecutá desde la raíz del proyecto."
        ) from exc
    return ask_direct_llm_with_metadata


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

    ask_direct_llm_with_metadata = import_llm_client()

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


# ---------------------------------------------------------------------------
# Reparación / decisión final
# ---------------------------------------------------------------------------

def normalize_final_decision(value: Any) -> str:
    decision = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    if decision in FINAL_DECISIONS:
        return decision
    return "corrected"


def collect_evidence_ids_from_claim_results(claim_results: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for result in claim_results:
        ids.extend(result.get("supporting_chunk_ids", []) or [])
        ids.extend(result.get("refuting_chunk_ids", []) or [])
        for round_item in result.get("rounds_trace", []) or []:
            ids.extend(round_item.get("retrieved_chunk_ids", []) or [])
            ids.extend(round_item.get("supporting_chunk_ids", []) or [])
            ids.extend(round_item.get("refuting_chunk_ids", []) or [])
    return unique_preserve_order(ids)


def summarize_claim_verdicts(claim_results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [clean_text(item.get("verdict", "")) for item in claim_results]
    claim_types = [clean_text(item.get("claim_type", "")) for item in claim_results]

    n_claims = len(claim_results)
    n_supported = sum(v == "supported" for v in verdicts)
    n_refuted = sum(v == "refuted" for v in verdicts)
    n_nei = sum(v == "not_enough_info" for v in verdicts)
    n_abstention = sum(t == "abstention" for t in claim_types)
    n_clarification = sum(t == "clarification" for t in claim_types)
    n_answer_choice = sum(t == "answer_choice" for t in claim_types)
    n_requires_evidence = sum(bool(item.get("requires_evidence")) for item in claim_results)

    core_results = [
        item for item in claim_results
        if clean_text(item.get("importance", "core")) == "core"
    ] or claim_results

    core_verdicts = [clean_text(item.get("verdict", "")) for item in core_results]
    core_types = [clean_text(item.get("claim_type", "")) for item in core_results]
    core_refuted = any(v == "refuted" for v in core_verdicts)
    core_nei = any(v == "not_enough_info" for v in core_verdicts)
    core_supported_all = bool(core_results) and all(v == "supported" for v in core_verdicts)
    core_is_abstention = bool(core_results) and all(t == "abstention" for t in core_types)
    core_is_clarification = bool(core_results) and all(t == "clarification" for t in core_types)

    return {
        "num_claims": int(n_claims),
        "num_supported_claims": int(n_supported),
        "num_refuted_claims": int(n_refuted),
        "num_nei_claims": int(n_nei),
        "num_abstention_claims": int(n_abstention),
        "num_clarification_claims": int(n_clarification),
        "num_answer_choice_claims": int(n_answer_choice),
        "num_claims_requiring_evidence": int(n_requires_evidence),
        "core_refuted": bool(core_refuted),
        "core_nei": bool(core_nei),
        "core_supported_all": bool(core_supported_all),
        "core_is_abstention": bool(core_is_abstention),
        "core_is_clarification": bool(core_is_clarification),
    }


def repair_final_answer_rules(
    *,
    question: str,
    initial_answer: str,
    claim_results: list[dict[str, Any]],
    expected_behavior: str = "",
) -> dict[str, Any]:
    summary = summarize_claim_verdicts(claim_results)
    evidence_ids = collect_evidence_ids_from_claim_results(claim_results)

    if summary["num_claims"] == 0:
        return {
            "answer": "No hay información suficiente para construir una respuesta verificada.",
            "confidence": 0.70,
            "abstained": True,
            "final_decision": "no_claims",
            "correction_applied": False,
            "unsupported_claims_removed": [],
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    if summary["core_is_clarification"]:
        return {
            "answer": initial_answer,
            "confidence": 0.95,
            "abstained": True,
            "final_decision": "clarification_kept",
            "correction_applied": False,
            "unsupported_claims_removed": [],
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    if summary["core_is_abstention"]:
        return {
            "answer": initial_answer,
            "confidence": 0.95,
            "abstained": True,
            "final_decision": "abstained",
            "correction_applied": False,
            "unsupported_claims_removed": [],
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    refuted_claim_ids = [
        item["claim_id"] for item in claim_results
        if clean_text(item.get("verdict", "")) == "refuted"
    ]
    nei_claim_ids = [
        item["claim_id"] for item in claim_results
        if clean_text(item.get("verdict", "")) == "not_enough_info"
    ]

    if summary["core_refuted"]:
        return {
            "answer": (
                "La respuesta inicial contiene una afirmación contradicha por la evidencia recuperada, "
                "por lo que no puedo mantenerla como respuesta confiable."
            ),
            "confidence": 0.85,
            "abstained": True,
            "final_decision": "abstained",
            "correction_applied": True,
            "unsupported_claims_removed": refuted_claim_ids,
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    if summary["core_nei"]:
        return {
            "answer": (
                "No hay información suficiente en la evidencia recuperada para verificar la respuesta inicial."
            ),
            "confidence": 0.80,
            "abstained": True,
            "final_decision": "abstained",
            "correction_applied": True,
            "unsupported_claims_removed": nei_claim_ids,
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    if summary["core_supported_all"]:
        return {
            "answer": initial_answer,
            "confidence": 0.90,
            "abstained": False,
            "final_decision": "unchanged",
            "correction_applied": False,
            "unsupported_claims_removed": [],
            "corrected_claims": [],
            "evidence_ids": evidence_ids,
        }

    # Fallback conservador.
    return {
        "answer": initial_answer,
        "confidence": 0.65,
        "abstained": False,
        "final_decision": "unchanged",
        "correction_applied": False,
        "unsupported_claims_removed": [],
        "corrected_claims": [],
        "evidence_ids": evidence_ids,
    }


def validate_repair_payload(
    obj: dict[str, Any] | None,
    *,
    fallback_answer: str,
    claim_results: list[dict[str, Any]],
) -> dict[str, Any]:
    obj = obj or {}
    evidence_ids = obj.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = collect_evidence_ids_from_claim_results(claim_results)

    unsupported_claims_removed = obj.get("unsupported_claims_removed", [])
    if not isinstance(unsupported_claims_removed, list):
        unsupported_claims_removed = []

    corrected_claims = obj.get("corrected_claims", [])
    if not isinstance(corrected_claims, list):
        corrected_claims = []

    answer = clean_text(obj.get("answer", "")) or fallback_answer
    final_decision = normalize_final_decision(obj.get("final_decision", "corrected"))
    abstained = coerce_bool(obj.get("abstained"), default=final_decision in {"abstained", "clarification_kept", "no_claims"})
    correction_applied = coerce_bool(obj.get("correction_applied"), default=final_decision in {"corrected", "abstained"})
    confidence = safe_float(obj.get("confidence", 0.0), default=0.0)

    return {
        "answer": answer,
        "confidence": confidence,
        "abstained": bool(abstained),
        "final_decision": final_decision,
        "correction_applied": bool(correction_applied),
        "unsupported_claims_removed": [clean_text(x) for x in unsupported_claims_removed if clean_text(x)],
        "corrected_claims": [clean_text(x) for x in corrected_claims if clean_text(x)],
        "evidence_ids": unique_preserve_order(evidence_ids),
    }


def repair_final_answer_llm(
    *,
    question: str,
    initial_answer: str,
    claim_results: list[dict[str, Any]],
    expected_behavior: str = "",
    source_system: str = "",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    prompt = build_final_repair_prompt(
        question=question,
        initial_answer=initial_answer,
        claim_results=claim_results,
        expected_behavior=expected_behavior,
        source_system=source_system,
    )

    dry_payload = repair_final_answer_rules(
        question=question,
        initial_answer=initial_answer,
        claim_results=claim_results,
        expected_behavior=expected_behavior,
    )

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S4_FINAL_REPAIR_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    validated = validate_repair_payload(
        call["parsed"],
        fallback_answer=initial_answer,
        claim_results=claim_results,
    )

    return {
        **validated,
        "repair_strategy": "llm",
        "repair_model": call["model"],
        "repair_raw_output": call["raw_output"],
        "repair_parse_error": call["parse_error"],
        "repair_usage_json": call["usage_json"],
        "repair_input_tokens": call["input_tokens"],
        "repair_output_tokens": call["output_tokens"],
        "repair_total_tokens": call["total_tokens"],
        "repair_latency_seconds": call["latency_seconds"],
        "repair_dry_run": call["dry_run"],
    }


def repair_final_answer(
    *,
    question: str,
    initial_answer: str,
    claim_results: list[dict[str, Any]],
    expected_behavior: str = "",
    source_system: str = "",
    repair_strategy: RepairStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    repair_strategy = clean_text(repair_strategy).lower()
    if repair_strategy not in VALID_REPAIR_STRATEGIES:
        raise ValueError(
            f"repair_strategy inválida: {repair_strategy!r}. "
            f"Opciones válidas: {sorted(VALID_REPAIR_STRATEGIES)}"
        )

    start = time.time()

    if repair_strategy == "none":
        payload = {
            "answer": initial_answer,
            "confidence": 0.0,
            "abstained": False,
            "final_decision": "unchanged",
            "correction_applied": False,
            "unsupported_claims_removed": [],
            "corrected_claims": [],
            "evidence_ids": collect_evidence_ids_from_claim_results(claim_results),
        }
        payload.update(
            {
                "repair_strategy": "none",
                "repair_model": "",
                "repair_raw_output": "",
                "repair_parse_error": "",
                "repair_usage_json": "{}",
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                "repair_total_tokens": 0,
                "repair_latency_seconds": round(time.time() - start, 3),
                "repair_dry_run": False,
            }
        )
        return payload

    if repair_strategy == "rules":
        payload = repair_final_answer_rules(
            question=question,
            initial_answer=initial_answer,
            claim_results=claim_results,
            expected_behavior=expected_behavior,
        )
        payload.update(
            {
                "repair_strategy": "rules",
                "repair_model": "",
                "repair_raw_output": "",
                "repair_parse_error": "",
                "repair_usage_json": "{}",
                "repair_input_tokens": 0,
                "repair_output_tokens": 0,
                "repair_total_tokens": 0,
                "repair_latency_seconds": round(time.time() - start, 3),
                "repair_dry_run": False,
            }
        )
        return payload

    return repair_final_answer_llm(
        question=question,
        initial_answer=initial_answer,
        claim_results=claim_results,
        expected_behavior=expected_behavior,
        source_system=source_system,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# FIRE loop por claim
# ---------------------------------------------------------------------------

def choose_next_search_query(
    *,
    question: str,
    claim: dict[str, Any],
    verification: dict[str, Any],
    evidence_so_far: list[dict[str, Any]],
    query_strategy: QueryStrategy,
    model: str | None,
    max_retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    query_strategy = clean_text(query_strategy).lower()
    if query_strategy not in VALID_QUERY_STRATEGIES:
        raise ValueError(
            f"query_strategy inválida: {query_strategy!r}. "
            f"Opciones válidas: {sorted(VALID_QUERY_STRATEGIES)}"
        )

    existing_query = clean_text(verification.get("next_search_query", ""))
    if existing_query and query_strategy in {"rules", "hybrid"}:
        return {
            "search_query": existing_query,
            "reason": "Query tomada del verificador.",
            "query_strategy": f"{query_strategy}_from_verifier",
            "query_model": "",
            "query_raw_output": "",
            "query_parse_error": "",
            "query_total_tokens": 0,
            "query_latency_seconds": 0.0,
            "query_dry_run": False,
        }

    if query_strategy == "rules":
        return {
            "search_query": build_search_query_rules(question, claim.get("claim_text", "")),
            "reason": "Query generada por reglas desde pregunta y claim.",
            "query_strategy": "rules",
            "query_model": "",
            "query_raw_output": "",
            "query_parse_error": "",
            "query_total_tokens": 0,
            "query_latency_seconds": 0.0,
            "query_dry_run": False,
        }

    query_result = generate_search_query_llm(
        question=question,
        claim=claim,
        previous_evidence=evidence_so_far,
        previous_verdict=verification.get("verdict", ""),
        previous_rationale=verification.get("rationale", ""),
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
    )
    query_result["query_strategy"] = "llm" if query_strategy == "llm" else "hybrid_llm"
    return query_result


def verify_single_claim_iteratively(
    *,
    question: str,
    claim: dict[str, Any],
    retriever: Any | None,
    initial_evidence: list[dict[str, Any]] | None,
    claim_strategy: ClaimStrategy = "rules",
    verification_strategy: VerificationStrategy = "rules",
    query_strategy: QueryStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
    max_rounds_per_claim: int = DEFAULT_MAX_ROUNDS_PER_CLAIM,
    top_k_per_round: int = DEFAULT_TOP_K_PER_ROUND,
    max_total_retrievals: int = DEFAULT_MAX_TOTAL_RETRIEVALS,
    max_total_chunks: int = DEFAULT_MAX_TOTAL_CHUNKS,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    global_seen_chunk_ids: set[str] | None = None,
    global_retrieval_counter: int = 0,
) -> tuple[dict[str, Any], int, float]:
    """
    Devuelve:
    - claim_result
    - nuevo global_retrieval_counter
    - retrieval_latency_total
    """
    claim = dict(claim)
    evidence_set = sanitize_chunk_list(initial_evidence)
    rounds_trace: list[dict[str, Any]] = []
    previous_verifications: list[dict[str, Any]] = []
    retrieval_latency_total = 0.0

    global_seen_chunk_ids = global_seen_chunk_ids or set()
    local_seen_chunk_ids = {get_chunk_id(c) for c in evidence_set if get_chunk_id(c)}

    final_verification: dict[str, Any] | None = None

    for round_id in range(1, max_rounds_per_claim + 1):
        verification = verify_claim_once(
            question=question,
            claim=claim,
            retrieved_chunks=evidence_set,
            round_id=round_id,
            max_rounds=max_rounds_per_claim,
            previous_verifications=previous_verifications,
            strategy=verification_strategy,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
            max_chars_per_chunk=max_chars_per_chunk,
        )

        final_verification = verification
        previous_verifications.append(
            {
                "round_id": round_id,
                "verdict": verification.get("verdict", ""),
                "confidence": verification.get("confidence", ""),
                "rationale": verification.get("rationale", ""),
                "needs_more_evidence": verification.get("needs_more_evidence", False),
                "next_search_query": verification.get("next_search_query", None),
                "supporting_chunk_ids": verification.get("supporting_chunk_ids", []),
                "refuting_chunk_ids": verification.get("refuting_chunk_ids", []),
            }
        )

        trace_item = {
            "round_id": round_id,
            "verdict": verification.get("verdict", ""),
            "confidence": verification.get("confidence", 0.0),
            "rationale": verification.get("rationale", ""),
            "needs_more_evidence": verification.get("needs_more_evidence", False),
            "next_search_query": verification.get("next_search_query", None),
            "supporting_chunk_ids": verification.get("supporting_chunk_ids", []),
            "refuting_chunk_ids": verification.get("refuting_chunk_ids", []),
            "evidence_ids_before_retrieval": unique_preserve_order([get_chunk_id(c) for c in evidence_set]),
            "retrieval_query": None,
            "retrieved_chunk_ids": [],
            "retrieval_latency_seconds": 0.0,
        }

        verdict = clean_text(verification.get("verdict", ""))
        needs_more = coerce_bool(verification.get("needs_more_evidence"), default=False)

        if verdict in {"supported", "refuted"}:
            rounds_trace.append(trace_item)
            break

        if not needs_more:
            rounds_trace.append(trace_item)
            break

        if round_id >= max_rounds_per_claim:
            rounds_trace.append(trace_item)
            break

        if global_retrieval_counter >= max_total_retrievals:
            trace_item["needs_more_evidence"] = False
            trace_item["retrieval_blocked_reason"] = "max_total_retrievals_reached"
            rounds_trace.append(trace_item)
            break

        remaining_chunks = max_total_chunks - len(evidence_set)
        if remaining_chunks <= 0:
            trace_item["needs_more_evidence"] = False
            trace_item["retrieval_blocked_reason"] = "max_total_chunks_reached"
            rounds_trace.append(trace_item)
            break

        query_result = choose_next_search_query(
            question=question,
            claim=claim,
            verification=verification,
            evidence_so_far=evidence_set,
            query_strategy=query_strategy,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
        )

        retrieval_query = clean_text(query_result.get("search_query", ""))

        if not retrieval_query:
            retrieval_query = build_search_query_rules(question, claim.get("claim_text", ""))

        already_seen = set(global_seen_chunk_ids) | set(local_seen_chunk_ids)
        new_chunks, retrieval_latency = retrieve_chunks(
            retriever=retriever,
            query=retrieval_query,
            top_k=top_k_per_round,
            already_seen_chunk_ids=already_seen,
            max_total_chunks_remaining=remaining_chunks,
        )

        global_retrieval_counter += 1
        retrieval_latency_total += retrieval_latency

        for chunk in new_chunks:
            cid = get_chunk_id(chunk)
            if cid:
                global_seen_chunk_ids.add(cid)
                local_seen_chunk_ids.add(cid)

        evidence_set = merge_evidence_chunks(
            evidence_set,
            new_chunks,
            max_total_chunks=max_total_chunks,
        )

        trace_item.update(
            {
                "retrieval_query": retrieval_query,
                "retrieved_chunk_ids": unique_preserve_order([get_chunk_id(c) for c in new_chunks]),
                "retrieval_latency_seconds": retrieval_latency,
                "query_strategy": query_result.get("query_strategy", query_strategy),
                "query_reason": query_result.get("reason", ""),
                "query_model": query_result.get("query_model", ""),
                "query_parse_error": query_result.get("query_parse_error", ""),
                "query_total_tokens": query_result.get("query_total_tokens", 0),
                "query_latency_seconds": query_result.get("query_latency_seconds", 0.0),
            }
        )

        rounds_trace.append(trace_item)

    if final_verification is None:
        final_verification = {
            "claim_id": claim.get("claim_id", ""),
            "claim_text": claim.get("claim_text", ""),
            "verdict": "not_enough_info",
            "confidence": 0.0,
            "rationale": "No se ejecutó ninguna ronda de verificación.",
            "supporting_chunk_ids": [],
            "refuting_chunk_ids": [],
            "needs_more_evidence": False,
            "next_search_query": None,
        }

    evidence_ids = unique_preserve_order([get_chunk_id(c) for c in evidence_set])
    retrieved_ids_from_trace: list[str] = []
    for item in rounds_trace:
        retrieved_ids_from_trace.extend(item.get("retrieved_chunk_ids", []) or [])

    claim_result = {
        "claim_id": claim.get("claim_id", ""),
        "claim_text": claim.get("claim_text", ""),
        "claim_type": claim.get("claim_type", "factual"),
        "requires_evidence": bool(claim.get("requires_evidence", True)),
        "importance": claim.get("importance", "core"),
        "verdict": final_verification.get("verdict", "not_enough_info"),
        "confidence": safe_float(final_verification.get("confidence", 0.0), default=0.0),
        "rationale": final_verification.get("rationale", ""),
        "supporting_chunk_ids": final_verification.get("supporting_chunk_ids", []),
        "refuting_chunk_ids": final_verification.get("refuting_chunk_ids", []),
        "needs_more_evidence": final_verification.get("needs_more_evidence", False),
        "next_search_query": final_verification.get("next_search_query", None),
        "rounds": len(rounds_trace),
        "evidence_ids": evidence_ids,
        "retrieved_chunk_ids": unique_preserve_order(retrieved_ids_from_trace),
        "rounds_trace": rounds_trace,
        "verifier_strategy": final_verification.get("verifier_strategy", verification_strategy),
        "verifier_model": final_verification.get("verifier_model", ""),
        "verifier_parse_error": final_verification.get("verifier_parse_error", ""),
        "verifier_total_tokens": sum(
            safe_int(item.get("verifier_total_tokens", 0), default=0)
            for item in previous_verifications
        ),
    }

    return claim_result, global_retrieval_counter, round(retrieval_latency_total, 3)


# ---------------------------------------------------------------------------
# Controller principal
# ---------------------------------------------------------------------------

def run_fire_controller_for_answer(
    *,
    question: str,
    initial_answer: str,
    source_system: str = "s2",
    question_type: str = "",
    expected_behavior: str = "",
    initial_evidence: list[dict[str, Any]] | None = None,
    retriever: Any | None = None,
    claim_strategy: ClaimStrategy = "rules",
    verification_strategy: VerificationStrategy = "rules",
    query_strategy: QueryStrategy = "rules",
    repair_strategy: RepairStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    max_rounds_per_claim: int = DEFAULT_MAX_ROUNDS_PER_CLAIM,
    top_k_per_round: int = DEFAULT_TOP_K_PER_ROUND,
    max_total_retrievals: int = DEFAULT_MAX_TOTAL_RETRIEVALS,
    max_total_chunks: int = DEFAULT_MAX_TOTAL_CHUNKS,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> dict[str, Any]:
    start = time.time()
    question = clean_text(question)
    initial_answer = clean_text(initial_answer)

    if not question:
        raise ValueError("question vacío.")
    if not initial_answer:
        raise ValueError("initial_answer vacío.")

    initial_evidence = sanitize_chunk_list(initial_evidence)

    extraction = extract_claims(
        question=question,
        initial_answer=initial_answer,
        source_system=source_system,
        question_type=question_type,
        expected_behavior=expected_behavior,
        max_claims=max_claims,
        strategy=claim_strategy,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
    )

    claims = extraction.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    claim_results: list[dict[str, Any]] = []
    global_seen_chunk_ids = {get_chunk_id(c) for c in initial_evidence if get_chunk_id(c)}
    global_retrieval_counter = 0
    retrieval_latency_total = 0.0

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        claim_result, global_retrieval_counter, claim_retrieval_latency = verify_single_claim_iteratively(
            question=question,
            claim=claim,
            retriever=retriever,
            initial_evidence=initial_evidence,
            claim_strategy=claim_strategy,
            verification_strategy=verification_strategy,
            query_strategy=query_strategy,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
            max_rounds_per_claim=max_rounds_per_claim,
            top_k_per_round=top_k_per_round,
            max_total_retrievals=max_total_retrievals,
            max_total_chunks=max_total_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
            global_seen_chunk_ids=global_seen_chunk_ids,
            global_retrieval_counter=global_retrieval_counter,
        )

        retrieval_latency_total += claim_retrieval_latency
        claim_results.append(claim_result)

    repair = repair_final_answer(
        question=question,
        initial_answer=initial_answer,
        claim_results=claim_results,
        expected_behavior=expected_behavior,
        source_system=source_system,
        repair_strategy=repair_strategy,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
    )

    verdict_summary = summarize_claim_verdicts(claim_results)

    total_tokens = (
        safe_int(extraction.get("extractor_total_tokens", 0), default=0)
        + sum(safe_int(item.get("verifier_total_tokens", 0), default=0) for item in claim_results)
        + safe_int(repair.get("repair_total_tokens", 0), default=0)
    )

    evidence_ids = unique_preserve_order(
        collect_evidence_ids_from_claim_results(claim_results)
        + repair.get("evidence_ids", [])
    )

    retrieved_chunk_ids: list[str] = []
    for item in claim_results:
        retrieved_chunk_ids.extend(item.get("retrieved_chunk_ids", []) or [])

    result = {
        "initial_answer": initial_answer,
        "answer": repair.get("answer", initial_answer),
        "confidence": safe_float(repair.get("confidence", 0.0), default=0.0),
        "abstained": bool(repair.get("abstained", False)),
        "final_decision": repair.get("final_decision", "unchanged"),
        "correction_applied": bool(repair.get("correction_applied", False)),
        "unsupported_claims_removed": repair.get("unsupported_claims_removed", []),
        "corrected_claims": repair.get("corrected_claims", []),
        "retrieval_mode": "iterative_verification",
        "source_system": source_system,
        "num_claims": verdict_summary["num_claims"],
        "num_supported_claims": verdict_summary["num_supported_claims"],
        "num_refuted_claims": verdict_summary["num_refuted_claims"],
        "num_nei_claims": verdict_summary["num_nei_claims"],
        "num_claims_requiring_evidence": verdict_summary["num_claims_requiring_evidence"],
        "num_verification_rounds": int(sum(safe_int(item.get("rounds", 0), default=0) for item in claim_results)),
        "num_retrieval_rounds": int(global_retrieval_counter),
        "num_chunks_retrieved_total": int(len(unique_preserve_order(retrieved_chunk_ids))),
        "evidence_ids": evidence_ids,
        "retrieved_chunk_ids": unique_preserve_order(retrieved_chunk_ids),
        "claims": claims,
        "claim_results": claim_results,
        "claim_trace": claim_results,
        "extraction": extraction,
        "repair": repair,
        "claim_strategy": claim_strategy,
        "verification_strategy": verification_strategy,
        "query_strategy": query_strategy,
        "repair_strategy": repair_strategy,
        "input_tokens": safe_int(extraction.get("extractor_input_tokens", 0), default=0) + safe_int(repair.get("repair_input_tokens", 0), default=0),
        "output_tokens": safe_int(extraction.get("extractor_output_tokens", 0), default=0) + safe_int(repair.get("repair_output_tokens", 0), default=0),
        "total_tokens": int(total_tokens),
        "retrieval_latency_seconds": round(retrieval_latency_total, 3),
        "latency_seconds": round(time.time() - start, 3),
        "error": "",
    }

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

class FakeRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        return self.chunks[:top_k]


def run_self_test() -> None:
    fake_chunks = [
        {
            "rank": 1,
            "chunk_id": "chunk_test_001",
            "doc_id": "doc_test",
            "title": "La traviata",
            "score": 0.91,
            "text": "La traviata is an opera in three acts by Giuseppe Verdi.",
            "source": "self_test",
            "topic": "opera",
        }
    ]
    fake_retriever = FakeRetriever(fake_chunks)

    examples = [
        {
            "name": "factual_with_retrieval",
            "question": "According to the available corpus, who composed La traviata?",
            "initial_answer": "La traviata was composed by Giuseppe Verdi.",
            "source_system": "s2",
            "question_type": "rag_multi_hop",
            "expected_behavior": "answer",
            "initial_evidence": [],
            "retriever": fake_retriever,
        },
        {
            "name": "already_abstained",
            "question": "According to the available corpus, what is the unpublished access code associated with La traviata?",
            "initial_answer": "No hay información suficiente para responder con la evidencia disponible.",
            "source_system": "s2",
            "question_type": "rag_no_answer",
            "expected_behavior": "abstain",
            "initial_evidence": [],
            "retriever": fake_retriever,
        },
        {
            "name": "already_clarification",
            "question": "When was he born?",
            "initial_answer": "Necesito una aclaración mínima para saber a qué persona te referís.",
            "source_system": "s2",
            "question_type": "ambiguous",
            "expected_behavior": "clarify",
            "initial_evidence": [],
            "retriever": fake_retriever,
        },
    ]

    print("S4 FIRE controller self-test")
    print("=" * 72)

    for ex in examples:
        result = run_fire_controller_for_answer(
            question=ex["question"],
            initial_answer=ex["initial_answer"],
            source_system=ex["source_system"],
            question_type=ex["question_type"],
            expected_behavior=ex["expected_behavior"],
            initial_evidence=ex["initial_evidence"],
            retriever=ex["retriever"],
            claim_strategy="rules",
            verification_strategy="rules",
            query_strategy="rules",
            repair_strategy="rules",
            dry_run=True,
            max_claims=5,
            max_rounds_per_claim=2,
            top_k_per_round=2,
            max_total_retrievals=4,
            max_total_chunks=6,
        )

        preview = {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "abstained": result["abstained"],
            "final_decision": result["final_decision"],
            "num_claims": result["num_claims"],
            "num_supported_claims": result["num_supported_claims"],
            "num_refuted_claims": result["num_refuted_claims"],
            "num_nei_claims": result["num_nei_claims"],
            "num_verification_rounds": result["num_verification_rounds"],
            "num_retrieval_rounds": result["num_retrieval_rounds"],
            "evidence_ids": result["evidence_ids"],
            "retrieved_chunk_ids": result["retrieved_chunk_ids"],
            "claim_results": result["claim_results"],
            "error": result["error"],
        }

        print(f"Example: {ex['name']}")
        print(safe_json_dumps(preview, indent=2))
        print("-" * 72)

    print("S4 FIRE controller self-test OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_initial_evidence_json(value: str) -> list[dict[str, Any]]:
    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"initial evidence JSON inválido: {exc}") from exc

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError("La evidencia inicial debe ser un objeto o lista de objetos JSON.")

    chunks: list[dict[str, Any]] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError("Cada chunk de evidencia inicial debe ser un objeto/dict.")
        chunks.append(sanitize_chunk(item, fallback_rank=i))

    return chunks


def parse_initial_evidence_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.initial_evidence_json:
        return parse_initial_evidence_json(args.initial_evidence_json)

    if args.initial_evidence_path:
        path = Path(args.initial_evidence_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe --initial-evidence-path: {path}")
        return parse_initial_evidence_json(path.read_text(encoding="utf-8"))

    if args.initial_evidence_text:
        return [
            {
                "rank": 1,
                "chunk_id": args.initial_evidence_chunk_id or "chunk_cli_initial_001",
                "doc_id": "doc_cli",
                "title": args.initial_evidence_title or "Initial evidence",
                "score": 1.0,
                "text": args.initial_evidence_text,
                "source": "cli",
                "topic": "",
            }
        ]

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlador FIRE-like para una pregunta/respuesta inicial."
    )
    parser.add_argument("--question", type=str, default="", help="Pregunta original.")
    parser.add_argument("--initial-answer", type=str, default="", help="Respuesta inicial de S2/S3.")
    parser.add_argument("--source-system", type=str, default="s2", help="Sistema fuente: s2, s3, etc.")
    parser.add_argument("--question-type", type=str, default="", help="Tipo de caso/pregunta.")
    parser.add_argument("--expected-behavior", type=str, default="", help="answer/abstain/clarify si está disponible.")

    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Directorio del índice S2.")
    parser.add_argument("--use-index", action="store_true", help="Usa retriever real desde --index-dir.")
    parser.add_argument("--dry-run", action="store_true", help="No llama API. Puede usar fake/no retriever salvo --use-index.")

    parser.add_argument("--initial-evidence-text", type=str, default="", help="Chunk inicial inline.")
    parser.add_argument("--initial-evidence-title", type=str, default="", help="Título del chunk inicial inline.")
    parser.add_argument("--initial-evidence-chunk-id", type=str, default="", help="chunk_id del chunk inicial inline.")
    parser.add_argument("--initial-evidence-json", type=str, default="", help="Objeto/lista JSON de chunks iniciales.")
    parser.add_argument("--initial-evidence-path", type=str, default="", help="Archivo JSON con evidencia inicial.")

    parser.add_argument("--claim-strategy", choices=["rules", "llm", "hybrid"], default="rules")
    parser.add_argument("--verification-strategy", choices=["rules", "llm", "hybrid"], default="rules")
    parser.add_argument("--query-strategy", choices=["rules", "llm", "hybrid"], default="rules")
    parser.add_argument("--repair-strategy", choices=["rules", "llm", "none"], default="rules")
    parser.add_argument("--model", type=str, default=None, help="Modelo LLM.")
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS)
    parser.add_argument("--max-rounds-per-claim", type=int, default=DEFAULT_MAX_ROUNDS_PER_CLAIM)
    parser.add_argument("--top-k-per-round", type=int, default=DEFAULT_TOP_K_PER_ROUND)
    parser.add_argument("--max-total-retrievals", type=int, default=DEFAULT_MAX_TOTAL_RETRIEVALS)
    parser.add_argument("--max-total-chunks", type=int, default=DEFAULT_MAX_TOTAL_CHUNKS)
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)

    parser.add_argument("--output-path", type=Path, default=None, help="Opcional: guarda JSON en archivo.")
    parser.add_argument("--self-test", action="store_true", help="Ejecuta self-test con FakeRetriever.")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not clean_text(args.question) or not clean_text(args.initial_answer):
        raise ValueError(
            "Debés pasar --question y --initial-answer, o usar --self-test."
        )

    initial_evidence = parse_initial_evidence_from_args(args)

    retriever = None
    if args.use_index:
        retriever = create_retriever(args.index_dir)

    result = run_fire_controller_for_answer(
        question=args.question,
        initial_answer=args.initial_answer,
        source_system=args.source_system,
        question_type=args.question_type,
        expected_behavior=args.expected_behavior,
        initial_evidence=initial_evidence,
        retriever=retriever,
        claim_strategy=args.claim_strategy,
        verification_strategy=args.verification_strategy,
        query_strategy=args.query_strategy,
        repair_strategy=args.repair_strategy,
        model=args.model,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        max_claims=args.max_claims,
        max_rounds_per_claim=args.max_rounds_per_claim,
        top_k_per_round=args.top_k_per_round,
        max_total_retrievals=args.max_total_retrievals,
        max_total_chunks=args.max_total_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )

    output = safe_json_dumps(result, indent=2)

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(output, encoding="utf-8")
        print(f"Resultado guardado en: {args.output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()

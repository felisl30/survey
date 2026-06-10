#!/usr/bin/env python3
"""
fire_verifier_s4.py

Verificador de claims para S4: FIRE-like / FIRE-inspired claim verification.

Objetivo
--------
Verificar UN claim contra una lista de chunks de evidencia recuperada.

Este archivo:
- No extrae claims. Eso lo hace claim_extractor_s4.py.
- No ejecuta el loop completo FIRE. Eso lo hará fire_controller_s4.py.
- No repara la respuesta final. Eso lo hará el módulo de reparación/controlador.
- Sí decide si un claim está:
    supported
    refuted
    not_enough_info

Estrategias:
- rules: verificador determinístico barato para smoke tests.
- llm: verificador con LLM usando prompts_s4.py.
- hybrid: reglas para casos triviales y LLM para claims factuales.

Uso rápido
----------
Self-test sin API:
python s4_model_code/fire_verifier_s4.py --self-test

Ejemplo por reglas con evidencia textual:
python s4_model_code/fire_verifier_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --claim "La traviata was composed by Giuseppe Verdi." \
  --evidence-text "La traviata is an opera in three acts by Giuseppe Verdi." \
  --strategy rules

Ejemplo LLM sin gastar API:
python s4_model_code/fire_verifier_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --claim "La traviata was composed by Giuseppe Verdi." \
  --evidence-text "La traviata is an opera in three acts by Giuseppe Verdi." \
  --strategy llm \
  --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Paths e imports del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S4_CODE_DIR = Path(__file__).resolve().parent

for path in [PROJECT_ROOT, S4_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from prompts_s4 import (
        DEFAULT_MAX_CHARS_PER_CHUNK,
        S4_CLAIM_VERIFICATION_SYSTEM_PROMPT,
        S4_SEARCH_QUERY_SYSTEM_PROMPT,
        VALID_VERDICTS,
        build_claim_verification_prompt,
        build_search_query_prompt,
    )
except ModuleNotFoundError:
    from s4_model_code.prompts_s4 import (
        DEFAULT_MAX_CHARS_PER_CHUNK,
        S4_CLAIM_VERIFICATION_SYSTEM_PROMPT,
        S4_SEARCH_QUERY_SYSTEM_PROMPT,
        VALID_VERDICTS,
        build_claim_verification_prompt,
        build_search_query_prompt,
    )


VerificationStrategy = Literal["rules", "llm", "hybrid"]
VALID_VERIFICATION_STRATEGIES = {"rules", "llm", "hybrid"}

DEFAULT_TOP_K_HINT = 2
DEFAULT_MAX_ROUNDS = 2


STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "it", "its", "as", "at", "from", "but", "not", "does", "do", "did", "can",
    "could", "would", "should", "will", "may", "might", "have", "has", "had",
    "if", "then", "than", "into", "about", "which", "who", "what", "where",
    "when", "why", "how", "also", "only",
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de",
    "del", "en", "con", "por", "para", "es", "son", "fue", "eran", "ser",
    "que", "este", "esta", "esto", "como", "hay", "si", "no", "se", "su",
    "sus", "tambien", "también",
}

NEGATION_MARKERS = {
    "not", "no", "never", "neither", "nor", "without",
    "false", "incorrect", "did not", "does not", "is not", "was not",
    "nunca", "sin", "falso", "incorrecto", "incorrecta",
    "no fue", "no es", "no era",
}

NON_FACTUAL_CLAIM_TYPES = {"abstention", "clarification", "meta", "answer_choice"}


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


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", clean_text(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: Any) -> str:
    text = strip_accents(clean_text(text).lower())
    text = text.replace("’", "'")
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def safe_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, indent=indent)


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
# Evidencia y claims
# ---------------------------------------------------------------------------

def normalize_claim(claim: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(claim, dict):
        claim_text = clean_text(claim.get("claim_text", claim.get("text", "")))
        claim_id = clean_text(claim.get("claim_id", "c1")) or "c1"
        claim_type = clean_text(claim.get("claim_type", "factual")).lower().strip()
        requires_evidence = coerce_bool(
            claim.get("requires_evidence", claim_type not in NON_FACTUAL_CLAIM_TYPES),
            default=claim_type not in NON_FACTUAL_CLAIM_TYPES,
        )
        importance = clean_text(claim.get("importance", "core")).lower().strip() or "core"
    else:
        claim_text = clean_text(claim)
        claim_id = "c1"
        claim_type = "factual"
        requires_evidence = True
        importance = "core"

    if claim_type not in {"factual", "answer_choice", "abstention", "clarification", "meta"}:
        claim_type = "factual"

    if importance not in {"core", "supporting", "low"}:
        importance = "core"

    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_type": claim_type,
        "requires_evidence": bool(requires_evidence),
        "importance": importance,
    }


def sanitize_chunk(item: dict[str, Any], *, fallback_rank: int = 1) -> dict[str, Any]:
    return {
        "rank": item.get("rank", fallback_rank),
        "chunk_id": clean_text(item.get("chunk_id", f"chunk_inline_{fallback_rank}")),
        "doc_id": clean_text(item.get("doc_id", "")),
        "title": clean_text(item.get("title", "")),
        "source": clean_text(item.get("source", "")),
        "topic": clean_text(item.get("topic", "")),
        "score": item.get("score", ""),
        "text": clean_text(item.get("text", "")),
    }


def sanitize_chunks(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for i, item in enumerate(chunks or [], start=1):
        if isinstance(item, dict):
            safe = sanitize_chunk(item, fallback_rank=i)
            if safe["text"]:
                output.append(safe)
    return output


def chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in chunks:
        cid = clean_text(item.get("chunk_id", ""))
        if cid and cid not in ids:
            ids.append(cid)
    return ids


def tokenize_content(text: Any) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def token_f1(a: Any, b: Any) -> tuple[float, float, float]:
    a_tokens = tokenize_content(a)
    b_tokens = tokenize_content(b)

    if not a_tokens or not b_tokens:
        return 0.0, 0.0, 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    overlap = sum((a_counts & b_counts).values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def evidence_text(chunks: list[dict[str, Any]]) -> str:
    return "\n".join(clean_text(chunk.get("text", "")) for chunk in chunks if clean_text(chunk.get("text", "")))


def has_negation(text: Any) -> bool:
    normalized = normalize_text(text)
    padded = f" {normalized} "
    return any(f" {normalize_text(marker)} " in padded for marker in NEGATION_MARKERS)


def best_evidence_match(
    claim_text: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    best = {
        "chunk_id": "",
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "chunk": None,
    }

    for chunk in chunks:
        text = clean_text(chunk.get("text", ""))
        if not text:
            continue
        precision, recall, f1 = token_f1(claim_text, text)

        claim_norm = normalize_text(claim_text)
        text_norm = normalize_text(text)

        # Bonus si el claim normalizado aparece casi textual en la evidencia.
        if claim_norm and claim_norm in text_norm:
            f1 = max(f1, 0.98)
            precision = max(precision, 0.98)
            recall = max(recall, 0.98)

        if f1 > best["f1"]:
            best = {
                "chunk_id": clean_text(chunk.get("chunk_id", "")),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "chunk": chunk,
            }

    return best


def build_search_query_rules(question: str, claim_text: str, *, max_terms: int = 12) -> str:
    """
    Query simple y reproducible para pedir más evidencia.
    Prioriza tokens del claim y conserva algo de la pregunta.
    """
    claim_tokens = tokenize_content(claim_text)
    question_tokens = tokenize_content(question)

    ordered: list[str] = []
    for token in claim_tokens + question_tokens:
        if token not in ordered:
            ordered.append(token)

    if not ordered:
        return normalize_space(f"{question} {claim_text}")

    return " ".join(ordered[:max_terms])


# ---------------------------------------------------------------------------
# Validación de payload de verificación
# ---------------------------------------------------------------------------

def normalize_verdict(value: Any) -> str:
    verdict = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    if verdict in VALID_VERDICTS:
        return verdict
    aliases = {
        "support": "supported",
        "supports": "supported",
        "entails": "supported",
        "entailed": "supported",
        "contradicted": "refuted",
        "contradiction": "refuted",
        "unsupported": "not_enough_info",
        "nei": "not_enough_info",
        "not_enough": "not_enough_info",
        "unknown": "not_enough_info",
        "insufficient": "not_enough_info",
    }
    return aliases.get(verdict, "not_enough_info")


def validate_verification_payload(
    obj: dict[str, Any] | None,
    *,
    claim: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    default_needs_more_evidence: bool = False,
) -> dict[str, Any]:
    claim = normalize_claim(claim)
    obj = obj or {}

    verdict = normalize_verdict(obj.get("verdict", "not_enough_info"))
    confidence = safe_float(obj.get("confidence", 0.0), default=0.0)
    rationale = clean_text(obj.get("rationale", ""))

    valid_chunk_ids = set(chunk_ids(retrieved_chunks))

    def filter_ids(value: Any) -> list[str]:
        ids: list[str] = []
        if isinstance(value, list):
            raw_values = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                raw_values = parsed if isinstance(parsed, list) else [value]
            except json.JSONDecodeError:
                raw_values = value.split("|") if "|" in value else [value]
        else:
            raw_values = []

        for raw in raw_values:
            cid = clean_text(raw)
            if cid and cid in valid_chunk_ids and cid not in ids:
                ids.append(cid)

        return ids

    supporting_chunk_ids = filter_ids(obj.get("supporting_chunk_ids", []))
    refuting_chunk_ids = filter_ids(obj.get("refuting_chunk_ids", []))

    needs_more_evidence = coerce_bool(
        obj.get("needs_more_evidence"),
        default=default_needs_more_evidence,
    )

    next_search_query = obj.get("next_search_query", None)
    if next_search_query is not None:
        next_search_query = clean_text(next_search_query) or None

    if verdict in {"supported", "refuted"}:
        needs_more_evidence = False
        next_search_query = None

    if verdict == "supported" and not supporting_chunk_ids:
        # Si el LLM dijo supported pero no indicó chunk válido, usar todos los chunks
        # disponibles como fallback conservador solo si hay evidencia.
        if retrieved_chunks:
            supporting_chunk_ids = [chunk_ids(retrieved_chunks)[0]]

    if verdict == "refuted" and not refuting_chunk_ids:
        if retrieved_chunks:
            refuting_chunk_ids = [chunk_ids(retrieved_chunks)[0]]

    if not rationale:
        rationale = (
            "Claim supported by retrieved evidence."
            if verdict == "supported"
            else "Claim refuted by retrieved evidence."
            if verdict == "refuted"
            else "Retrieved evidence is not sufficient to verify the claim."
        )

    return {
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "supporting_chunk_ids": supporting_chunk_ids,
        "refuting_chunk_ids": refuting_chunk_ids,
        "needs_more_evidence": bool(needs_more_evidence),
        "next_search_query": next_search_query,
    }


# ---------------------------------------------------------------------------
# Verificador por reglas
# ---------------------------------------------------------------------------

def verify_claim_rules(
    *,
    question: str,
    claim: dict[str, Any] | str,
    retrieved_chunks: list[dict[str, Any]] | None,
    round_id: int = 1,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict[str, Any]:
    claim_obj = normalize_claim(claim)
    chunks = sanitize_chunks(retrieved_chunks)
    claim_text = claim_obj["claim_text"]

    if not claim_text:
        return validate_verification_payload(
            {
                "verdict": "not_enough_info",
                "confidence": 1.0,
                "rationale": "El claim está vacío.",
                "needs_more_evidence": False,
                "next_search_query": None,
            },
            claim=claim_obj,
            retrieved_chunks=chunks,
        )

    if claim_obj["claim_type"] in NON_FACTUAL_CLAIM_TYPES or not claim_obj["requires_evidence"]:
        return validate_verification_payload(
            {
                "verdict": "supported",
                "confidence": 1.0,
                "rationale": (
                    "El claim no requiere verificación documental "
                    f"(claim_type={claim_obj['claim_type']})."
                ),
                "supporting_chunk_ids": [],
                "refuting_chunk_ids": [],
                "needs_more_evidence": False,
                "next_search_query": None,
            },
            claim=claim_obj,
            retrieved_chunks=chunks,
        )

    if not chunks:
        can_continue = round_id < max_rounds
        return validate_verification_payload(
            {
                "verdict": "not_enough_info",
                "confidence": 0.90,
                "rationale": "No hay evidencia recuperada para verificar el claim.",
                "supporting_chunk_ids": [],
                "refuting_chunk_ids": [],
                "needs_more_evidence": can_continue,
                "next_search_query": build_search_query_rules(question, claim_text) if can_continue else None,
            },
            claim=claim_obj,
            retrieved_chunks=chunks,
            default_needs_more_evidence=can_continue,
        )

    match = best_evidence_match(claim_text, chunks)
    best_chunk = match.get("chunk") or {}
    best_chunk_id = clean_text(match.get("chunk_id", ""))
    f1 = float(match.get("f1", 0.0))
    precision = float(match.get("precision", 0.0))
    recall = float(match.get("recall", 0.0))

    all_evidence = evidence_text(chunks)
    claim_negated = has_negation(claim_text)
    evidence_negated = has_negation(all_evidence)

    # Regla conservadora de contradicción:
    # solo marca refuted si hay alto overlap y difiere la negación.
    if f1 >= 0.60 and claim_negated != evidence_negated:
        return validate_verification_payload(
            {
                "verdict": "refuted",
                "confidence": min(0.85, max(0.55, f1)),
                "rationale": (
                    "La evidencia recuperada comparte términos centrales con el claim, "
                    "pero presenta una polaridad/negación incompatible."
                ),
                "supporting_chunk_ids": [],
                "refuting_chunk_ids": [best_chunk_id] if best_chunk_id else [],
                "needs_more_evidence": False,
                "next_search_query": None,
            },
            claim=claim_obj,
            retrieved_chunks=chunks,
        )

    # Regla de soporte:
    # funciona para smoke tests y claims con buena coincidencia léxica.
    if f1 >= 0.45 or (precision >= 0.55 and recall >= 0.25):
        return validate_verification_payload(
            {
                "verdict": "supported",
                "confidence": min(0.92, max(0.55, f1)),
                "rationale": (
                    f"La evidencia recuperada tiene suficiente solapamiento léxico "
                    f"con el claim (precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f})."
                ),
                "supporting_chunk_ids": [best_chunk_id] if best_chunk_id else [],
                "refuting_chunk_ids": [],
                "needs_more_evidence": False,
                "next_search_query": None,
            },
            claim=claim_obj,
            retrieved_chunks=chunks,
        )

    can_continue = round_id < max_rounds
    return validate_verification_payload(
        {
            "verdict": "not_enough_info",
            "confidence": 0.72,
            "rationale": (
                f"La evidencia recuperada no alcanza para confirmar ni refutar el claim "
                f"(mejor f1={f1:.3f})."
            ),
            "supporting_chunk_ids": [],
            "refuting_chunk_ids": [],
            "needs_more_evidence": can_continue,
            "next_search_query": build_search_query_rules(question, claim_text) if can_continue else None,
        },
        claim=claim_obj,
        retrieved_chunks=chunks,
        default_needs_more_evidence=can_continue,
    )


# ---------------------------------------------------------------------------
# Verificador por LLM
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


def verify_claim_llm(
    *,
    question: str,
    claim: dict[str, Any] | str,
    retrieved_chunks: list[dict[str, Any]] | None,
    round_id: int = 1,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    previous_verifications: list[dict[str, Any]] | None = None,
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> dict[str, Any]:
    claim_obj = normalize_claim(claim)
    chunks = sanitize_chunks(retrieved_chunks)

    prompt = build_claim_verification_prompt(
        question=question,
        claim=claim_obj,
        retrieved_chunks=chunks,
        round_id=round_id,
        max_rounds=max_rounds,
        previous_verifications=previous_verifications or [],
        max_chars_per_chunk=max_chars_per_chunk,
    )

    dry_payload = verify_claim_rules(
        question=question,
        claim=claim_obj,
        retrieved_chunks=chunks,
        round_id=round_id,
        max_rounds=max_rounds,
    )

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S4_CLAIM_VERIFICATION_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    validated = validate_verification_payload(
        call["parsed"],
        claim=claim_obj,
        retrieved_chunks=chunks,
        default_needs_more_evidence=round_id < max_rounds,
    )

    return {
        **validated,
        "verifier_strategy": "llm",
        "verifier_model": call["model"],
        "verifier_raw_output": call["raw_output"],
        "verifier_parse_error": call["parse_error"],
        "verifier_usage_json": call["usage_json"],
        "verifier_input_tokens": call["input_tokens"],
        "verifier_output_tokens": call["output_tokens"],
        "verifier_total_tokens": call["total_tokens"],
        "verifier_latency_seconds": call["latency_seconds"],
        "verifier_dry_run": call["dry_run"],
    }


def generate_search_query_llm(
    *,
    question: str,
    claim: dict[str, Any] | str,
    previous_evidence: list[dict[str, Any]] | None = None,
    previous_verdict: str = "",
    previous_rationale: str = "",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Utilidad opcional para el próximo controlador FIRE.
    No es obligatoria para verify_claim_once, pero queda lista.
    """
    claim_obj = normalize_claim(claim)
    chunks = sanitize_chunks(previous_evidence)

    prompt = build_search_query_prompt(
        question=question,
        claim=claim_obj,
        previous_evidence=chunks,
        previous_verdict=previous_verdict,
        previous_rationale=previous_rationale,
    )

    dry_payload = {
        "search_query": build_search_query_rules(question, claim_obj["claim_text"]),
        "reason": "Dry-run/rules query generated from question and claim keywords.",
    }

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S4_SEARCH_QUERY_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    parsed = call["parsed"] or {}
    search_query = clean_text(parsed.get("search_query", ""))
    reason = clean_text(parsed.get("reason", ""))

    if not search_query:
        search_query = build_search_query_rules(question, claim_obj["claim_text"])
    if not reason:
        reason = "Fallback query generated from question and claim."

    return {
        "search_query": search_query,
        "reason": reason,
        "query_model": call["model"],
        "query_raw_output": call["raw_output"],
        "query_parse_error": call["parse_error"],
        "query_usage_json": call["usage_json"],
        "query_input_tokens": call["input_tokens"],
        "query_output_tokens": call["output_tokens"],
        "query_total_tokens": call["total_tokens"],
        "query_latency_seconds": call["latency_seconds"],
        "query_dry_run": call["dry_run"],
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def verify_claim_once(
    *,
    question: str,
    claim: dict[str, Any] | str,
    retrieved_chunks: list[dict[str, Any]] | None,
    round_id: int = 1,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    previous_verifications: list[dict[str, Any]] | None = None,
    strategy: VerificationStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> dict[str, Any]:
    strategy = clean_text(strategy).lower()
    if strategy not in VALID_VERIFICATION_STRATEGIES:
        raise ValueError(
            f"strategy inválida: {strategy!r}. "
            f"Opciones válidas: {sorted(VALID_VERIFICATION_STRATEGIES)}"
        )

    start = time.time()
    claim_obj = normalize_claim(claim)
    chunks = sanitize_chunks(retrieved_chunks)

    if strategy == "rules":
        result = verify_claim_rules(
            question=question,
            claim=claim_obj,
            retrieved_chunks=chunks,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        result.update(
            {
                "verifier_strategy": "rules",
                "verifier_model": "",
                "verifier_raw_output": "",
                "verifier_parse_error": "",
                "verifier_usage_json": "{}",
                "verifier_input_tokens": 0,
                "verifier_output_tokens": 0,
                "verifier_total_tokens": 0,
                "verifier_latency_seconds": round(time.time() - start, 3),
                "verifier_dry_run": False,
            }
        )
        return result

    if strategy == "llm":
        return verify_claim_llm(
            question=question,
            claim=claim_obj,
            retrieved_chunks=chunks,
            round_id=round_id,
            max_rounds=max_rounds,
            previous_verifications=previous_verifications or [],
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
            max_chars_per_chunk=max_chars_per_chunk,
        )

    # hybrid:
    # - non factual / no evidence required: rules
    # - no evidence: rules genera next_search_query
    # - factual con evidencia: LLM para juicio más fino
    if claim_obj["claim_type"] in NON_FACTUAL_CLAIM_TYPES or not claim_obj["requires_evidence"] or not chunks:
        result = verify_claim_rules(
            question=question,
            claim=claim_obj,
            retrieved_chunks=chunks,
            round_id=round_id,
            max_rounds=max_rounds,
        )
        result.update(
            {
                "verifier_strategy": "hybrid_rules",
                "verifier_model": "",
                "verifier_raw_output": "",
                "verifier_parse_error": "",
                "verifier_usage_json": "{}",
                "verifier_input_tokens": 0,
                "verifier_output_tokens": 0,
                "verifier_total_tokens": 0,
                "verifier_latency_seconds": round(time.time() - start, 3),
                "verifier_dry_run": False,
            }
        )
        return result

    result = verify_claim_llm(
        question=question,
        claim=claim_obj,
        retrieved_chunks=chunks,
        round_id=round_id,
        max_rounds=max_rounds,
        previous_verifications=previous_verifications or [],
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    result["verifier_strategy"] = "hybrid_llm"
    return result


# ---------------------------------------------------------------------------
# CLI helpers / self-test
# ---------------------------------------------------------------------------

def parse_evidence_json(value: str) -> list[dict[str, Any]]:
    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--evidence-json inválido: {exc}") from exc

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError("--evidence-json debe ser un objeto o lista de objetos.")

    chunks: list[dict[str, Any]] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError("Cada item de --evidence-json debe ser objeto/dict.")
        chunks.append(sanitize_chunk(item, fallback_rank=i))
    return chunks


def chunks_from_cli_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.evidence_json:
        return parse_evidence_json(args.evidence_json)

    if args.evidence_path:
        path = Path(args.evidence_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe --evidence-path: {path}")
        return parse_evidence_json(path.read_text(encoding="utf-8"))

    if args.evidence_text:
        return [
            {
                "rank": 1,
                "chunk_id": args.evidence_chunk_id or "chunk_cli_001",
                "doc_id": "doc_cli",
                "title": args.evidence_title or "CLI evidence",
                "source": "cli",
                "topic": "",
                "score": 1.0,
                "text": args.evidence_text,
            }
        ]

    return []


def run_self_test() -> None:
    examples = [
        {
            "name": "supported",
            "question": "According to the available corpus, who composed La traviata?",
            "claim": {
                "claim_id": "c1",
                "claim_text": "La traviata was composed by Giuseppe Verdi.",
                "claim_type": "factual",
                "requires_evidence": True,
                "importance": "core",
            },
            "chunks": [
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
            ],
        },
        {
            "name": "not_enough_info_no_chunks",
            "question": "According to the available corpus, who composed La traviata?",
            "claim": {
                "claim_id": "c1",
                "claim_text": "La traviata was composed by Giuseppe Verdi.",
                "claim_type": "factual",
                "requires_evidence": True,
                "importance": "core",
            },
            "chunks": [],
        },
        {
            "name": "non_factual_abstention",
            "question": "According to the available corpus, what is the unpublished access code?",
            "claim": {
                "claim_id": "c1",
                "claim_text": "No hay información suficiente para responder con la evidencia disponible.",
                "claim_type": "abstention",
                "requires_evidence": False,
                "importance": "core",
            },
            "chunks": [],
        },
    ]

    print("S4 FIRE verifier self-test")
    print("=" * 72)

    for ex in examples:
        result = verify_claim_once(
            question=ex["question"],
            claim=ex["claim"],
            retrieved_chunks=ex["chunks"],
            strategy="rules",
            round_id=1,
            max_rounds=2,
        )

        print(f"Example: {ex['name']}")
        print(safe_json_dumps(result, indent=2))
        print("-" * 72)

    dry_result = verify_claim_once(
        question=examples[0]["question"],
        claim=examples[0]["claim"],
        retrieved_chunks=examples[0]["chunks"],
        strategy="llm",
        dry_run=True,
    )
    print("Example: llm_dry_run")
    print(safe_json_dumps(dry_result, indent=2))
    print("-" * 72)

    print("S4 FIRE verifier self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verificador de claims para S4 FIRE-like."
    )
    parser.add_argument("--question", type=str, default="", help="Pregunta original.")
    parser.add_argument("--claim", type=str, default="", help="Texto del claim.")
    parser.add_argument("--claim-id", type=str, default="c1", help="ID del claim.")
    parser.add_argument(
        "--claim-type",
        type=str,
        default="factual",
        choices=["factual", "answer_choice", "abstention", "clarification", "meta"],
        help="Tipo del claim.",
    )
    parser.add_argument(
        "--requires-evidence",
        action="store_true",
        help="Marca explícitamente que el claim requiere evidencia.",
    )
    parser.add_argument(
        "--no-requires-evidence",
        action="store_true",
        help="Marca explícitamente que el claim no requiere evidencia.",
    )
    parser.add_argument("--importance", type=str, default="core", help="core/supporting/low.")
    parser.add_argument("--evidence-text", type=str, default="", help="Evidencia inline.")
    parser.add_argument("--evidence-title", type=str, default="", help="Título de la evidencia inline.")
    parser.add_argument("--evidence-chunk-id", type=str, default="", help="chunk_id de evidencia inline.")
    parser.add_argument("--evidence-json", type=str, default="", help="JSON de chunk o lista de chunks.")
    parser.add_argument("--evidence-path", type=str, default="", help="Archivo JSON con chunk o lista de chunks.")
    parser.add_argument(
        "--strategy",
        choices=sorted(VALID_VERIFICATION_STRATEGIES),
        default="rules",
        help="Estrategia de verificación.",
    )
    parser.add_argument("--round-id", type=int, default=1, help="Ronda actual de verificación.")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="Máximo de rondas.")
    parser.add_argument("--model", type=str, default=None, help="Modelo LLM para strategy=llm/hybrid.")
    parser.add_argument("--max-retries", type=int, default=2, help="Reintentos para llamada LLM.")
    parser.add_argument("--dry-run", action="store_true", help="No llama API; simula salida LLM con reglas.")
    parser.add_argument("--self-test", action="store_true", help="Ejecuta ejemplos locales.")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not clean_text(args.question) or not clean_text(args.claim):
        raise ValueError(
            "Debés pasar --question y --claim, o usar --self-test."
        )

    if args.requires_evidence and args.no_requires_evidence:
        raise ValueError("No podés usar --requires-evidence y --no-requires-evidence a la vez.")

    if args.requires_evidence:
        requires_evidence = True
    elif args.no_requires_evidence:
        requires_evidence = False
    else:
        requires_evidence = args.claim_type not in NON_FACTUAL_CLAIM_TYPES

    claim = {
        "claim_id": args.claim_id,
        "claim_text": args.claim,
        "claim_type": args.claim_type,
        "requires_evidence": requires_evidence,
        "importance": args.importance,
    }

    chunks = chunks_from_cli_args(args)

    result = verify_claim_once(
        question=args.question,
        claim=claim,
        retrieved_chunks=chunks,
        round_id=args.round_id,
        max_rounds=args.max_rounds,
        strategy=args.strategy,
        model=args.model,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    print(safe_json_dumps(result, indent=2))


if __name__ == "__main__":
    main()

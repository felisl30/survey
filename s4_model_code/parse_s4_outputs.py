#!/usr/bin/env python3
"""
parse_s4_outputs.py

Parser/normalizador para salidas crudas de S4 FIRE-like.

Entrada típica:
    outputs/s4/generation/fire_like_s4_raw.csv

Salida típica:
    outputs/s4/generation/fire_like_s4_parsed.csv

Motivación
----------
run_s4_fire_like.py ya guarda muchas columnas "parseadas" porque S4 devuelve
un JSON estructurado. Aun así, este archivo existe para mantener el pipeline
ordenado y comparable con S0/S1/S2/S3:

    run_s4_fire_like.py
        -> parse_s4_outputs.py
            -> evaluate_s4_claims.py
            -> evaluate_s4_answers.py
            -> verify_s4_results.py

Este parser:
- No llama al LLM.
- Conserva todas las columnas originales útiles.
- Lee raw_output cuando existe y extrae el JSON S4.
- Normaliza:
    parsed_answer
    parsed_confidence
    parsed_abstained
    valid_answer_format
    valid_format
    s4_final_decision
    s4_correction_applied
    s4_num_claims
    s4_num_supported_claims
    s4_num_refuted_claims
    s4_num_nei_claims
    s4_num_verification_rounds
    s4_num_retrieval_rounds
    s4_num_chunks_retrieved_total
    s4_claims_json
    s4_claim_results_json
    s4_claim_trace_json
    s4_evidence_ids
    s4_retrieved_chunk_ids
    run_error_present

Uso rápido
----------

python s4_model_code/parse_s4_outputs.py \
  --input-path outputs/s4/generation/fire_like_s4_raw_retrieve_test_5_hybrid_claims_llm_verify.csv \
  --output-path outputs/s4/generation/fire_like_s4_parsed_retrieve_test_5_hybrid_claims_llm_verify.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s4/generation/fire_like_s4_raw.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s4/generation/fire_like_s4_parsed.csv")

VALID_FINAL_DECISIONS = {
    "unchanged",
    "corrected",
    "abstained",
    "clarification_kept",
    "no_claims",
    "error",
}

VALID_VERDICTS = {
    "supported",
    "refuted",
    "not_enough_info",
}

ADDITIONAL_COLUMNS = [
    "run_error_present",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "valid_answer_format",
    "valid_format",
    "parse_method",
    "parse_error",
    "parsed_json",

    "s4_answer",
    "s4_confidence",
    "s4_abstained",
    "s4_final_decision",
    "s4_correction_applied",
    "s4_unsupported_claims_removed",
    "s4_corrected_claims",

    "s4_num_claims",
    "s4_num_supported_claims",
    "s4_num_refuted_claims",
    "s4_num_nei_claims",
    "s4_num_claims_requiring_evidence",
    "s4_num_verification_rounds",
    "s4_num_retrieval_rounds",
    "s4_num_chunks_retrieved_total",

    "s4_claim_ids",
    "s4_claim_texts",
    "s4_claim_verdicts",
    "s4_supported_claim_ids",
    "s4_refuted_claim_ids",
    "s4_nei_claim_ids",
    "s4_supporting_chunk_ids",
    "s4_refuting_chunk_ids",
    "s4_evidence_ids",
    "s4_retrieved_chunk_ids",

    "s4_claims_json",
    "s4_claim_results_json",
    "s4_claim_trace_json",
    "s4_extraction_json",
    "s4_repair_json",

    "s4_claim_strategy",
    "s4_verification_strategy",
    "s4_query_strategy",
    "s4_repair_strategy",

    "s4_input_tokens",
    "s4_output_tokens",
    "s4_total_tokens",
    "s4_latency_seconds",
    "s4_retrieval_latency_seconds",
]


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


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    if is_missing(value):
        return default
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default

    if 1.0 < number <= 100.0:
        number = number / 100.0

    return min(max(number, 0.0), 1.0)


def coerce_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return bool(value)
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def safe_json_loads(value: Any) -> Any:
    text = clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


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
        return None, "No se encontró un objeto JSON."

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON inválido: {exc}"

    if not isinstance(parsed, dict):
        return None, "El JSON parseado no es un objeto/dict."

    return parsed, ""


def normalize_final_decision(value: Any) -> str:
    text = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    if text in VALID_FINAL_DECISIONS:
        return text
    return "error" if text == "run_error" else text


def normalize_verdict(value: Any) -> str:
    text = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    aliases = {
        "support": "supported",
        "supports": "supported",
        "entailed": "supported",
        "entails": "supported",
        "contradicted": "refuted",
        "contradiction": "refuted",
        "nei": "not_enough_info",
        "unknown": "not_enough_info",
        "unsupported": "not_enough_info",
    }
    text = aliases.get(text, text)
    return text if text in VALID_VERDICTS else text


def unique_preserve_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def join_pipe(values: list[Any]) -> str:
    return "|".join(unique_preserve_order(values))


# ---------------------------------------------------------------------------
# Extracción desde raw_output / columnas existentes
# ---------------------------------------------------------------------------

def get_raw_s4_object(row: pd.Series) -> tuple[dict[str, Any], str, str]:
    """
    Devuelve:
    - objeto S4 parseado
    - parse_method
    - parse_error

    Si raw_output no existe o no es parseable, arma un objeto desde columnas.
    """
    raw_output = clean_text(row.get("raw_output", ""))

    if raw_output:
        obj, error = load_json_object(raw_output)
        if obj is not None:
            return obj, "raw_output_json", ""

        # Si raw_output falló, seguimos con columnas pero registramos error.
        fallback = build_object_from_columns(row)
        return fallback, "columns_fallback_after_raw_output_parse_error", error

    return build_object_from_columns(row), "columns", ""


def build_object_from_columns(row: pd.Series) -> dict[str, Any]:
    claims = safe_json_loads(row.get("s4_claims_json", "")) or []
    claim_results = safe_json_loads(row.get("s4_claim_results_json", "")) or []
    claim_trace = safe_json_loads(row.get("s4_claim_trace_json", "")) or claim_results

    extraction = safe_json_loads(row.get("s4_extraction_json", "")) or {}
    repair = safe_json_loads(row.get("s4_repair_json", "")) or {}

    return {
        "initial_answer": clean_text(row.get("s4_initial_answer", row.get("source_initial_answer", ""))),
        "answer": clean_text(row.get("s4_answer", row.get("parsed_answer", ""))),
        "confidence": safe_float(row.get("s4_confidence", row.get("parsed_confidence", None)), default=None),
        "abstained": coerce_bool(row.get("s4_abstained", row.get("parsed_abstained", False)), default=False),
        "final_decision": clean_text(row.get("s4_final_decision", "")),
        "correction_applied": coerce_bool(row.get("s4_correction_applied", False), default=False),
        "unsupported_claims_removed": parse_pipe_or_json_list(row.get("s4_unsupported_claims_removed", "")),
        "corrected_claims": safe_json_loads(row.get("s4_corrected_claims", "")) or [],
        "num_claims": safe_int(row.get("s4_num_claims", 0), default=0),
        "num_supported_claims": safe_int(row.get("s4_num_supported_claims", 0), default=0),
        "num_refuted_claims": safe_int(row.get("s4_num_refuted_claims", 0), default=0),
        "num_nei_claims": safe_int(row.get("s4_num_nei_claims", 0), default=0),
        "num_claims_requiring_evidence": safe_int(row.get("s4_num_claims_requiring_evidence", 0), default=0),
        "num_verification_rounds": safe_int(row.get("s4_num_verification_rounds", 0), default=0),
        "num_retrieval_rounds": safe_int(row.get("s4_num_retrieval_rounds", 0), default=0),
        "num_chunks_retrieved_total": safe_int(row.get("s4_num_chunks_retrieved_total", 0), default=0),
        "evidence_ids": parse_pipe_or_json_list(row.get("s4_evidence_ids", "")),
        "retrieved_chunk_ids": parse_pipe_or_json_list(row.get("s4_retrieved_chunk_ids", "")),
        "claims": claims,
        "claim_results": claim_results,
        "claim_trace": claim_trace,
        "extraction": extraction,
        "repair": repair,
        "claim_strategy": clean_text(row.get("s4_claim_strategy", "")),
        "verification_strategy": clean_text(row.get("s4_verification_strategy", "")),
        "query_strategy": clean_text(row.get("s4_query_strategy", "")),
        "repair_strategy": clean_text(row.get("s4_repair_strategy", "")),
        "input_tokens": safe_int(row.get("s4_input_tokens", row.get("input_tokens", 0)), default=0),
        "output_tokens": safe_int(row.get("s4_output_tokens", row.get("output_tokens", 0)), default=0),
        "total_tokens": safe_int(row.get("s4_total_tokens", row.get("total_tokens", 0)), default=0),
        "latency_seconds": row.get("s4_latency_seconds", row.get("latency_seconds", None)),
        "retrieval_latency_seconds": row.get("s4_retrieval_latency_seconds", row.get("retrieval_latency_seconds", None)),
        "error": clean_text(row.get("error", "")),
    }


def parse_pipe_or_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]

    text = clean_text(value)
    if not text:
        return []

    parsed = safe_json_loads(text)
    if isinstance(parsed, list):
        return [clean_text(x) for x in parsed if clean_text(x)]

    if "|" in text:
        return [clean_text(x) for x in text.split("|") if clean_text(x)]

    return [text]


def flatten_claim_results(claim_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(claim_results, list):
        claim_results = []

    claim_ids: list[str] = []
    claim_texts: list[str] = []
    verdicts: list[str] = []

    supported_claim_ids: list[str] = []
    refuted_claim_ids: list[str] = []
    nei_claim_ids: list[str] = []

    supporting_chunk_ids: list[str] = []
    refuting_chunk_ids: list[str] = []
    evidence_ids: list[str] = []
    retrieved_chunk_ids: list[str] = []

    for item in claim_results:
        if not isinstance(item, dict):
            continue

        cid = clean_text(item.get("claim_id", ""))
        ctext = clean_text(item.get("claim_text", ""))
        verdict = normalize_verdict(item.get("verdict", ""))

        claim_ids.append(cid)
        claim_texts.append(ctext)
        verdicts.append(verdict)

        if verdict == "supported":
            supported_claim_ids.append(cid)
        elif verdict == "refuted":
            refuted_claim_ids.append(cid)
        elif verdict == "not_enough_info":
            nei_claim_ids.append(cid)

        supporting_chunk_ids.extend(item.get("supporting_chunk_ids", []) or [])
        refuting_chunk_ids.extend(item.get("refuting_chunk_ids", []) or [])
        evidence_ids.extend(item.get("evidence_ids", []) or [])
        retrieved_chunk_ids.extend(item.get("retrieved_chunk_ids", []) or [])

        for round_item in item.get("rounds_trace", []) or []:
            if isinstance(round_item, dict):
                retrieved_chunk_ids.extend(round_item.get("retrieved_chunk_ids", []) or [])

    return {
        "s4_claim_ids": join_pipe(claim_ids),
        "s4_claim_texts": safe_json_dumps([x for x in claim_texts if x]),
        "s4_claim_verdicts": join_pipe(verdicts),
        "s4_supported_claim_ids": join_pipe(supported_claim_ids),
        "s4_refuted_claim_ids": join_pipe(refuted_claim_ids),
        "s4_nei_claim_ids": join_pipe(nei_claim_ids),
        "s4_supporting_chunk_ids": join_pipe(supporting_chunk_ids),
        "s4_refuting_chunk_ids": join_pipe(refuting_chunk_ids),
        "s4_evidence_ids": join_pipe(evidence_ids),
        "s4_retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),
    }


def infer_counts_from_claim_results(claim_results: list[dict[str, Any]]) -> dict[str, int]:
    if not isinstance(claim_results, list):
        claim_results = []

    n_claims = 0
    n_supported = 0
    n_refuted = 0
    n_nei = 0
    n_requires_evidence = 0
    n_verification_rounds = 0

    for item in claim_results:
        if not isinstance(item, dict):
            continue

        n_claims += 1
        verdict = normalize_verdict(item.get("verdict", ""))
        if verdict == "supported":
            n_supported += 1
        elif verdict == "refuted":
            n_refuted += 1
        elif verdict == "not_enough_info":
            n_nei += 1

        if coerce_bool(item.get("requires_evidence"), default=False):
            n_requires_evidence += 1

        n_verification_rounds += safe_int(item.get("rounds", 0), default=0)

    return {
        "num_claims": n_claims,
        "num_supported_claims": n_supported,
        "num_refuted_claims": n_refuted,
        "num_nei_claims": n_nei,
        "num_claims_requiring_evidence": n_requires_evidence,
        "num_verification_rounds": n_verification_rounds,
    }


# ---------------------------------------------------------------------------
# Parser por fila
# ---------------------------------------------------------------------------

def parse_row(row: pd.Series) -> dict[str, Any]:
    obj, parse_method, parse_error = get_raw_s4_object(row)

    answer = clean_text(obj.get("answer", row.get("s4_answer", row.get("parsed_answer", ""))))
    confidence = safe_float(obj.get("confidence", row.get("s4_confidence", row.get("parsed_confidence", None))), default=None)
    abstained = coerce_bool(obj.get("abstained", row.get("s4_abstained", row.get("parsed_abstained", False))), default=False)

    final_decision = normalize_final_decision(obj.get("final_decision", row.get("s4_final_decision", "")))
    correction_applied = coerce_bool(
        obj.get("correction_applied", row.get("s4_correction_applied", False)),
        default=False,
    )

    claim_results = obj.get("claim_results", [])
    if not isinstance(claim_results, list):
        claim_results = []

    claims = obj.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    claim_trace = obj.get("claim_trace", claim_results)
    if not isinstance(claim_trace, list):
        claim_trace = claim_results

    inferred_counts = infer_counts_from_claim_results(claim_results)

    num_claims = safe_int(obj.get("num_claims", row.get("s4_num_claims", inferred_counts["num_claims"])), default=inferred_counts["num_claims"])
    num_supported = safe_int(obj.get("num_supported_claims", row.get("s4_num_supported_claims", inferred_counts["num_supported_claims"])), default=inferred_counts["num_supported_claims"])
    num_refuted = safe_int(obj.get("num_refuted_claims", row.get("s4_num_refuted_claims", inferred_counts["num_refuted_claims"])), default=inferred_counts["num_refuted_claims"])
    num_nei = safe_int(obj.get("num_nei_claims", row.get("s4_num_nei_claims", inferred_counts["num_nei_claims"])), default=inferred_counts["num_nei_claims"])
    num_requires_evidence = safe_int(
        obj.get("num_claims_requiring_evidence", row.get("s4_num_claims_requiring_evidence", inferred_counts["num_claims_requiring_evidence"])),
        default=inferred_counts["num_claims_requiring_evidence"],
    )

    num_verification_rounds = safe_int(
        obj.get("num_verification_rounds", row.get("s4_num_verification_rounds", inferred_counts["num_verification_rounds"])),
        default=inferred_counts["num_verification_rounds"],
    )
    num_retrieval_rounds = safe_int(obj.get("num_retrieval_rounds", row.get("s4_num_retrieval_rounds", 0)), default=0)
    num_chunks_retrieved_total = safe_int(obj.get("num_chunks_retrieved_total", row.get("s4_num_chunks_retrieved_total", 0)), default=0)

    flat = flatten_claim_results(claim_results)

    evidence_ids = parse_pipe_or_json_list(obj.get("evidence_ids", []))
    retrieved_chunk_ids = parse_pipe_or_json_list(obj.get("retrieved_chunk_ids", []))

    if not evidence_ids:
        evidence_ids = parse_pipe_or_json_list(flat["s4_evidence_ids"])
    if not retrieved_chunk_ids:
        retrieved_chunk_ids = parse_pipe_or_json_list(flat["s4_retrieved_chunk_ids"])

    if num_chunks_retrieved_total <= 0:
        num_chunks_retrieved_total = len(unique_preserve_order(retrieved_chunk_ids))

    error_text = clean_text(obj.get("error", row.get("error", "")))
    existing_run_error = coerce_bool(row.get("run_error_present"), default=False)
    run_error_present = bool(existing_run_error or error_text)

    valid_answer_format = bool(answer and not run_error_present and final_decision != "error")

    parsed_json = {
        "answer": answer,
        "confidence": confidence,
        "abstained": abstained,
        "final_decision": final_decision,
        "correction_applied": correction_applied,
        "num_claims": num_claims,
        "num_supported_claims": num_supported,
        "num_refuted_claims": num_refuted,
        "num_nei_claims": num_nei,
    }

    final_parse_error = parse_error
    if not answer:
        final_parse_error = (final_parse_error + " " if final_parse_error else "") + "Respuesta S4 vacía."
    if final_decision and final_decision not in VALID_FINAL_DECISIONS:
        final_parse_error = (final_parse_error + " " if final_parse_error else "") + f"final_decision desconocida: {final_decision!r}."

    return {
        "run_error_present": run_error_present,
        "parsed_answer": answer,
        "parsed_confidence": confidence,
        "parsed_abstained": bool(abstained),
        "valid_answer_format": valid_answer_format,
        "valid_format": valid_answer_format,
        "parse_method": parse_method,
        "parse_error": final_parse_error,
        "parsed_json": safe_json_dumps(parsed_json),

        "s4_answer": answer,
        "s4_confidence": confidence,
        "s4_abstained": bool(abstained),
        "s4_final_decision": final_decision,
        "s4_correction_applied": bool(correction_applied),
        "s4_unsupported_claims_removed": join_pipe(obj.get("unsupported_claims_removed", []) or []),
        "s4_corrected_claims": safe_json_dumps(obj.get("corrected_claims", []) or []),

        "s4_num_claims": int(num_claims),
        "s4_num_supported_claims": int(num_supported),
        "s4_num_refuted_claims": int(num_refuted),
        "s4_num_nei_claims": int(num_nei),
        "s4_num_claims_requiring_evidence": int(num_requires_evidence),
        "s4_num_verification_rounds": int(num_verification_rounds),
        "s4_num_retrieval_rounds": int(num_retrieval_rounds),
        "s4_num_chunks_retrieved_total": int(num_chunks_retrieved_total),

        **flat,
        "s4_evidence_ids": join_pipe(evidence_ids),
        "s4_retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),

        "s4_claims_json": safe_json_dumps(claims),
        "s4_claim_results_json": safe_json_dumps(claim_results),
        "s4_claim_trace_json": safe_json_dumps(claim_trace),
        "s4_extraction_json": safe_json_dumps(obj.get("extraction", {}) or {}),
        "s4_repair_json": safe_json_dumps(obj.get("repair", {}) or {}),

        "s4_claim_strategy": clean_text(obj.get("claim_strategy", row.get("s4_claim_strategy", ""))),
        "s4_verification_strategy": clean_text(obj.get("verification_strategy", row.get("s4_verification_strategy", ""))),
        "s4_query_strategy": clean_text(obj.get("query_strategy", row.get("s4_query_strategy", ""))),
        "s4_repair_strategy": clean_text(obj.get("repair_strategy", row.get("s4_repair_strategy", ""))),

        "s4_input_tokens": safe_int(obj.get("input_tokens", row.get("s4_input_tokens", row.get("input_tokens", 0))), default=0),
        "s4_output_tokens": safe_int(obj.get("output_tokens", row.get("s4_output_tokens", row.get("output_tokens", 0))), default=0),
        "s4_total_tokens": safe_int(obj.get("total_tokens", row.get("s4_total_tokens", row.get("total_tokens", 0))), default=0),
        "s4_latency_seconds": obj.get("latency_seconds", row.get("s4_latency_seconds", row.get("latency_seconds", None))),
        "s4_retrieval_latency_seconds": obj.get("retrieval_latency_seconds", row.get("s4_retrieval_latency_seconds", row.get("retrieval_latency_seconds", None))),
    }


# ---------------------------------------------------------------------------
# Pipeline de archivo
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna 'id'.")

    if "raw_output" not in df.columns and "s4_answer" not in df.columns and "parsed_answer" not in df.columns:
        raise ValueError(
            "El CSV S4 debe tener raw_output, s4_answer o parsed_answer. "
            "Asegurate de haber corrido run_s4_fire_like.py."
        )


def parse_outputs(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    parsed_rows = [parse_row(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)

    # Evita duplicar columnas si se re-parsea un archivo ya parseado.
    base_cols = [col for col in df.columns if col not in ADDITIONAL_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), parsed_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


# ---------------------------------------------------------------------------
# Resumen de terminal
# ---------------------------------------------------------------------------

def mean_bool(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.astype(bool).mean())


def mean_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(df: pd.DataFrame) -> None:
    print("\nResumen de parseo S4")
    print("--------------------")
    print(f"Filas totales: {len(df)}")

    if "s4_final_decision" in df.columns:
        print("\nDecisiones finales S4:")
        print(df["s4_final_decision"].value_counts(dropna=False).to_string())

    if "parse_method" in df.columns:
        print("\nMétodos de parseo:")
        print(df["parse_method"].value_counts(dropna=False).to_string())

    if "valid_answer_format" in df.columns:
        print(f"\nValid answer format rate: {fmt(mean_bool(df['valid_answer_format']))}")

    if "run_error_present" in df.columns:
        print(f"Run error rate: {fmt(mean_bool(df['run_error_present']))}")

    if "s4_abstained" in df.columns:
        print(f"S4 abstention rate: {fmt(mean_bool(df['s4_abstained']))}")

    if "s4_correction_applied" in df.columns:
        print(f"S4 correction rate: {fmt(mean_bool(df['s4_correction_applied']))}")

    for col, label in [
        ("s4_num_claims", "Avg claims"),
        ("s4_num_supported_claims", "Avg supported claims"),
        ("s4_num_refuted_claims", "Avg refuted claims"),
        ("s4_num_nei_claims", "Avg NEI claims"),
        ("s4_num_verification_rounds", "Avg verification rounds"),
        ("s4_num_retrieval_rounds", "Avg retrieval rounds"),
        ("s4_num_chunks_retrieved_total", "Avg chunks retrieved"),
        ("s4_total_tokens", "Avg total tokens"),
        ("s4_latency_seconds", "Avg latency seconds"),
    ]:
        if col in df.columns:
            print(f"{label}: {fmt(mean_numeric(df[col]))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parsea/normaliza salidas crudas de S4 FIRE-like."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV crudo generado por run_s4_fire_like.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV parseado/normalizado de salida.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_df = parse_outputs(args.input_path, args.output_path)
    print(f"Resultados S4 parseados guardados en: {args.output_path}")
    print_summary(output_df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
parse_s3_outputs.py

Parser para S3: FLARE-like / active retrieval.

Entrada por defecto:
    outputs/s3/generation/flare_like_s3_raw.csv

Salida por defecto:
    outputs/s3/generation/flare_like_s3_parsed.csv

Diseño
------
- No llama al LLM.
- Conserva todas las columnas originales del CSV raw.
- Parsea raw_response_json / raw_output / columnas fallback.
- Normaliza respuesta final, abstención, confidence, trazas FLARE, evidencia,
  retrieval activo, costos y formato multiple-choice.
- Deja columnas listas para evaluación posterior.

Uso recomendado
---------------
python s3_model_code/parse_s3_outputs.py \
  --input-path outputs/s3/generation/flare_like_s3_raw_test_5_v4.csv \
  --output-path outputs/s3/generation/flare_like_s3_parsed_test_5_v4.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s3/generation/flare_like_s3_raw.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s3/generation/flare_like_s3_parsed.csv")

VALID_SUPPORT_STATUSES = {"supported", "corrected", "not_enough_info", "refuted", "not_checked"}
VALID_RETRIEVAL_MODES = {"active", "none", "single_step", "multi_step"}

ABSTENTION_MARKERS = [
    "not enough information",
    "insufficient information",
    "not enough evidence",
    "insufficient evidence",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "can't be determined",
    "i don't know",
    "i do not know",
    "does not provide enough information",
    "does not provide sufficient information",
    "the retrieved evidence does not provide",
    "the recovered evidence does not provide",
    "not supported by the context",
    "not supported by the available context",
    "not supported by the recovered evidence",
    "no hay información suficiente",
    "no hay informacion suficiente",
    "información insuficiente",
    "informacion insuficiente",
    "no tengo información suficiente",
    "no tengo informacion suficiente",
    "no se puede determinar",
    "no puedo determinar",
    "sin evidencia suficiente",
    "evidencia insuficiente",
]

CLARIFICATION_MARKERS = [
    "need clarification",
    "needs clarification",
    "please clarify",
    "could you clarify",
    "ambiguous",
    "unclear",
    "necesito una aclaración",
    "necesito una aclaracion",
    "podrías aclarar",
    "podrias aclarar",
    "pregunta ambigua",
]

ADDITIONAL_COLUMNS = [
    "run_error_present",
    "parsed_answer",
    "parsed_answer_normalized",
    "parsed_choice",
    "parsed_confidence",
    "parsed_abstained",
    "parsed_clarification",
    "parsed_retrieval_mode",
    "parsed_active_retrieval_triggered",
    "parsed_num_generation_steps",
    "parsed_num_retrieval_steps",
    "parsed_num_chunks_retrieved_total",
    "parsed_unique_chunks_retrieved",
    "parsed_evidence_ids",
    "parsed_evidence_ids_json",
    "parsed_retrieved_chunk_ids",
    "parsed_retrieved_chunk_ids_json",
    "parsed_retrieval_queries_json",
    "parsed_flare_trace_json",
    "parsed_trace_num_steps",
    "parsed_trace_needs_retrieval_steps",
    "parsed_trace_supported_steps",
    "parsed_trace_corrected_steps",
    "parsed_trace_not_enough_info_steps",
    "parsed_trace_refuted_steps",
    "parsed_trace_not_checked_steps",
    "parsed_trace_abstain_steps",
    "parsed_first_retrieval_rule",
    "parsed_retrieval_rules_json",
    "parsed_support_statuses",
    "parsed_support_statuses_json",
    "parsed_final_support_status",
    "parsed_any_supported_or_corrected",
    "parsed_any_not_enough_info_or_refuted",
    "parsed_input_tokens",
    "parsed_output_tokens",
    "parsed_total_tokens",
    "parsed_latency_seconds",
    "valid_json_format",
    "valid_trace_format",
    "valid_answer_format",
    "valid_mc_format",
    "valid_s3_format",
    "parse_method",
    "parse_error",
    "parsed_json",
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
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_json_string(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def coerce_float(value: Any) -> float | None:
    """
    Convierte floats genéricos.

    Nota:
    - Esta función mantiene compatibilidad con valores tipo porcentaje.
    - No debe usarse para latencias, porque una latencia de 15.7 segundos
      no debe convertirse en 0.157.
    """
    if is_missing(value):
        return None

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
        return None

    if 1.0 < number <= 100.0:
        number = number / 100.0

    return number


def coerce_plain_float(value: Any) -> float | None:
    """
    Convierte floats sin normalización porcentual.

    Usar para:
    - latency_seconds
    - row_wall_latency_seconds
    - scores o métricas que no sean probabilidades/porcentajes
    """
    if is_missing(value):
        return None

    try:
        if isinstance(value, str):
            text = value.strip().replace(",", ".")
            if text.endswith("%"):
                return float(text[:-1]) / 100.0
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_confidence(value: Any) -> float | None:
    number = coerce_float(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    return None


def coerce_int(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False

    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def contains_any(text: str, markers: list[str]) -> bool:
    lower = normalize_text(text)
    return any(marker in lower for marker in markers)


def infer_abstention_from_text(text: str) -> bool:
    return contains_any(text, ABSTENTION_MARKERS)


def infer_clarification_from_text(text: str) -> bool:
    return contains_any(text, CLARIFICATION_MARKERS)


def unique_preserve_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def parse_pipe_or_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return unique_preserve_order(value)

    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return unique_preserve_order(parsed)
        if isinstance(parsed, str):
            return [parsed] if parsed else []
    except json.JSONDecodeError:
        pass

    if "|" in text:
        return unique_preserve_order(text.split("|"))

    if "," in text and not text.strip().startswith("{"):
        return unique_preserve_order(text.split(","))

    return [text]


def join_pipe(values: list[Any]) -> str:
    return "|".join(unique_preserve_order(values))


# ---------------------------------------------------------------------------
# Extracción robusta de JSON
# ---------------------------------------------------------------------------

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


def load_json_object(value: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(value, dict):
        return value, ""

    text = clean_text(value)
    if not text:
        return None, "Texto JSON vacío."

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


def load_json_list(value: Any) -> tuple[list[Any], str]:
    if isinstance(value, list):
        return value, ""

    text = clean_text(value)
    if not text:
        return [], "Lista JSON vacía."

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"Lista JSON inválida: {exc}"

    if not isinstance(parsed, list):
        return [], "El JSON parseado no es una lista."

    return parsed, ""


# ---------------------------------------------------------------------------
# Parse S3 específico
# ---------------------------------------------------------------------------

def get_row_value(row: pd.Series, key: str, default: Any = "") -> Any:
    if key in row.index:
        value = row.get(key, default)
        if not is_missing(value):
            return value
    return default


def parse_s3_json_from_row(row: pd.Series) -> tuple[dict[str, Any], str, str]:
    """
    Devuelve (obj, parse_method, parse_error).

    Prioridad:
    1. raw_response_json
    2. raw_output
    3. fallback desde columnas finales del runner
    """
    errors: list[str] = []

    for col in ["raw_response_json", "raw_output"]:
        raw = get_row_value(row, col, "")
        if clean_text(raw):
            obj, err = load_json_object(raw)
            if obj is not None:
                return obj, col, ""
            errors.append(f"{col}: {err}")

    # Fallback usando columnas ya normalizadas por el runner.
    fallback = {
        "answer": get_row_value(row, "final_answer", ""),
        "confidence": get_row_value(row, "final_confidence", None),
        "abstained": get_row_value(row, "final_abstained", None),
        "retrieval_mode": "active",
        "num_generation_steps": get_row_value(row, "num_generation_steps", None),
        "num_retrieval_steps": get_row_value(row, "num_retrieval_steps", None),
        "num_chunks_retrieved_total": get_row_value(row, "num_chunks_retrieved_total", None),
        "evidence_ids": parse_pipe_or_json_list(get_row_value(row, "evidence_ids", "")),
        "flare_trace": [],
        "input_tokens": get_row_value(row, "input_tokens", None),
        "output_tokens": get_row_value(row, "output_tokens", None),
        "total_tokens": get_row_value(row, "total_tokens", None),
        "latency_seconds": get_row_value(row, "latency_seconds", None),
        "error": get_row_value(row, "error", ""),
    }

    trace_raw = get_row_value(row, "flare_trace_json", "")
    trace, trace_err = load_json_list(trace_raw)
    if trace:
        fallback["flare_trace"] = trace
    elif clean_text(trace_raw):
        errors.append(f"flare_trace_json: {trace_err}")

    return fallback, "columns_fallback", "; ".join(errors)


def normalize_support_status(value: Any) -> str:
    status = clean_text(value).lower()
    if status in VALID_SUPPORT_STATUSES:
        return status
    return ""


def trace_to_list(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    output: list[dict[str, Any]] = []
    for item in trace:
        if isinstance(item, dict):
            output.append(item)
    return output


def extract_chunk_ids_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for step in trace:
        step_ids = step.get("retrieved_chunk_ids", [])
        if isinstance(step_ids, list):
            ids.extend(step_ids)
        else:
            ids.extend(parse_pipe_or_json_list(step_ids))
    return unique_preserve_order(ids)


def extract_retrieval_queries_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for step in trace:
        query = clean_text(step.get("retrieval_query", ""))
        if query:
            queries.append(query)
    return unique_preserve_order(queries)


def extract_rules_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    rules: list[str] = []
    for step in trace:
        rule = clean_text(step.get("retrieval_rule_name", ""))
        if rule:
            rules.append(rule)
    return rules


def extract_support_statuses(trace: list[dict[str, Any]]) -> list[str]:
    statuses: list[str] = []
    for step in trace:
        status = normalize_support_status(step.get("support_status", ""))
        if status:
            statuses.append(status)
    return statuses


def count_status(statuses: list[str], status: str) -> int:
    return int(sum(1 for item in statuses if item == status))


def count_abstain_steps(trace: list[dict[str, Any]]) -> int:
    return int(sum(1 for step in trace if coerce_bool(step.get("abstain_sentence")) is True))


def count_needs_retrieval_steps(trace: list[dict[str, Any]]) -> int:
    return int(sum(1 for step in trace if coerce_bool(step.get("needs_retrieval")) is True))


def infer_mc_choice(answer: str) -> str:
    text = clean_text(answer)
    if not text:
        return ""

    stripped = text.strip()

    if re.fullmatch(r"[ABCDabcd]", stripped):
        return stripped.upper()

    patterns = [
        r"^\(?\s*([ABCDabcd])\s*\)?[\.\):\-]?\s*$",
        r"^\(?\s*([ABCDabcd])\s*\)?[\.\):\-]\s+",
        r"\b(?:answer|option|choice|respuesta|opción|opcion)\s*(?:is|es|:)?\s*\(?\s*([ABCDabcd])\s*\)?\b",
        r"\b(?:the\s+answer\s+is)\s*\(?\s*([ABCDabcd])\s*\)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


def infer_valid_answer_format(
    *,
    answer: str,
    parsed_choice: str,
    task_type: str,
    question_format: str,
) -> tuple[bool, bool | None]:
    answer_present = bool(clean_text(answer))

    is_mc = task_type == "multiple_choice" or question_format == "multiple_choice"
    if is_mc:
        valid_mc = parsed_choice in {"A", "B", "C", "D"}
        return bool(answer_present and valid_mc), valid_mc

    return answer_present, None


def parse_s3_row(row: pd.Series) -> dict[str, Any]:
    obj, parse_method, parse_error = parse_s3_json_from_row(row)

    answer = clean_text(obj.get("answer", ""))
    if not answer:
        answer = clean_text(get_row_value(row, "final_answer", ""))

    confidence = coerce_confidence(obj.get("confidence"))
    if confidence is None:
        confidence = coerce_confidence(get_row_value(row, "final_confidence", None))

    abstained = coerce_bool(obj.get("abstained"))
    if abstained is None:
        abstained = coerce_bool(get_row_value(row, "final_abstained", None))
    if abstained is None:
        abstained = infer_abstention_from_text(answer)

    clarification = infer_clarification_from_text(answer)

    retrieval_mode = clean_text(obj.get("retrieval_mode", "")) or "active"
    retrieval_mode_norm = retrieval_mode.lower()
    if retrieval_mode_norm not in VALID_RETRIEVAL_MODES:
        retrieval_mode_norm = retrieval_mode or "active"

    trace = trace_to_list(obj.get("flare_trace", []))
    if not trace:
        trace_raw = get_row_value(row, "flare_trace_json", "")
        loaded_trace, trace_err = load_json_list(trace_raw)
        trace = trace_to_list(loaded_trace)
        if clean_text(trace_raw) and not trace:
            parse_error = "; ".join(x for x in [parse_error, f"trace: {trace_err}"] if x)

    evidence_ids = obj.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = parse_pipe_or_json_list(evidence_ids)
    if not evidence_ids:
        evidence_ids = parse_pipe_or_json_list(get_row_value(row, "evidence_ids", ""))

    retrieved_chunk_ids = extract_chunk_ids_from_trace(trace)
    if not retrieved_chunk_ids:
        retrieved_chunk_ids = parse_pipe_or_json_list(get_row_value(row, "retrieved_chunk_ids", ""))

    retrieval_queries = extract_retrieval_queries_from_trace(trace)
    if not retrieval_queries:
        retrieval_queries_raw = get_row_value(row, "retrieval_queries_json", "")
        loaded_queries, _ = load_json_list(retrieval_queries_raw)
        retrieval_queries = unique_preserve_order(loaded_queries)

    support_statuses = extract_support_statuses(trace)
    rules = extract_rules_from_trace(trace)

    parsed_num_generation_steps = coerce_int(obj.get("num_generation_steps"))
    if parsed_num_generation_steps is None:
        parsed_num_generation_steps = coerce_int(get_row_value(row, "num_generation_steps", None))
    if parsed_num_generation_steps is None:
        parsed_num_generation_steps = len(trace)

    parsed_num_retrieval_steps = coerce_int(obj.get("num_retrieval_steps"))
    if parsed_num_retrieval_steps is None:
        parsed_num_retrieval_steps = coerce_int(get_row_value(row, "num_retrieval_steps", None))
    if parsed_num_retrieval_steps is None:
        parsed_num_retrieval_steps = count_needs_retrieval_steps(trace)

    parsed_num_chunks = coerce_int(obj.get("num_chunks_retrieved_total"))
    if parsed_num_chunks is None:
        parsed_num_chunks = coerce_int(get_row_value(row, "num_chunks_retrieved_total", None))
    if parsed_num_chunks is None:
        parsed_num_chunks = len(retrieved_chunk_ids)

    active_retrieval_triggered = (
        bool(parsed_num_retrieval_steps and parsed_num_retrieval_steps > 0)
        or bool(parsed_num_chunks and parsed_num_chunks > 0)
        or bool(retrieved_chunk_ids)
    )

    task_type = clean_text(get_row_value(row, "task_type", ""))
    question_format = clean_text(get_row_value(row, "question_format", ""))
    parsed_choice = infer_mc_choice(answer)

    valid_answer_format, valid_mc_format = infer_valid_answer_format(
        answer=answer,
        parsed_choice=parsed_choice,
        task_type=task_type,
        question_format=question_format,
    )

    run_error = clean_text(obj.get("error", "")) or clean_text(get_row_value(row, "error", ""))
    run_error_present = bool(run_error)

    valid_json_format = parse_method in {"raw_response_json", "raw_output"} and not parse_error
    valid_trace_format = isinstance(trace, list)
    valid_s3_format = bool(
        clean_text(answer)
        and confidence is not None
        and abstained is not None
        and valid_trace_format
        and not run_error_present
    )

    final_support_status = support_statuses[-1] if support_statuses else ""

    return {
        "run_error_present": run_error_present,
        "parsed_answer": answer,
        "parsed_answer_normalized": normalize_text(answer),
        "parsed_choice": parsed_choice,
        "parsed_confidence": confidence,
        "parsed_abstained": bool(abstained),
        "parsed_clarification": bool(clarification),
        "parsed_retrieval_mode": retrieval_mode_norm,
        "parsed_active_retrieval_triggered": bool(active_retrieval_triggered),
        "parsed_num_generation_steps": parsed_num_generation_steps,
        "parsed_num_retrieval_steps": parsed_num_retrieval_steps,
        "parsed_num_chunks_retrieved_total": parsed_num_chunks,
        "parsed_unique_chunks_retrieved": len(retrieved_chunk_ids),
        "parsed_evidence_ids": join_pipe(evidence_ids),
        "parsed_evidence_ids_json": to_json_string(unique_preserve_order(evidence_ids)),
        "parsed_retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),
        "parsed_retrieved_chunk_ids_json": to_json_string(retrieved_chunk_ids),
        "parsed_retrieval_queries_json": to_json_string(retrieval_queries),
        "parsed_flare_trace_json": to_json_string(trace),
        "parsed_trace_num_steps": len(trace),
        "parsed_trace_needs_retrieval_steps": count_needs_retrieval_steps(trace),
        "parsed_trace_supported_steps": count_status(support_statuses, "supported"),
        "parsed_trace_corrected_steps": count_status(support_statuses, "corrected"),
        "parsed_trace_not_enough_info_steps": count_status(support_statuses, "not_enough_info"),
        "parsed_trace_refuted_steps": count_status(support_statuses, "refuted"),
        "parsed_trace_not_checked_steps": count_status(support_statuses, "not_checked"),
        "parsed_trace_abstain_steps": count_abstain_steps(trace),
        "parsed_first_retrieval_rule": next((rule for rule in rules if rule), ""),
        "parsed_retrieval_rules_json": to_json_string(rules),
        "parsed_support_statuses": "|".join(support_statuses),
        "parsed_support_statuses_json": to_json_string(support_statuses),
        "parsed_final_support_status": final_support_status,
        "parsed_any_supported_or_corrected": any(s in {"supported", "corrected"} for s in support_statuses),
        "parsed_any_not_enough_info_or_refuted": any(s in {"not_enough_info", "refuted"} for s in support_statuses),
        "parsed_input_tokens": coerce_int(obj.get("input_tokens")) if coerce_int(obj.get("input_tokens")) is not None else coerce_int(get_row_value(row, "input_tokens", None)),
        "parsed_output_tokens": coerce_int(obj.get("output_tokens")) if coerce_int(obj.get("output_tokens")) is not None else coerce_int(get_row_value(row, "output_tokens", None)),
        "parsed_total_tokens": coerce_int(obj.get("total_tokens")) if coerce_int(obj.get("total_tokens")) is not None else coerce_int(get_row_value(row, "total_tokens", None)),
        "parsed_latency_seconds": coerce_plain_float(obj.get("latency_seconds")) if coerce_plain_float(obj.get("latency_seconds")) is not None else coerce_plain_float(get_row_value(row, "latency_seconds", None)),
        "valid_json_format": bool(valid_json_format),
        "valid_trace_format": bool(valid_trace_format),
        "valid_answer_format": bool(valid_answer_format),
        "valid_mc_format": valid_mc_format,
        "valid_s3_format": bool(valid_s3_format and valid_answer_format),
        "parse_method": parse_method,
        "parse_error": clean_text(parse_error),
        "parsed_json": to_json_string(obj),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def parse_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    df = pd.read_csv(input_path)
    parsed_rows = [parse_s3_row(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)

    # Evita duplicar columnas si el input ya estaba parseado.
    df = df.drop(columns=[c for c in ADDITIONAL_COLUMNS if c in df.columns], errors="ignore")

    output_df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def mean_bool(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None

    values: list[bool] = []
    for value in cleaned:
        coerced = coerce_bool(value)
        if coerced is not None:
            values.append(coerced)

    if not values:
        return None

    return sum(values) / len(values)


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


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    print(f"\nParsed S3 guardado en: {output_path}")
    print("\nResumen parse S3")
    print("----------------")
    print(f"Filas: {len(df)}")

    if "s2_case_type" in df.columns:
        print("\nTipos de caso:")
        print(df["s2_case_type"].value_counts(dropna=False).to_string())

    print(f"\nRun error rate: {fmt(mean_bool(df['run_error_present']))}")
    print(f"Valid S3 format rate: {fmt(mean_bool(df['valid_s3_format']))}")
    print(f"Valid answer format rate: {fmt(mean_bool(df['valid_answer_format']))}")
    print(f"Abstention rate: {fmt(mean_bool(df['parsed_abstained']))}")
    print(f"Clarification rate: {fmt(mean_bool(df['parsed_clarification']))}")
    print(f"Active retrieval triggered rate: {fmt(mean_bool(df['parsed_active_retrieval_triggered']))}")
    print(f"Avg generation steps: {fmt(mean_numeric(df['parsed_num_generation_steps']))}")
    print(f"Avg retrieval steps: {fmt(mean_numeric(df['parsed_num_retrieval_steps']))}")
    print(f"Avg chunks retrieved total: {fmt(mean_numeric(df['parsed_num_chunks_retrieved_total']))}")
    print(f"Avg total tokens: {fmt(mean_numeric(df['parsed_total_tokens']))}")
    print(f"Avg latency seconds: {fmt(mean_numeric(df['parsed_latency_seconds']))}")

    if "task_type" in df.columns:
        print("\nResumen por task_type:")
        grouped = df.groupby("task_type", dropna=False).agg(
            n=("id", "count") if "id" in df.columns else ("parsed_answer", "count"),
            abstention_rate=("parsed_abstained", lambda s: mean_bool(s)),
            retrieval_rate=("parsed_active_retrieval_triggered", lambda s: mean_bool(s)),
            avg_steps=("parsed_num_generation_steps", lambda s: mean_numeric(s)),
            valid_format_rate=("valid_s3_format", lambda s: mean_bool(s)),
        )
        print(grouped.to_string())

    errors = df[df["parse_error"].fillna("").astype(str).str.strip().ne("")]
    if not errors.empty:
        print("\nPrimeros parse_error:")
        for _, row in errors.head(5).iterrows():
            row_id = row.get("id", "")
            print(f"- id={row_id}: {row.get('parse_error', '')}")

    bad_format = df[~df["valid_s3_format"].astype(bool)]
    if not bad_format.empty:
        print("\nPrimeras filas con valid_s3_format=False:")
        cols = [c for c in ["id", "s2_case_type", "task_type", "parse_method", "parse_error", "parsed_answer"] if c in bad_format.columns]
        print(bad_format[cols].head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parsea outputs crudos de S3 FLARE-like active retrieval."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV raw generado por run_s3_flare_like.py.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV parseado de salida.",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="No imprime resumen al terminar.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output_df = parse_file(
        input_path=args.input_path,
        output_path=args.output_path,
    )

    if not args.no_summary:
        print_summary(output_df, args.output_path)


if __name__ == "__main__":
    main()

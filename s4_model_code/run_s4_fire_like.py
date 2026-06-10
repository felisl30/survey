#!/usr/bin/env python3
"""
run_s4_fire_like.py

Runner completo para S4: FIRE-like / FIRE-inspired claim verification.

Entrada típica:
    outputs/s2/generation/adaptive_rag_s2_parsed.csv

Índice típico:
    indexes/s2/adaptive_rag

Salida típica:
    outputs/s4/generation/fire_like_s4_raw.csv

Diseño
------
Este runner recorre un CSV de respuestas ya generadas por S2 o S3 y, para cada fila:

    1. Obtiene la pregunta original.
    2. Obtiene la respuesta inicial, normalmente parsed_answer.
    3. Opcionalmente toma evidencia ya recuperada por S2/S3 desde retrieved_context_json.
    4. Ejecuta run_fire_controller_for_answer(...).
    5. Guarda respuesta final S4, claims, veredictos, evidence_ids, trace, tokens,
       latencia y errores.
    6. Permite limit, resume, dry-run y guardado incremental.

Este archivo asume que ya existen:
    s4_model_code/prompts_s4.py
    s4_model_code/claim_extractor_s4.py
    s4_model_code/fire_verifier_s4.py
    s4_model_code/fire_controller_s4.py

Uso recomendado
---------------

Smoke test realista sobre 5 respuestas parseadas de S2, usando índice real:

python s4_model_code/run_s4_fire_like.py \
  --input-path outputs/s2/generation/adaptive_rag_s2_parsed.csv \
  --index-dir indexes/s2/adaptive_rag \
  --output-path outputs/s4/generation/fire_like_s4_raw_test_5.csv \
  --limit 5 \
  --use-index \
  --claim-strategy rules \
  --verification-strategy rules \
  --query-strategy rules \
  --repair-strategy rules \
  --initial-evidence-mode auto \
  --max-claims 4 \
  --max-rounds-per-claim 3 \
  --top-k-per-round 5 \
  --max-total-retrievals 8 \
  --max-total-chunks 12

Smoke test sin índice ni API:

python s4_model_code/run_s4_fire_like.py \
  --input-path outputs/s2/generation/adaptive_rag_s2_parsed.csv \
  --output-path outputs/s4/generation/fire_like_s4_raw_test_5_dry.csv \
  --limit 5 \
  --dry-run \
  --no-index
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths e imports del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S4_CODE_DIR = Path(__file__).resolve().parent
S2_CODE_DIR = PROJECT_ROOT / "s2_model_code"
S1_CODE_DIR = PROJECT_ROOT / "s1_model_code"
S3_CODE_DIR = PROJECT_ROOT / "s3_model_code"

for path in [PROJECT_ROOT, S4_CODE_DIR, S3_CODE_DIR, S2_CODE_DIR, S1_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from fire_controller_s4 import (
        DEFAULT_INDEX_DIR,
        DEFAULT_MAX_CLAIMS,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        DEFAULT_MAX_ROUNDS_PER_CLAIM,
        DEFAULT_MAX_TOTAL_CHUNKS,
        DEFAULT_MAX_TOTAL_RETRIEVALS,
        DEFAULT_TOP_K_PER_ROUND,
        create_retriever,
        run_fire_controller_for_answer,
    )
except ModuleNotFoundError:
    from s4_model_code.fire_controller_s4 import (
        DEFAULT_INDEX_DIR,
        DEFAULT_MAX_CLAIMS,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        DEFAULT_MAX_ROUNDS_PER_CLAIM,
        DEFAULT_MAX_TOTAL_CHUNKS,
        DEFAULT_MAX_TOTAL_RETRIEVALS,
        DEFAULT_TOP_K_PER_ROUND,
        create_retriever,
        run_fire_controller_for_answer,
    )


# ---------------------------------------------------------------------------
# Defaults y columnas
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = Path("outputs/s2/generation/adaptive_rag_s2_parsed.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s4/generation/fire_like_s4_raw.csv")

QUESTION_COLUMNS_PRIORITY = [
    "s2_question",
    "s3_question",
    "routing_question",
    "original_question",
    "question",
    "prompt",
    "source_prompt",
]

INITIAL_ANSWER_COLUMNS_PRIORITY = [
    "parsed_answer",
    "answer",
    "s3_answer",
    "final_answer",
    "initial_answer",
]

VALID_INITIAL_EVIDENCE_MODES = {
    "auto",
    "from_input",
    "none",
}

VALID_SOURCE_SYSTEMS = {
    "auto",
    "s2",
    "s3",
    "manual",
}

COLUMNS_TO_COPY_IF_PRESENT = [
    # Identificación
    "id",
    "source_system",
    "source_question_id",
    "source_dataset",
    "dataset",
    "case_type",
    "s2_case_type",
    "task_type",
    "question_format",

    # Decisión esperada / evaluación
    "expected_route",
    "acceptable_routes_json",
    "requires_retrieval",
    "retrieval_mode",
    "expected_behavior",
    "expected_final_behavior",

    # Texto
    "routing_question",
    "original_question",
    "question",
    "prompt",
    "source_prompt",
    "s2_question",
    "s3_question",

    # Metadata
    "subject",
    "topic",
    "difficulty",
    "level",
    "hotpot_type",
    "source",
    "source_split",
    "original_hotpotqa_id",
    "truthfulqa_category",
    "truthfulqa_type",
    "original_source",

    # Opciones / labels
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_idx",
    "gold_answer_text",
    "expected_answer",
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",

    # Evidencia esperada
    "gold_evidence_ids",
    "gold_evidence_titles",
    "context_chunk_ids",

    # Diagnóstico de sistema fuente
    "predicted_route",
    "predicted_retrieval_mode",
    "parsed_route",
    "parsed_retrieval_mode",
    "generation_policy",
    "retrieval_query",
    "retrieved_chunk_ids",
    "retrieved_doc_ids",
    "retrieved_titles",
    "retrieved_scores",
    "retrieved_context_json",

    # Control experimental
    "is_synthetic",
    "synthetic_strategy",
    "expected_model_output_format",
    "evaluation_notes",
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


# ---------------------------------------------------------------------------
# Extracción de pregunta, respuesta inicial y evidencia
# ---------------------------------------------------------------------------

def get_first_nonempty(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value
    return ""


def get_question_for_s4(row: pd.Series) -> str:
    question = get_first_nonempty(row, QUESTION_COLUMNS_PRIORITY)
    if not question:
        row_id = clean_text(row.get("id", "<sin id>"))
        raise ValueError(
            f"Fila id={row_id} sin pregunta válida. "
            f"Se esperaba alguna columna de {QUESTION_COLUMNS_PRIORITY}."
        )
    return question


def parse_answer_from_raw_output(raw_output: str) -> str:
    obj = safe_json_loads(raw_output)
    if isinstance(obj, dict):
        answer = clean_text(obj.get("answer", ""))
        if answer:
            return answer

    # Fallback: usar el texto crudo solo si existe.
    return clean_text(raw_output)


def get_initial_answer_for_s4(row: pd.Series) -> str:
    answer = get_first_nonempty(row, INITIAL_ANSWER_COLUMNS_PRIORITY)
    if answer:
        return answer

    raw_output = clean_text(row.get("raw_output", ""))
    if raw_output:
        answer = parse_answer_from_raw_output(raw_output)
        if answer:
            return answer

    row_id = clean_text(row.get("id", "<sin id>"))
    raise ValueError(
        f"Fila id={row_id} sin respuesta inicial válida. "
        f"Se esperaba parsed_answer, answer, s3_answer, final_answer, initial_answer o raw_output JSON."
    )


def infer_source_system(row: pd.Series, explicit_source_system: str) -> str:
    explicit_source_system = clean_text(explicit_source_system).lower() or "auto"

    if explicit_source_system != "auto":
        return explicit_source_system

    system = clean_text(row.get("system", "")).lower()
    if "s3" in system or "flare" in system:
        return "s3"
    if "s2" in system or "adaptive" in system:
        return "s2"

    if "parsed_active_retrieval_triggered" in row.index or "flare_trace" in row.index:
        return "s3"

    return "s2"


def infer_question_type(row: pd.Series) -> str:
    for col in ["s2_case_type", "task_type", "case_type", "dataset"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def infer_expected_behavior(row: pd.Series) -> str:
    for col in ["expected_final_behavior", "expected_behavior"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def sanitize_chunk(item: dict[str, Any], *, fallback_rank: int = 1) -> dict[str, Any]:
    return {
        "rank": item.get("rank", fallback_rank),
        "chunk_id": clean_text(item.get("chunk_id", f"chunk_input_{fallback_rank}")),
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


def parse_chunks_json(value: Any) -> list[dict[str, Any]]:
    parsed = safe_json_loads(value)
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    chunks: list[dict[str, Any]] = []
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        chunk = sanitize_chunk(item, fallback_rank=i)
        if chunk["text"]:
            chunks.append(chunk)

    return chunks


def get_initial_evidence_for_s4(
    row: pd.Series,
    *,
    initial_evidence_mode: str,
) -> list[dict[str, Any]]:
    initial_evidence_mode = clean_text(initial_evidence_mode).lower() or "auto"
    if initial_evidence_mode not in VALID_INITIAL_EVIDENCE_MODES:
        raise ValueError(
            f"initial_evidence_mode inválido: {initial_evidence_mode}. "
            f"Opciones: {sorted(VALID_INITIAL_EVIDENCE_MODES)}"
        )

    if initial_evidence_mode == "none":
        return []

    # Lo más útil cuando venimos desde S2/S3: reutilizar contexto ya recuperado.
    if "retrieved_context_json" in row.index:
        chunks = parse_chunks_json(row.get("retrieved_context_json", ""))
        if chunks:
            return chunks

    # Compatibilidad con nombres posibles de S3.
    for col in ["retrieved_chunks_json", "evidence_json", "context_json"]:
        if col in row.index:
            chunks = parse_chunks_json(row.get(col, ""))
            if chunks:
                return chunks

    return []


# ---------------------------------------------------------------------------
# Construcción de salida
# ---------------------------------------------------------------------------

def flatten_claim_results(claim_results: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [clean_text(c.get("verdict", "")) for c in claim_results]
    claim_ids = [clean_text(c.get("claim_id", "")) for c in claim_results]
    claim_texts = [clean_text(c.get("claim_text", "")) for c in claim_results]

    supported_ids = [
        clean_text(c.get("claim_id", ""))
        for c in claim_results
        if clean_text(c.get("verdict", "")) == "supported"
    ]
    refuted_ids = [
        clean_text(c.get("claim_id", ""))
        for c in claim_results
        if clean_text(c.get("verdict", "")) == "refuted"
    ]
    nei_ids = [
        clean_text(c.get("claim_id", ""))
        for c in claim_results
        if clean_text(c.get("verdict", "")) == "not_enough_info"
    ]

    supporting_chunk_ids: list[str] = []
    refuting_chunk_ids: list[str] = []
    evidence_ids: list[str] = []
    retrieved_chunk_ids: list[str] = []

    for c in claim_results:
        supporting_chunk_ids.extend(c.get("supporting_chunk_ids", []) or [])
        refuting_chunk_ids.extend(c.get("refuting_chunk_ids", []) or [])
        evidence_ids.extend(c.get("evidence_ids", []) or [])
        retrieved_chunk_ids.extend(c.get("retrieved_chunk_ids", []) or [])

    return {
        "s4_claim_ids": join_pipe(claim_ids),
        "s4_claim_texts": safe_json_dumps(claim_texts),
        "s4_claim_verdicts": join_pipe(verdicts),
        "s4_supported_claim_ids": join_pipe(supported_ids),
        "s4_refuted_claim_ids": join_pipe(refuted_ids),
        "s4_nei_claim_ids": join_pipe(nei_ids),
        "s4_supporting_chunk_ids": join_pipe(supporting_chunk_ids),
        "s4_refuting_chunk_ids": join_pipe(refuting_chunk_ids),
        "s4_evidence_ids": join_pipe(evidence_ids),
        "s4_retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),
    }


def build_output_row(
    input_row: pd.Series,
    *,
    result: dict[str, Any],
    question: str,
    initial_answer: str,
    source_system: str,
    question_type: str,
    expected_behavior: str,
    initial_evidence: list[dict[str, Any]],
    run_error: str,
    dry_run: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    # Conserva metadata original útil para evaluación/comparación.
    for col in COLUMNS_TO_COPY_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    # Conserva campos originales de respuesta bajo prefijo source_.
    output["source_system_for_s4"] = source_system
    output["source_question_for_s4"] = question
    output["source_initial_answer"] = initial_answer
    output["source_parsed_answer"] = clean_text(input_row.get("parsed_answer", ""))
    output["source_parsed_confidence"] = input_row.get("parsed_confidence", "")
    output["source_parsed_abstained"] = input_row.get("parsed_abstained", "")
    output["source_raw_output"] = clean_text(input_row.get("raw_output", ""))
    output["source_model"] = clean_text(input_row.get("model", ""))
    output["source_total_tokens"] = input_row.get("total_tokens", "")
    output["source_latency_seconds"] = input_row.get("latency_seconds", "")

    # Campos compatibles con evaluadores tipo S2: parsed_answer pasa a ser la respuesta final S4.
    output["system"] = "S4_fire_like"
    output["model"] = clean_text(result.get("repair", {}).get("repair_model", "")) or clean_text(result.get("extraction", {}).get("extractor_model", ""))
    output["raw_output"] = safe_json_dumps(result)
    output["parsed_answer"] = clean_text(result.get("answer", ""))
    output["parsed_confidence"] = result.get("confidence", None)
    output["parsed_abstained"] = bool(result.get("abstained", False))
    output["valid_answer_format"] = not bool(run_error) and bool(clean_text(result.get("answer", "")))
    output["valid_format"] = output["valid_answer_format"]
    output["parse_method"] = "s4_structured_output"
    output["parse_error"] = run_error
    output["parsed_json"] = safe_json_dumps(
        {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", None),
            "abstained": result.get("abstained", False),
            "final_decision": result.get("final_decision", ""),
        }
    )

    # Campos específicos S4.
    output["s4_question"] = question
    output["s4_initial_answer"] = initial_answer
    output["s4_answer"] = clean_text(result.get("answer", ""))
    output["s4_confidence"] = result.get("confidence", None)
    output["s4_abstained"] = bool(result.get("abstained", False))
    output["s4_final_decision"] = clean_text(result.get("final_decision", ""))
    output["s4_correction_applied"] = bool(result.get("correction_applied", False))
    output["s4_unsupported_claims_removed"] = join_pipe(result.get("unsupported_claims_removed", []) or [])
    output["s4_corrected_claims"] = safe_json_dumps(result.get("corrected_claims", []) or [])
    output["s4_num_claims"] = safe_int(result.get("num_claims", 0), default=0)
    output["s4_num_supported_claims"] = safe_int(result.get("num_supported_claims", 0), default=0)
    output["s4_num_refuted_claims"] = safe_int(result.get("num_refuted_claims", 0), default=0)
    output["s4_num_nei_claims"] = safe_int(result.get("num_nei_claims", 0), default=0)
    output["s4_num_claims_requiring_evidence"] = safe_int(result.get("num_claims_requiring_evidence", 0), default=0)
    output["s4_num_verification_rounds"] = safe_int(result.get("num_verification_rounds", 0), default=0)
    output["s4_num_retrieval_rounds"] = safe_int(result.get("num_retrieval_rounds", 0), default=0)
    output["s4_num_chunks_retrieved_total"] = safe_int(result.get("num_chunks_retrieved_total", 0), default=0)
    output["s4_initial_evidence_chunk_ids"] = join_pipe([c.get("chunk_id", "") for c in initial_evidence])
    output["s4_initial_evidence_count"] = int(len(initial_evidence))

    flat = flatten_claim_results(result.get("claim_results", []) or [])
    output.update(flat)

    output["s4_claims_json"] = safe_json_dumps(result.get("claims", []) or [])
    output["s4_claim_results_json"] = safe_json_dumps(result.get("claim_results", []) or [])
    output["s4_claim_trace_json"] = safe_json_dumps(result.get("claim_trace", []) or [])
    output["s4_extraction_json"] = safe_json_dumps(result.get("extraction", {}) or {})
    output["s4_repair_json"] = safe_json_dumps(result.get("repair", {}) or {})

    output["s4_claim_strategy"] = clean_text(result.get("claim_strategy", ""))
    output["s4_verification_strategy"] = clean_text(result.get("verification_strategy", ""))
    output["s4_query_strategy"] = clean_text(result.get("query_strategy", ""))
    output["s4_repair_strategy"] = clean_text(result.get("repair_strategy", ""))

    # Tokens/latencias S4.
    output["input_tokens"] = result.get("input_tokens", 0)
    output["output_tokens"] = result.get("output_tokens", 0)
    output["total_tokens"] = result.get("total_tokens", 0)
    output["latency_seconds"] = result.get("latency_seconds", None)
    output["retrieval_latency_seconds"] = result.get("retrieval_latency_seconds", None)

    output["s4_input_tokens"] = result.get("input_tokens", 0)
    output["s4_output_tokens"] = result.get("output_tokens", 0)
    output["s4_total_tokens"] = result.get("total_tokens", 0)
    output["s4_latency_seconds"] = result.get("latency_seconds", None)
    output["s4_retrieval_latency_seconds"] = result.get("retrieval_latency_seconds", None)

    output["s4_question_type"] = question_type
    output["s4_expected_behavior"] = expected_behavior
    output["dry_run"] = dry_run
    output["run_error_present"] = bool(run_error)
    output["error"] = run_error

    return output


# ---------------------------------------------------------------------------
# Validación, resume y summary
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna 'id'.")

    if not (set(QUESTION_COLUMNS_PRIORITY) & set(df.columns)):
        raise ValueError(
            "El CSV debe tener al menos una columna de pregunta: "
            + ", ".join(QUESTION_COLUMNS_PRIORITY)
        )

    if not (set(INITIAL_ANSWER_COLUMNS_PRIORITY) & set(df.columns)) and "raw_output" not in df.columns:
        raise ValueError(
            "El CSV debe tener parsed_answer/answer/s3_answer/final_answer/initial_answer "
            "o raw_output JSON para obtener la respuesta inicial."
        )


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)
    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo existente {output_path} no tiene columna id; "
            "no puedo usar --resume de forma segura."
        )
    return existing


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def summarize_output(df: pd.DataFrame, *, include_groups: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": int(len(df)),
    }

    if df.empty:
        return summary

    if "s4_final_decision" in df.columns:
        summary["final_decision_counts"] = {
            clean_text(k): int(v)
            for k, v in df["s4_final_decision"].value_counts(dropna=False).items()
        }

    if "s4_abstained" in df.columns:
        summary["abstention_rate"] = mean_bool(df["s4_abstained"])

    if "s4_correction_applied" in df.columns:
        summary["correction_rate"] = mean_bool(df["s4_correction_applied"])

    for col in [
        "s4_num_claims",
        "s4_num_supported_claims",
        "s4_num_refuted_claims",
        "s4_num_nei_claims",
        "s4_num_verification_rounds",
        "s4_num_retrieval_rounds",
        "s4_num_chunks_retrieved_total",
        "s4_latency_seconds",
        "s4_retrieval_latency_seconds",
        "s4_total_tokens",
    ]:
        if col in df.columns:
            summary[f"avg_{col}"] = mean_numeric(df[col])

    if "run_error_present" in df.columns:
        summary["run_error_rate"] = mean_bool(df["run_error_present"])

    # Evita recursión infinita:
    # el resumen global puede agrupar por s2_case_type,
    # pero los resúmenes de cada grupo ya no vuelven a reagrupar.
    if include_groups and "s2_case_type" in df.columns:
        summary["by_s2_case_type"] = {
            clean_text(value): summarize_output(subset, include_groups=False)
            for value, subset in df.groupby("s2_case_type", dropna=False)
        }

    return summary


def print_summary(df: pd.DataFrame) -> None:
    summary = summarize_output(df)

    print("\nResumen S4 raw")
    print("--------------")
    print(f"Filas: {summary.get('n', 0)}")

    if "final_decision_counts" in summary:
        print("\nDecisiones finales S4:")
        for key, value in summary["final_decision_counts"].items():
            print(f"- {key}: {value}")

    print(f"\nAbstention rate: {fmt(summary.get('abstention_rate'))}")
    print(f"Correction rate: {fmt(summary.get('correction_rate'))}")
    print(f"Run error rate: {fmt(summary.get('run_error_rate'))}")

    print(f"\nAvg claims: {fmt(summary.get('avg_s4_num_claims'))}")
    print(f"Avg supported claims: {fmt(summary.get('avg_s4_num_supported_claims'))}")
    print(f"Avg refuted claims: {fmt(summary.get('avg_s4_num_refuted_claims'))}")
    print(f"Avg NEI claims: {fmt(summary.get('avg_s4_num_nei_claims'))}")
    print(f"Avg verification rounds: {fmt(summary.get('avg_s4_num_verification_rounds'))}")
    print(f"Avg retrieval rounds: {fmt(summary.get('avg_s4_num_retrieval_rounds'))}")
    print(f"Avg chunks retrieved: {fmt(summary.get('avg_s4_num_chunks_retrieved_total'))}")
    print(f"Avg latency seconds: {fmt(summary.get('avg_s4_latency_seconds'))}")
    print(f"Avg retrieval latency seconds: {fmt(summary.get('avg_s4_retrieval_latency_seconds'))}")
    print(f"Avg total tokens: {fmt(summary.get('avg_s4_total_tokens'))}")


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    input_path: Path,
    index_dir: Path,
    output_path: Path,
    source_system: str,
    limit: int | None,
    resume: bool,
    save_every: int,
    use_index: bool,
    dry_run: bool,
    initial_evidence_mode: str,
    claim_strategy: str,
    verification_strategy: str,
    query_strategy: str,
    repair_strategy: str,
    model: str | None,
    max_retries: int,
    max_claims: int,
    max_rounds_per_claim: int,
    top_k_per_round: int,
    max_total_retrievals: int,
    max_total_chunks: int,
    max_chars_per_chunk: int,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    if initial_evidence_mode not in VALID_INITIAL_EVIDENCE_MODES:
        raise ValueError(
            f"initial_evidence_mode inválido: {initial_evidence_mode}. "
            f"Opciones: {sorted(VALID_INITIAL_EVIDENCE_MODES)}"
        )

    df = pd.read_csv(input_path)
    validate_input(df)

    if limit is not None:
        df = df.head(limit).copy()

    existing = load_existing_results(output_path) if resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending_df = df[~df["id"].astype(str).isin(existing_ids)].copy()

    retriever = None
    if use_index:
        # Carga una sola vez.
        retriever = create_retriever(index_dir)

    print(f"Archivo de entrada: {input_path}")
    print(f"Índice S2/S4: {index_dir}")
    print(f"Archivo de salida: {output_path}")
    print(f"Source system: {source_system}")
    print(f"Use index: {use_index}")
    print(f"Dry run: {dry_run}")
    print(f"Initial evidence mode: {initial_evidence_mode}")
    print(f"Claim strategy: {claim_strategy}")
    print(f"Verification strategy: {verification_strategy}")
    print(f"Query strategy: {query_strategy}")
    print(f"Repair strategy: {repair_strategy}")
    print(f"Modelo: {model or '<default direct_llm.py>'}")
    print(f"Filas consideradas: {len(df)}")
    print(f"Filas existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Running S4 FIRE-like"),
        start=1,
    ):
        question = ""
        initial_answer = ""
        current_source_system = ""
        question_type = ""
        expected_behavior = ""
        initial_evidence: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        run_error = ""

        try:
            question = get_question_for_s4(row)
            initial_answer = get_initial_answer_for_s4(row)
            current_source_system = infer_source_system(row, source_system)
            question_type = infer_question_type(row)
            expected_behavior = infer_expected_behavior(row)
            initial_evidence = get_initial_evidence_for_s4(
                row,
                initial_evidence_mode=initial_evidence_mode,
            )

            result = run_fire_controller_for_answer(
                question=question,
                initial_answer=initial_answer,
                source_system=current_source_system,
                question_type=question_type,
                expected_behavior=expected_behavior,
                initial_evidence=initial_evidence,
                retriever=retriever,
                claim_strategy=claim_strategy,
                verification_strategy=verification_strategy,
                query_strategy=query_strategy,
                repair_strategy=repair_strategy,
                model=model,
                max_retries=max_retries,
                dry_run=dry_run,
                max_claims=max_claims,
                max_rounds_per_claim=max_rounds_per_claim,
                top_k_per_round=top_k_per_round,
                max_total_retrievals=max_total_retrievals,
                max_total_chunks=max_total_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            )

        except Exception as exc:
            run_error = str(exc)
            result = {
                "answer": initial_answer,
                "confidence": 0.0,
                "abstained": False,
                "final_decision": "error",
                "correction_applied": False,
                "unsupported_claims_removed": [],
                "corrected_claims": [],
                "num_claims": 0,
                "num_supported_claims": 0,
                "num_refuted_claims": 0,
                "num_nei_claims": 0,
                "num_claims_requiring_evidence": 0,
                "num_verification_rounds": 0,
                "num_retrieval_rounds": 0,
                "num_chunks_retrieved_total": 0,
                "claims": [],
                "claim_results": [],
                "claim_trace": [],
                "extraction": {},
                "repair": {},
                "claim_strategy": claim_strategy,
                "verification_strategy": verification_strategy,
                "query_strategy": query_strategy,
                "repair_strategy": repair_strategy,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
                "retrieval_latency_seconds": 0.0,
            }

        output_row = build_output_row(
            row,
            result=result,
            question=question,
            initial_answer=initial_answer,
            source_system=current_source_system or source_system,
            question_type=question_type,
            expected_behavior=expected_behavior,
            initial_evidence=initial_evidence,
            run_error=run_error,
            dry_run=dry_run,
        )
        rows.append(output_row)

        if save_every > 0 and i % save_every == 0:
            save_results(rows, output_path)

    save_results(rows, output_path)
    output_df = pd.DataFrame(rows)

    print(f"\nResultados S4 guardados en: {output_path}")
    print_summary(output_df)
    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta S4 FIRE-like sobre salidas parseadas de S2/S3."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV parseado de S2/S3.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio del índice.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV raw de salida S4.",
    )
    parser.add_argument(
        "--source-system",
        choices=sorted(VALID_SOURCE_SYSTEMS),
        default="auto",
        help="Sistema fuente de la respuesta inicial.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Corre solo las primeras N filas.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No repite IDs ya presentes en output-path.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Cada cuántas filas guardar resultados parciales.",
    )

    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--use-index",
        action="store_true",
        help="Usa retriever real desde --index-dir.",
    )
    index_group.add_argument(
        "--no-index",
        action="store_true",
        help="No usa índice real.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No llama API. Con --use-index igual puede recuperar chunks.",
    )
    parser.add_argument(
        "--initial-evidence-mode",
        choices=sorted(VALID_INITIAL_EVIDENCE_MODES),
        default="auto",
        help="Usa evidencia ya recuperada en el CSV de entrada.",
    )

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

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    use_index = bool(args.use_index)
    if args.no_index:
        use_index = False

    run_experiment(
        input_path=args.input_path,
        index_dir=args.index_dir,
        output_path=args.output_path,
        source_system=args.source_system,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        use_index=use_index,
        dry_run=args.dry_run,
        initial_evidence_mode=args.initial_evidence_mode,
        claim_strategy=args.claim_strategy,
        verification_strategy=args.verification_strategy,
        query_strategy=args.query_strategy,
        repair_strategy=args.repair_strategy,
        model=args.model,
        max_retries=args.max_retries,
        max_claims=args.max_claims,
        max_rounds_per_claim=args.max_rounds_per_claim,
        top_k_per_round=args.top_k_per_round,
        max_total_retrievals=args.max_total_retrievals,
        max_total_chunks=args.max_total_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )


if __name__ == "__main__":
    main()

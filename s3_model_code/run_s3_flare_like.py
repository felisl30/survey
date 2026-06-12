#!/usr/bin/env python3
"""
run_s3_flare_like.py

Runner completo para S3: FLARE-like / FLARE-inspired active retrieval.

Entrada por defecto:
    data/s2/adaptive_rag/questions_s2.csv

Índice por defecto:
    indexes/s2/adaptive_rag

Salida por defecto:
    outputs/s3/generation/flare_like_s3_raw.csv

Diseño
------
Este runner recorre un CSV de preguntas y, para cada fila:

    1. Obtiene la pregunta limpia.
    2. Ejecuta run_flare_controller_for_question(...).
    3. Guarda respuesta final, trace FLARE, evidencia, tokens, latencia y errores.
    4. Permite dry-run, limit, resume y guardado incremental.

Este archivo asume que ya existen:

    s3_model_code/prompts_s3.py
    s3_model_code/active_retrieval_policy_s3.py
    s3_model_code/flare_controller_s3.py

Uso recomendado
---------------
Smoke test sin API ni retrieval real:

python s3_model_code/run_s3_flare_like.py \
  --input-path data/s2/adaptive_rag/questions_s2.csv \
  --output-path outputs/s3/generation/flare_like_s3_raw_test_5_dry.csv \
  --limit 5 \
  --dry-run

Smoke test real con 5 preguntas:

python s3_model_code/run_s3_flare_like.py \
  --input-path data/s2/adaptive_rag/questions_s2.csv \
  --index-dir indexes/s2/adaptive_rag \
  --output-path outputs/s3/generation/flare_like_s3_raw_test_5.csv \
  --limit 5 \
  --max-steps 3 \
  --top-k-per-step 2 \
  --max-retrieval-steps 2 \
  --max-total-chunks 4 \
  --retrieval-strategy rules
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths e imports del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_CODE_DIR = Path(__file__).resolve().parent
S2_CODE_DIR = PROJECT_ROOT / "s2_model_code"
S1_CODE_DIR = PROJECT_ROOT / "s1_model_code"

for path in [PROJECT_ROOT, S3_CODE_DIR, S2_CODE_DIR, S1_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from project_paths import S2_INDEX_DIR, S2_QUESTIONS_PATH, S3_RAW_OUTPUT_PATH
    from flare_controller_s3 import (
        DEFAULT_INDEX_DIR,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        DEFAULT_MAX_RETRIEVAL_STEPS,
        DEFAULT_MAX_STEPS,
        DEFAULT_MAX_TOTAL_CHUNKS,
        DEFAULT_TOP_K_PER_STEP,
        create_retriever,
        run_flare_controller_for_question,
    )
except ModuleNotFoundError:
    from project_paths import S2_INDEX_DIR, S2_QUESTIONS_PATH, S3_RAW_OUTPUT_PATH
    from s3_model_code.flare_controller_s3 import (
        DEFAULT_INDEX_DIR,
        DEFAULT_MAX_CHARS_PER_CHUNK,
        DEFAULT_MAX_RETRIEVAL_STEPS,
        DEFAULT_MAX_STEPS,
        DEFAULT_MAX_TOTAL_CHUNKS,
        DEFAULT_TOP_K_PER_STEP,
        create_retriever,
        run_flare_controller_for_question,
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = S2_QUESTIONS_PATH
DEFAULT_OUTPUT_PATH = S3_RAW_OUTPUT_PATH
DEFAULT_INDEX_DIR = S2_INDEX_DIR
DEFAULT_RETRIEVAL_STRATEGY = "rules"

REQUIRED_COLUMNS = {
    "id",
}

QUESTION_COLUMNS_PRIORITY = [
    "routing_question",
    "original_question",
    "question",
    "prompt",
]

VALID_TASK_PROFILE_MODES = {
    "auto",
    "open_qa",
    "open_direct",
    "open_retrieval",
    "multiple_choice",
    "ambiguous",
}

VALID_RETRIEVAL_BIAS_MODES = {
    "auto",
    "conservative",
    "balanced",
    "aggressive",
}

COLUMNS_TO_KEEP_IF_PRESENT = [
    # Identificación S2
    "id",
    "source_system",
    "source_question_id",
    "source_dataset",
    "dataset",
    "case_type",
    "s2_case_type",

    # Decisión esperada / metadata experimental
    "expected_route",
    "acceptable_routes_json",
    "requires_retrieval",
    "retrieval_mode",
    "expected_behavior",
    "expected_final_behavior",

    # Entrada textual
    "routing_question",
    "original_question",
    "question",
    "prompt",
    "source_prompt",

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
    return str(value).strip()


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def parse_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]

    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(x) for x in parsed if clean_text(x)]
    except json.JSONDecodeError:
        pass

    if "|" in text:
        return [clean_text(x) for x in text.split("|") if clean_text(x)]

    return [text]


def join_pipe(values: list[Any]) -> str:
    cleaned: list[str] = []
    for value in values:
        text = clean_text(value)
        if text:
            cleaned.append(text)
    return "|".join(cleaned)


def has_multiple_choice_options(row: pd.Series) -> bool:
    # Caso típico MMLU: columnas A/B/C/D.
    option_cols = ["A", "B", "C", "D"]
    if all(col in row.index and clean_text(row.get(col, "")) for col in option_cols):
        return True

    # Fallback: columna JSON de opciones.
    if "answer_choices_json" in row.index and clean_text(row.get("answer_choices_json", "")):
        choices = parse_json_list(row.get("answer_choices_json", ""))
        return len(choices) >= 2

    return False


def get_base_question(row: pd.Series) -> str:
    for col in QUESTION_COLUMNS_PRIORITY:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value

    row_id = clean_text(row.get("id", "<sin id>"))
    raise ValueError(
        f"Fila id={row_id} sin pregunta válida. "
        f"Se esperaba alguna columna de {QUESTION_COLUMNS_PRIORITY}."
    )


def get_answer_choices(row: pd.Series) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []

    for label in ["A", "B", "C", "D"]:
        if label in row.index:
            text = clean_text(row.get(label, ""))
            if text:
                choices.append((label, text))

    if choices:
        return choices

    raw = clean_text(row.get("answer_choices_json", ""))
    if not raw:
        return choices

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return choices

    if isinstance(parsed, dict):
        for label in ["A", "B", "C", "D"]:
            text = clean_text(parsed.get(label, ""))
            if text:
                choices.append((label, text))
    elif isinstance(parsed, list):
        labels = ["A", "B", "C", "D"]
        for label, text in zip(labels, parsed):
            text = clean_text(text)
            if text:
                choices.append((label, text))

    return choices


def build_multiple_choice_question(row: pd.Series) -> str:
    base_question = get_base_question(row)
    choices = get_answer_choices(row)

    option_block = "\n".join(f"{label}. {text}" for label, text in choices)

    return f"""This is a multiple-choice question.

Question:
{base_question}

Options:
{option_block}

Instruction:
Answer with exactly one option letter among A, B, C, or D."""


def infer_task_profile(
    row: pd.Series,
    *,
    task_profile_mode: str = "auto",
    retrieval_bias_mode: str = "auto",
) -> tuple[str, str, str]:
    """
    Devuelve (task_type, retrieval_bias, question_format).

    No usa gold_answer, expected_answer ni expected_route para contestar.
    Usa el formato de la pregunta y metadata de origen para evitar over-retrieval
    en MMLU/TruthfulQA y mantener retrieval agresivo en HotpotQA.
    """
    if task_profile_mode != "auto":
        task_type = task_profile_mode
    else:
        source_dataset = clean_text(row.get("source_dataset", row.get("dataset", ""))).lower()
        s2_case_type = clean_text(row.get("s2_case_type", row.get("case_type", ""))).lower()
        original_hotpotqa_id = clean_text(row.get("original_hotpotqa_id", ""))

        if has_multiple_choice_options(row):
            task_type = "multiple_choice"
        elif "hotpot" in source_dataset or original_hotpotqa_id or "rag_" in s2_case_type:
            task_type = "open_retrieval"
        elif "truthful" in source_dataset or "truthfulqa" in s2_case_type:
            task_type = "open_direct"
        elif "ambiguous" in s2_case_type:
            task_type = "ambiguous"
        else:
            task_type = "open_qa"

    if retrieval_bias_mode != "auto":
        retrieval_bias = retrieval_bias_mode
    else:
        if task_type in {"multiple_choice", "open_direct", "ambiguous"}:
            retrieval_bias = "conservative"
        elif task_type == "open_retrieval":
            retrieval_bias = "aggressive"
        else:
            retrieval_bias = "balanced"

    question_format = "multiple_choice" if task_type == "multiple_choice" else "open"
    return task_type, retrieval_bias, question_format


def get_question_for_s3(row: pd.Series, *, task_type: str = "open_qa") -> str:
    if task_type == "multiple_choice":
        return build_multiple_choice_question(row)

    return get_base_question(row)


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(missing))
        )

    if not (set(QUESTION_COLUMNS_PRIORITY) & set(df.columns)):
        raise ValueError(
            "El CSV debe tener al menos una columna de pregunta: "
            + ", ".join(QUESTION_COLUMNS_PRIORITY)
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


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def extract_retrieval_queries_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for step in trace:
        query = clean_text(step.get("retrieval_query", ""))
        if query:
            queries.append(query)
    return unique_preserve_order(queries)


def extract_retrieved_chunk_ids_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for step in trace:
        step_ids = step.get("retrieved_chunk_ids", [])
        if isinstance(step_ids, list):
            ids.extend(clean_text(x) for x in step_ids if clean_text(x))
        elif clean_text(step_ids):
            ids.extend(parse_json_list(step_ids))
    return unique_preserve_order(ids)


def max_steps_reached(result: dict[str, Any], configured_max_steps: int) -> bool:
    try:
        num_steps = int(result.get("num_generation_steps", 0))
    except (TypeError, ValueError):
        return False

    if num_steps < configured_max_steps:
        return False

    trace = result.get("flare_trace", [])
    if not isinstance(trace, list) or not trace:
        return False

    last = trace[-1]
    if bool(last.get("abstain_sentence")):
        return False

    final_sentence = clean_text(last.get("final_sentence", ""))
    return bool(final_sentence)


# ---------------------------------------------------------------------------
# Construcción de fila de salida
# ---------------------------------------------------------------------------

def build_output_row(
    input_row: pd.Series,
    *,
    system_name: str,
    model: str | None,
    question: str,
    result: dict[str, Any],
    retrieval_strategy: str,
    task_type: str,
    retrieval_bias: str,
    question_format: str,
    index_dir: Path,
    max_steps: int,
    top_k_per_step: int,
    max_retrieval_steps: int,
    max_total_chunks: int,
    max_chars_per_chunk: int,
    dry_run: bool,
    outer_error: str,
    row_latency_seconds: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for col in COLUMNS_TO_KEEP_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    trace = result.get("flare_trace", [])
    if not isinstance(trace, list):
        trace = []

    evidence_ids = result.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = parse_json_list(evidence_ids)

    retrieved_chunk_ids = extract_retrieved_chunk_ids_from_trace(trace)
    retrieval_queries = extract_retrieval_queries_from_trace(trace)

    result_error = clean_text(result.get("error", ""))
    error = clean_text(outer_error or result_error)

    output.update(
        {
            "system": system_name,
            "model": model or "",
            "s3_question": question,
            "generation_policy": "flare_like_active_retrieval",
            "retrieval_strategy": retrieval_strategy,
            "task_type": task_type,
            "retrieval_bias": retrieval_bias,
            "question_format": question_format,
            "index_dir": str(index_dir),

            # Hiperparámetros S3
            "max_steps": max_steps,
            "top_k_per_step": top_k_per_step,
            "max_retrieval_steps": max_retrieval_steps,
            "max_total_chunks": max_total_chunks,
            "max_chars_per_chunk": max_chars_per_chunk,

            # Respuesta cruda y parseable
            "raw_output": safe_json_dumps(
                {
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", None),
                    "abstained": result.get("abstained", None),
                    "retrieval_mode": result.get("retrieval_mode", "active"),
                    "num_generation_steps": result.get("num_generation_steps", None),
                    "num_retrieval_steps": result.get("num_retrieval_steps", None),
                    "num_chunks_retrieved_total": result.get("num_chunks_retrieved_total", None),
                    "evidence_ids": evidence_ids,
                    "flare_trace": trace,
                }
            ),
            "raw_response_json": safe_json_dumps(result),

            # Campos directos para inspección rápida
            "final_answer": clean_text(result.get("answer", "")),
            "final_confidence": result.get("confidence", None),
            "final_abstained": result.get("abstained", None),
            "active_retrieval_triggered": bool(result.get("num_retrieval_steps", 0) or retrieved_chunk_ids),

            # Métricas de pasos/retrieval
            "num_generation_steps": result.get("num_generation_steps", None),
            "num_retrieval_steps": result.get("num_retrieval_steps", None),
            "num_chunks_retrieved_total": result.get("num_chunks_retrieved_total", None),
            "unique_chunks_retrieved": len(retrieved_chunk_ids),
            "max_steps_reached": max_steps_reached(result, max_steps),

            # Evidencia / trazas
            "evidence_ids": join_pipe([str(x) for x in evidence_ids]),
            "evidence_ids_json": safe_json_dumps(evidence_ids),
            "retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),
            "retrieved_chunk_ids_json": safe_json_dumps(retrieved_chunk_ids),
            "retrieval_queries_json": safe_json_dumps(retrieval_queries),
            "flare_trace_json": safe_json_dumps(trace),

            # Costos
            "input_tokens": result.get("input_tokens", None),
            "output_tokens": result.get("output_tokens", None),
            "total_tokens": result.get("total_tokens", None),
            "latency_seconds": result.get("latency_seconds", None),
            "row_wall_latency_seconds": round(float(row_latency_seconds), 3),

            # Control
            "dry_run": dry_run,
            "error": error,
        }
    )

    return output


def build_error_result(error: str) -> dict[str, Any]:
    return {
        "answer": "",
        "confidence": 0.0,
        "abstained": True,
        "retrieval_mode": "active",
        "num_generation_steps": 0,
        "num_retrieval_steps": 0,
        "num_chunks_retrieved_total": 0,
        "evidence_ids": [],
        "flare_trace": [],
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    input_path: Path,
    output_path: Path,
    index_dir: Path,
    model: str | None,
    limit: int | None,
    resume: bool,
    save_every: int,
    max_retries: int,
    max_steps: int,
    top_k_per_step: int,
    max_retrieval_steps: int,
    max_total_chunks: int,
    max_chars_per_chunk: int,
    retrieval_strategy: str,
    task_profile_mode: str,
    retrieval_bias_mode: str,
    dry_run: bool,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    if not dry_run and not index_dir.exists():
        raise FileNotFoundError(f"No existe index-dir: {index_dir}")

    if save_every < 0:
        raise ValueError("--save-every no puede ser negativo.")
    if max_steps <= 0:
        raise ValueError("--max-steps debe ser mayor que 0.")
    if top_k_per_step <= 0:
        raise ValueError("--top-k-per-step debe ser mayor que 0.")
    if max_retrieval_steps < 0:
        raise ValueError("--max-retrieval-steps no puede ser negativo.")
    if max_total_chunks < 0:
        raise ValueError("--max-total-chunks no puede ser negativo.")

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
    if not dry_run and not pending_df.empty:
        print(f"Cargando retriever desde índice: {index_dir}")
        retriever = create_retriever(index_dir)

    print("Runner S3 FLARE-like")
    print("--------------------")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Index dir: {index_dir}")
    print(f"Modelo: {model or '<default direct_llm.py>'}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print(f"Task profile mode: {task_profile_mode}")
    print(f"Retrieval bias mode: {retrieval_bias_mode}")
    print(f"Max steps: {max_steps}")
    print(f"Top-k per step: {top_k_per_step}")
    print(f"Max retrieval steps: {max_retrieval_steps}")
    print(f"Max total chunks: {max_total_chunks}")
    print(f"Dry run: {dry_run}")
    print(f"Filas consideradas: {len(df)}")
    print(f"Filas existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Running S3 FLARE-like"),
        start=1,
    ):
        row_start = time.time()
        question = ""
        outer_error = ""
        task_type = "open_qa"
        retrieval_bias = "balanced"
        question_format = "open"

        try:
            task_type, retrieval_bias, question_format = infer_task_profile(
                row,
                task_profile_mode=task_profile_mode,
                retrieval_bias_mode=retrieval_bias_mode,
            )
            question = get_question_for_s3(row, task_type=task_type)

            result = run_flare_controller_for_question(
                question=question,
                retriever=retriever,
                index_dir=index_dir,
                model=model,
                max_steps=max_steps,
                top_k_per_step=top_k_per_step,
                max_retrieval_steps=max_retrieval_steps,
                max_total_chunks=max_total_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
                retrieval_strategy=retrieval_strategy,
                task_type=task_type,
                retrieval_bias=retrieval_bias,
                max_retries=max_retries,
                dry_run=dry_run,
            )

        except Exception as exc:
            outer_error = str(exc)
            result = build_error_result(outer_error)

        row_latency = time.time() - row_start

        output_row = build_output_row(
            row,
            system_name="S3_flare_like_active_retrieval",
            model=model,
            question=question,
            result=result,
            retrieval_strategy=retrieval_strategy,
            task_type=task_type,
            retrieval_bias=retrieval_bias,
            question_format=question_format,
            index_dir=index_dir,
            max_steps=max_steps,
            top_k_per_step=top_k_per_step,
            max_retrieval_steps=max_retrieval_steps,
            max_total_chunks=max_total_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
            dry_run=dry_run,
            outer_error=outer_error,
            row_latency_seconds=row_latency,
        )
        rows.append(output_row)

        if save_every > 0 and i % save_every == 0:
            save_results(rows, output_path)

    save_results(rows, output_path)
    output_df = pd.DataFrame(rows)

    print(f"\nResultados S3 guardados en: {output_path}")
    print_summary(output_df)

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
    print("\nResumen S3 raw")
    print("--------------")
    print(f"Filas: {len(df)}")

    if "s2_case_type" in df.columns:
        print("\nTipos de caso:")
        print(df["s2_case_type"].value_counts(dropna=False).to_string())

    if "final_abstained" in df.columns:
        print(f"\nAbstention rate: {fmt(mean_bool(df['final_abstained']))}")

    if "active_retrieval_triggered" in df.columns:
        print(f"Active retrieval triggered rate: {fmt(mean_bool(df['active_retrieval_triggered']))}")

    if "num_generation_steps" in df.columns:
        print(f"Avg generation steps: {fmt(mean_numeric(df['num_generation_steps']))}")

    if "num_retrieval_steps" in df.columns:
        print(f"Avg retrieval steps: {fmt(mean_numeric(df['num_retrieval_steps']))}")

    if "num_chunks_retrieved_total" in df.columns:
        print(f"Avg chunks retrieved total: {fmt(mean_numeric(df['num_chunks_retrieved_total']))}")

    if "total_tokens" in df.columns:
        print(f"Avg total tokens: {fmt(mean_numeric(df['total_tokens']))}")

    if "latency_seconds" in df.columns:
        print(f"Avg LLM latency seconds: {fmt(mean_numeric(df['latency_seconds']))}")

    if "row_wall_latency_seconds" in df.columns:
        print(f"Avg row wall latency seconds: {fmt(mean_numeric(df['row_wall_latency_seconds']))}")

    if "max_steps_reached" in df.columns:
        print(f"Max steps reached rate: {fmt(mean_bool(df['max_steps_reached']))}")

    if "error" in df.columns:
        error_rate = df["error"].fillna("").astype(str).str.strip().ne("").mean()
        print(f"Run error rate: {error_rate:.3f}")

        errors = df[df["error"].fillna("").astype(str).str.strip().ne("")]
        if not errors.empty:
            print("\nPrimeros errores:")
            for _, row in errors.head(5).iterrows():
                print(f"- id={row.get('id', '')}: {row.get('error', '')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta S3 FLARE-like active retrieval sobre un CSV de preguntas."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV de preguntas. Por defecto usa data/s2/adaptive_rag/questions_s2.csv.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV raw de salida para S3.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio del índice vectorial. Por defecto usa indexes/s2/adaptive_rag.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo generador. Si se omite, usa el default de direct_llm.py.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de filas a procesar.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma desde un output existente usando la columna id.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Guarda resultados cada N filas. 0 desactiva guardado intermedio.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Reintentos para llamadas LLM.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Máximo de pasos de generación activa por pregunta.",
    )

    parser.add_argument(
        "--top-k-per-step",
        type=int,
        default=DEFAULT_TOP_K_PER_STEP,
        help="Chunks a recuperar por cada paso activo.",
    )

    parser.add_argument(
        "--max-retrieval-steps",
        type=int,
        default=DEFAULT_MAX_RETRIEVAL_STEPS,
        help="Máximo de pasos con retrieval por pregunta.",
    )

    parser.add_argument(
        "--max-total-chunks",
        type=int,
        default=DEFAULT_MAX_TOTAL_CHUNKS,
        help="Máximo total de chunks únicos por pregunta.",
    )

    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_CHUNK,
        help="Máximo de caracteres por chunk dentro del prompt de regeneración.",
    )

    parser.add_argument(
        "--retrieval-strategy",
        choices=["rules", "llm", "hybrid"],
        default=DEFAULT_RETRIEVAL_STRATEGY,
        help="Política para decidir retrieval activo.",
    )

    parser.add_argument(
        "--task-profile-mode",
        choices=sorted(VALID_TASK_PROFILE_MODES),
        default="auto",
        help="Perfil de tarea para prompts/política. auto infiere MMLU, TruthfulQA y HotpotQA.",
    )

    parser.add_argument(
        "--retrieval-bias-mode",
        choices=sorted(VALID_RETRIEVAL_BIAS_MODES),
        default="auto",
        help="Sesgo de retrieval. auto usa conservative para direct/MC y aggressive para HotpotQA.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No llama API ni carga índice real. Útil para validar flujo.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit debe ser mayor que 0 si se especifica.")

    run_experiment(
        input_path=args.input_path,
        output_path=args.output_path,
        index_dir=args.index_dir,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        max_retries=args.max_retries,
        max_steps=args.max_steps,
        top_k_per_step=args.top_k_per_step,
        max_retrieval_steps=args.max_retrieval_steps,
        max_total_chunks=args.max_total_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        retrieval_strategy=args.retrieval_strategy,
        task_profile_mode=args.task_profile_mode,
        retrieval_bias_mode=args.retrieval_bias_mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_s0_direct.py

Ejecuta el baseline S0 sobre questions_s0.csv.

Entrada esperada:
    data/questions_s0.csv

Salida por defecto:
    outputs/s0/results_s0_raw.csv

Este script guarda respuestas crudas. El parseo y la evaluación deben hacerse
en pasos posteriores para no repetir llamados al modelo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from direct_llm import ask_direct_llm_with_metadata


DEFAULT_INPUT_PATH = Path("data/questions_s0.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s0/results_s0_raw.csv")


REQUIRED_COLUMNS = {
    "id",
    "dataset",
    "case_type",
    "question",
}


COLUMNS_TO_KEEP_IF_PRESENT = [
    "id",
    "dataset",
    "case_type",
    "subject",
    "difficulty",
    "source",
    "source_split",
    "original_question",
    "question",
    "prompt",
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_idx",
    "gold_answer_text",
    "expected_answer",
    "expected_behavior",
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",
    "truthfulqa_category",
    "truthfulqa_type",
    "requires_retrieval",
    "original_source",
    "expected_model_output_format",
    "evaluation_notes",
]


def validate_input(df: pd.DataFrame) -> None:
    """Valida columnas mínimas para correr S0."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(missing))
        )


def get_prompt(row: pd.Series) -> str:
    """
    Devuelve el prompt final.

    Preferimos la columna `prompt` si existe y no está vacía.
    Si no, usamos `question`.

    Esto permite compatibilidad con:
    - datasets viejos, donde question era la pregunta cruda;
    - questions_s0.csv, donde question/prompt ya contienen el prompt final.
    """
    prompt = row.get("prompt", "")

    if isinstance(prompt, str) and prompt.strip():
        return prompt

    question = row.get("question", "")

    if isinstance(question, str) and question.strip():
        return question

    raise ValueError(f"Fila con id={row.get('id', '<sin id>')} sin prompt/question válido.")


def load_existing_results(output_path: Path) -> pd.DataFrame:
    """Carga resultados previos si existen."""
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)

    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo de salida existente {output_path} no tiene columna 'id'. "
            "No puedo usar --resume de forma segura."
        )

    return existing


def build_result_row(
    input_row: pd.Series,
    *,
    system_name: str,
    model: str | None,
    model_result: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    """Construye una fila de salida conservando metadatos útiles del dataset."""
    output: dict[str, Any] = {}

    for col in COLUMNS_TO_KEEP_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    output["system"] = system_name
    output["model"] = model or ""

    if model_result is None:
        output["raw_output"] = ""
        output["latency_seconds"] = None
        output["usage_json"] = ""
        output["input_tokens"] = None
        output["output_tokens"] = None
        output["total_tokens"] = None
    else:
        output["model"] = model_result.get("model", model or "")
        output["raw_output"] = model_result.get("raw_output", "")
        output["latency_seconds"] = model_result.get("latency_seconds")
        output["usage_json"] = model_result.get("usage_json", "")
        output["input_tokens"] = model_result.get("input_tokens")
        output["output_tokens"] = model_result.get("output_tokens")
        output["total_tokens"] = model_result.get("total_tokens")

    output["error"] = error

    return output


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Guarda resultados acumulados."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_experiment(
    *,
    input_path: Path,
    output_path: Path,
    model: str | None,
    limit: int | None,
    resume: bool,
    save_every: int,
    max_retries: int,
) -> pd.DataFrame:
    """
    Corre el baseline S0 y guarda respuestas crudas.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

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

    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida: {output_path}")
    print(f"Filas totales consideradas: {len(df)}")
    print(f"Filas ya existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(
            pending_df.iterrows(),
            total=len(pending_df),
            desc="Running S0 Direct LLM",
        ),
        start=1,
    ):
        try:
            prompt = get_prompt(row)

            model_result = ask_direct_llm_with_metadata(
                prompt,
                model=model,
                max_retries=max_retries,
            )
            error = ""

        except Exception as exc:
            model_result = None
            error = str(exc)

        result_row = build_result_row(
            row,
            system_name="S0_direct_llm",
            model=model,
            model_result=model_result,
            error=error,
        )

        rows.append(result_row)

        if save_every > 0 and i % save_every == 0:
            save_results(rows, output_path)

    save_results(rows, output_path)

    print(f"\nResultados guardados en: {output_path}")
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corre el baseline S0 directo sobre questions_s0.csv."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV normalizado de entrada.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV donde guardar respuestas crudas.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo a usar. Si se omite, usa OPENAI_MODEL o el default de direct_llm.py.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Corre solo las primeras N filas. Útil para testear.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="No repite IDs que ya existan en el archivo de salida.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Cada cuántas filas guardar resultados parciales.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Cantidad máxima de reintentos por pregunta.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_experiment(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()

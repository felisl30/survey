#!/usr/bin/env python3
"""
check_s1_raw.py

Verificación rápida de resultados crudos de S1 RAG.

No llama al LLM.
No evalúa calidad semántica final.
Solo revisa:
- cantidad de filas;
- errores de ejecución;
- respuestas vacías;
- parseo JSON básico;
- chunks recuperados;
- top-k usado;
- tokens y latencia;
- algunos ejemplos para inspección manual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_raw.csv")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_pipe(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Intenta extraer el primer JSON object de una respuesta.
    Soporta salidas tipo:
      {"answer": "..."}
      ```json
      {"answer": "..."}
      ```
      Texto antes/después del JSON.
    """
    text = clean_text(text)

    if not text:
        return None

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, dict):
        return obj

    return None


def summarize_bool(series: pd.Series) -> str:
    if len(series) == 0:
        return "n/a"
    return f"{series.mean():.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chequea resultados crudos de S1 RAG."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV crudo generado por run_s1_rag.py.",
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=20,
        help="Cantidad esperada de preguntas para HotpotQA-mini.",
    )

    parser.add_argument(
        "--expected-top-k",
        type=int,
        default=5,
        help="Top-k esperado para esta corrida.",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Cantidad de ejemplos a imprimir.",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {args.input_path}")

    df = pd.read_csv(args.input_path)

    print("\nCHECK S1 RAW")
    print("============")
    print(f"Archivo: {args.input_path}")
    print(f"Filas encontradas: {len(df)}")
    print(f"Filas esperadas: {args.expected_n}")

    required_cols = [
        "id",
        "original_question",
        "expected_answer",
        "retrieval_query",
        "retrieved_chunk_ids",
        "retrieved_titles",
        "raw_output",
        "error",
    ]

    print("\nColumnas requeridas")
    print("-------------------")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print("Faltan columnas:", ", ".join(missing))
    else:
        print("OK: están todas las columnas mínimas.")

    # Errores
    error_series = (
        df["error"].fillna("").astype(str).str.strip()
        if "error" in df.columns
        else pd.Series([""] * len(df))
    )
    has_error = error_series.ne("")

    raw_series = (
        df["raw_output"].fillna("").astype(str).str.strip()
        if "raw_output" in df.columns
        else pd.Series([""] * len(df))
    )
    empty_output = raw_series.eq("")

    print("\nEjecución")
    print("---------")
    print(f"Filas con error: {int(has_error.sum())}")
    print(f"Filas con raw_output vacío: {int(empty_output.sum())}")

    # Parseo JSON básico
    parsed_objects = [extract_json_object(x) for x in raw_series]
    json_ok = pd.Series([obj is not None for obj in parsed_objects])

    answers = []
    confidences = []
    abstained_values = []

    for obj in parsed_objects:
        if obj is None:
            answers.append("")
            confidences.append(None)
            abstained_values.append(None)
        else:
            answers.append(clean_text(obj.get("answer", "")))
            confidences.append(obj.get("confidence", None))
            abstained_values.append(obj.get("abstained", None))

    parsed_answer_empty = pd.Series([not bool(x) for x in answers])

    print("\nFormato de respuesta")
    print("--------------------")
    print(f"JSON parseable rate: {summarize_bool(json_ok)}")
    print(f"JSON parseables: {int(json_ok.sum())}/{len(df)}")
    print(f"parsed answer vacío: {int(parsed_answer_empty.sum())}/{len(df)}")

    # Retrieval
    print("\nRetrieval")
    print("---------")

    if "retrieved_chunk_ids" in df.columns:
        retrieved_counts = df["retrieved_chunk_ids"].apply(lambda x: len(split_pipe(x)))
        print(f"Chunks recuperados promedio: {retrieved_counts.mean():.2f}")
        print(f"Chunks recuperados mínimo: {retrieved_counts.min()}")
        print(f"Chunks recuperados máximo: {retrieved_counts.max()}")
        print(f"Filas con != expected_top_k: {int((retrieved_counts != args.expected_top_k).sum())}")
    else:
        retrieved_counts = pd.Series(dtype=int)
        print("No está la columna retrieved_chunk_ids.")

    if "top_k" in df.columns:
        print("\nValores de top_k encontrados:")
        print(df["top_k"].value_counts(dropna=False).to_string())
    else:
        print("\nNo está la columna top_k. No es grave si el top-k está implícito en la corrida.")

    # Latencia y tokens
    print("\nCosto / latencia")
    print("----------------")

    for col in ["latency_seconds", "input_tokens", "output_tokens", "total_tokens", "rag_prompt_chars"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            print(
                f"{col}: "
                f"mean={values.mean():.2f}, "
                f"min={values.min():.2f}, "
                f"max={values.max():.2f}"
            )

    # Problemas principales
    problem_mask = has_error | empty_output | (~json_ok)

    if "retrieved_chunk_ids" in df.columns and len(retrieved_counts) == len(df):
        problem_mask = problem_mask | (retrieved_counts != args.expected_top_k)

    problems = df[problem_mask].copy()

    print("\nFilas problemáticas")
    print("-------------------")
    print(f"Cantidad: {len(problems)}")

    if len(problems) > 0:
        cols = [
            col for col in [
                "id",
                "original_question",
                "expected_answer",
                "retrieved_chunk_ids",
                "raw_output",
                "error",
            ]
            if col in problems.columns
        ]

        print(problems[cols].head(10).to_string(index=False))

    # Ejemplos para inspección manual
    print("\nEjemplos")
    print("--------")

    example_cols = [
        col for col in [
            "id",
            "original_question",
            "expected_answer",
            "retrieved_titles",
            "raw_output",
            "error",
        ]
        if col in df.columns
    ]

    print(df[example_cols].head(args.show_examples).to_string(index=False))

    # Resumen compacto para pegar en ChatGPT
    summary = {
        "input_path": str(args.input_path),
        "n_rows": int(len(df)),
        "expected_n": int(args.expected_n),
        "n_errors": int(has_error.sum()),
        "n_empty_raw_output": int(empty_output.sum()),
        "json_parseable_rate": float(json_ok.mean()) if len(df) else None,
        "n_json_parseable": int(json_ok.sum()),
        "n_parsed_answer_empty": int(parsed_answer_empty.sum()),
        "n_problem_rows": int(len(problems)),
    }

    if "retrieved_chunk_ids" in df.columns and len(retrieved_counts):
        summary.update(
            {
                "retrieved_chunks_mean": float(retrieved_counts.mean()),
                "retrieved_chunks_min": int(retrieved_counts.min()),
                "retrieved_chunks_max": int(retrieved_counts.max()),
                "n_rows_with_wrong_top_k": int((retrieved_counts != args.expected_top_k).sum()),
            }
        )

    for col in ["latency_seconds", "input_tokens", "output_tokens", "total_tokens", "rag_prompt_chars"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_mean"] = None if values.dropna().empty else float(values.mean())
            summary[f"{col}_max"] = None if values.dropna().empty else float(values.max())

    print("\nRESUMEN_COMPACTO_PARA_PASAR")
    print("---------------------------")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
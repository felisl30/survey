#!/usr/bin/env python3
"""
inspect_mc_rag_readiness.py

Inspecciona un benchmark multiple-choice ya generado para saber si contiene
información suficiente para correr sistemas RAG sobre el mismo set.

No modifica el dataset. Solo genera un reporte de diagnóstico.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CONTEXT_HINTS = [
    "context",
    "contexts",
    "paragraph",
    "paragraphs",
    "passage",
    "passages",
    "document",
    "documents",
    "doc",
    "docs",
    "evidence",
    "evidences",
    "support",
    "supporting",
    "supporting_facts",
    "supporting_contexts",
    "title",
    "titles",
    "article",
    "articles",
    "source_text",
    "source_context",
    "retrieval",
    "corpus",
    "gold_context",
    "gold_passages",
    "gold_docs",
    "raw_example",
    "original_example",
]

CRITICAL_MC_COLUMNS = ["id", "dataset", "question", "A", "B", "C", "D", "gold_answer"]


def clean_str(value: Any, max_len: int = 500) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def try_parse_json(value: Any) -> Any | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (dict, list)):
        return value

    text = str(value).strip()
    if not text:
        return None

    if not (
        text.startswith("{")
        or text.startswith("[")
    ):
        return None

    try:
        return json.loads(text)
    except Exception:
        return None


def summarize_json_shape(obj: Any) -> str:
    if obj is None:
        return "not_json"

    if isinstance(obj, dict):
        keys = list(obj.keys())
        return f"dict keys={keys[:15]}"

    if isinstance(obj, list):
        if not obj:
            return "empty_list"

        first = obj[0]
        if isinstance(first, dict):
            return f"list len={len(obj)} first_keys={list(first.keys())[:15]}"
        return f"list len={len(obj)} first_type={type(first).__name__}"

    return type(obj).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/questions_musique_mc_100.csv"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/eval_mc/build_summary_musique_mc_100.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/rag_readiness"),
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="musique_mc_100",
    )
    args = parser.parse_args()

    if not args.questions_path.exists():
        raise FileNotFoundError(f"No existe: {args.questions_path}")

    df = pd.read_csv(args.questions_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = list(df.columns)
    lower_map = {col.lower(): col for col in columns}

    missing_mc_cols = [col for col in CRITICAL_MC_COLUMNS if col not in df.columns]

    candidate_context_cols = []
    for col in columns:
        col_lower = col.lower()
        if any(hint in col_lower for hint in CONTEXT_HINTS):
            candidate_context_cols.append(col)

    json_like_cols = []
    json_shapes = {}

    for col in columns:
        non_empty = df[col].dropna()
        if non_empty.empty:
            continue

        sample_values = non_empty.head(10).tolist()
        parsed_count = 0
        first_shape = None

        for value in sample_values:
            parsed = try_parse_json(value)
            if parsed is not None:
                parsed_count += 1
                if first_shape is None:
                    first_shape = summarize_json_shape(parsed)

        if parsed_count > 0:
            json_like_cols.append(col)
            json_shapes[col] = {
                "parsed_in_first_10_non_empty": parsed_count,
                "first_shape": first_shape,
            }

    useful_json_cols = []
    for col in json_like_cols:
        col_lower = col.lower()
        if any(hint in col_lower for hint in CONTEXT_HINTS):
            useful_json_cols.append(col)

    first_row_preview = {}
    if len(df) > 0:
        row = df.iloc[0]
        for col in columns:
            if col in ["prompt", "raw_output"]:
                continue
            if col in candidate_context_cols or col in CRITICAL_MC_COLUMNS or col in useful_json_cols:
                first_row_preview[col] = clean_str(row.get(col), max_len=700)

    summary_loaded = None
    if args.summary_path.exists():
        with args.summary_path.open("r", encoding="utf-8") as f:
            summary_loaded = json.load(f)

    rag_ready_guess = bool(candidate_context_cols or useful_json_cols)

    report = {
        "benchmark_name": args.benchmark_name,
        "questions_path": str(args.questions_path),
        "summary_path": str(args.summary_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(columns)),
        "columns": columns,
        "missing_mc_columns": missing_mc_cols,
        "candidate_context_columns": candidate_context_cols,
        "json_like_columns": json_like_cols,
        "json_shapes": json_shapes,
        "useful_json_context_columns": useful_json_cols,
        "rag_ready_guess": rag_ready_guess,
        "first_row_preview": first_row_preview,
        "summary_loaded": summary_loaded,
    }

    json_path = args.output_dir / f"{args.benchmark_name}_rag_readiness.json"
    txt_path = args.output_dir / f"{args.benchmark_name}_rag_readiness.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"RAG readiness report: {args.benchmark_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Questions path: {args.questions_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(columns)}\n\n")

        f.write("Missing MC columns:\n")
        if missing_mc_cols:
            for col in missing_mc_cols:
                f.write(f"- {col}\n")
        else:
            f.write("- None\n")

        f.write("\nCandidate context/evidence columns:\n")
        if candidate_context_cols:
            for col in candidate_context_cols:
                f.write(f"- {col}\n")
        else:
            f.write("- None\n")

        f.write("\nJSON-like columns:\n")
        if json_like_cols:
            for col in json_like_cols:
                f.write(f"- {col}: {json_shapes.get(col)}\n")
        else:
            f.write("- None\n")

        f.write("\nUseful JSON context columns:\n")
        if useful_json_cols:
            for col in useful_json_cols:
                f.write(f"- {col}\n")
        else:
            f.write("- None\n")

        f.write("\nRAG-ready guess:\n")
        f.write(f"- {rag_ready_guess}\n")

        f.write("\nFirst row preview:\n")
        f.write(json.dumps(first_row_preview, ensure_ascii=False, indent=2))
        f.write("\n")

    print("\nRAG readiness report")
    print("--------------------")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(columns)}")
    print(f"Missing MC columns: {missing_mc_cols if missing_mc_cols else 'None'}")
    print(f"Candidate context/evidence columns: {candidate_context_cols if candidate_context_cols else 'None'}")
    print(f"Useful JSON context columns: {useful_json_cols if useful_json_cols else 'None'}")
    print(f"RAG-ready guess: {rag_ready_guess}")
    print(f"JSON report: {json_path}")
    print(f"TXT report: {txt_path}")

    if not rag_ready_guess:
        print("\nDIAGNÓSTICO:")
        print("El CSV parece tener preguntas/opciones, pero no evidencia recuperable.")
        print("Probablemente haya que modificar build_mc_eval_dataset.py para exportar corpus/qrels.")
    else:
        print("\nDIAGNÓSTICO:")
        print("El CSV parece contener alguna forma de contexto/evidencia.")
        print("El próximo paso sería convertir esas columnas en corpus/qrels para S1.")


if __name__ == "__main__":
    main()

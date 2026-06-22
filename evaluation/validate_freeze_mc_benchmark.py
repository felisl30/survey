#!/usr/bin/env python3
"""
validate_freeze_mc_benchmark.py

Valida y "congela" un benchmark multiple-choice ya generado.

Uso recomendado:
    python evaluation/validate_freeze_mc_benchmark.py \
      --questions-path data/eval_mc/questions_musique_mc_100.csv \
      --summary-path data/eval_mc/build_summary_musique_mc_100.json \
      --expected-n 100 \
      --benchmark-name musique_mc_100
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


VALID_OPTIONS = {"A", "B", "C", "D"}

CRITICAL_BASE_COLUMNS = {
    "id",
    "dataset",
    "question",
}

OPTION_COLUMN_SETS = [
    {"A", "B", "C", "D"},
    {"option_a", "option_b", "option_c", "option_d"},
]

GOLD_COLUMNS = [
    "gold_answer",
    "answer",
    "correct_answer",
    "label",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_option_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)

    for candidate_set in OPTION_COLUMN_SETS:
        if candidate_set.issubset(cols):
            return list(candidate_set)

    # Fallback: busca columnas que parezcan opciones.
    normalized = {c.lower(): c for c in df.columns}
    if all(k in normalized for k in ["a", "b", "c", "d"]):
        return [normalized["a"], normalized["b"], normalized["c"], normalized["d"]]

    return []


def find_gold_column(df: pd.DataFrame) -> str | None:
    for col in GOLD_COLUMNS:
        if col in df.columns:
            return col
    return None


def clean_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def validate_benchmark(
    *,
    questions_path: Path,
    summary_path: Path | None,
    expected_n: int,
    benchmark_name: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not questions_path.exists():
        raise FileNotFoundError(f"No existe el archivo de preguntas: {questions_path}")

    df = pd.read_csv(questions_path)

    missing_base = sorted(CRITICAL_BASE_COLUMNS - set(df.columns))
    if missing_base:
        errors.append(f"Faltan columnas críticas: {missing_base}")

    option_cols = find_option_columns(df)
    if not option_cols:
        errors.append(
            "No se encontraron columnas de opciones A/B/C/D. "
            "Esperaba A,B,C,D u option_a,option_b,option_c,option_d."
        )

    gold_col = find_gold_column(df)
    if gold_col is None:
        errors.append(
            f"No se encontró columna de respuesta correcta. "
            f"Busqué alguna de estas: {GOLD_COLUMNS}"
        )

    n_rows = len(df)
    if n_rows != expected_n:
        warnings.append(
            f"La cantidad de filas es {n_rows}, pero se esperaba {expected_n}."
        )

    duplicate_ids = 0
    duplicate_questions = 0

    if "id" in df.columns:
        duplicate_ids = int(df["id"].astype(str).duplicated().sum())
        if duplicate_ids > 0:
            errors.append(f"Hay {duplicate_ids} IDs duplicados.")

    if "question" in df.columns:
        normalized_questions = (
            df["question"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        duplicate_questions = int(normalized_questions.duplicated().sum())
        if duplicate_questions > 0:
            warnings.append(f"Hay {duplicate_questions} preguntas duplicadas o casi iguales por texto exacto normalizado.")

    empty_questions = 0
    if "question" in df.columns:
        empty_questions = int(df["question"].map(clean_str).eq("").sum())
        if empty_questions > 0:
            errors.append(f"Hay {empty_questions} preguntas vacías.")

    empty_options_by_col: dict[str, int] = {}
    if option_cols:
        for col in option_cols:
            empty_count = int(df[col].map(clean_str).eq("").sum())
            empty_options_by_col[col] = empty_count
            if empty_count > 0:
                errors.append(f"La columna de opción {col!r} tiene {empty_count} valores vacíos.")

    invalid_gold_count = 0
    gold_distribution: dict[str, int] = {}

    if gold_col is not None:
        gold_series = df[gold_col].map(clean_str).str.upper()
        gold_distribution = gold_series.value_counts(dropna=False).to_dict()
        invalid_gold_count = int((~gold_series.isin(VALID_OPTIONS)).sum())

        if invalid_gold_count > 0:
            errors.append(
                f"La columna {gold_col!r} tiene {invalid_gold_count} respuestas inválidas. "
                "Se esperaba A/B/C/D."
            )

    dataset_distribution = {}
    if "dataset" in df.columns:
        dataset_distribution = df["dataset"].map(clean_str).value_counts(dropna=False).to_dict()

    source_distribution = {}
    if "source_dataset" in df.columns:
        source_distribution = df["source_dataset"].map(clean_str).value_counts(dropna=False).to_dict()
    elif "source" in df.columns:
        source_distribution = df["source"].map(clean_str).value_counts(dropna=False).to_dict()

    difficulty_distribution = {}
    if "difficulty" in df.columns:
        difficulty_distribution = df["difficulty"].map(clean_str).value_counts(dropna=False).to_dict()

    summary_json = read_json_if_exists(summary_path)

    if summary_path is not None and not summary_path.exists():
        warnings.append(f"No existe el summary JSON indicado: {summary_path}")

    report = {
        "benchmark_name": benchmark_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions_path": str(questions_path),
        "summary_path": str(summary_path) if summary_path else None,
        "questions_sha256": sha256_file(questions_path),
        "summary_sha256": sha256_file(summary_path) if summary_path and summary_path.exists() else None,
        "n_rows": n_rows,
        "expected_n": expected_n,
        "columns": list(df.columns),
        "option_columns": option_cols,
        "gold_column": gold_col,
        "duplicate_ids": duplicate_ids,
        "duplicate_questions_exact_normalized": duplicate_questions,
        "empty_questions": empty_questions,
        "empty_options_by_col": empty_options_by_col,
        "invalid_gold_count": invalid_gold_count,
        "dataset_distribution": dataset_distribution,
        "source_distribution": source_distribution,
        "difficulty_distribution": difficulty_distribution,
        "gold_distribution": gold_distribution,
        "build_summary_loaded": summary_json is not None,
        "build_summary": summary_json,
        "status": "PASS" if not errors else "FAIL",
        "warnings": warnings,
        "errors": errors,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = output_dir / f"{benchmark_name}_freeze_report.json"
    txt_report_path = output_dir / f"{benchmark_name}_freeze_report.txt"

    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with txt_report_path.open("w", encoding="utf-8") as f:
        f.write(f"Benchmark freeze report: {benchmark_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Status: {report['status']}\n")
        f.write(f"Questions path: {questions_path}\n")
        f.write(f"Questions SHA256: {report['questions_sha256']}\n")
        f.write(f"Rows: {n_rows} / expected {expected_n}\n")
        f.write(f"Option columns: {option_cols}\n")
        f.write(f"Gold column: {gold_col}\n\n")

        f.write("Dataset distribution:\n")
        f.write(json.dumps(dataset_distribution, ensure_ascii=False, indent=2))
        f.write("\n\nGold distribution:\n")
        f.write(json.dumps(gold_distribution, ensure_ascii=False, indent=2))
        f.write("\n\nWarnings:\n")
        if warnings:
            for w in warnings:
                f.write(f"- {w}\n")
        else:
            f.write("- None\n")

        f.write("\nErrors:\n")
        if errors:
            for e in errors:
                f.write(f"- {e}\n")
        else:
            f.write("- None\n")

    report["json_report_path"] = str(json_report_path)
    report["txt_report_path"] = str(txt_report_path)

    return report, warnings, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida y congela un benchmark MC generado."
    )

    parser.add_argument(
        "--questions-path",
        type=Path,
        required=True,
        help="CSV de preguntas MC a validar.",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="JSON de summary generado al construir el benchmark.",
    )

    parser.add_argument(
        "--expected-n",
        type=int,
        default=100,
        help="Cantidad esperada de preguntas.",
    )

    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="benchmark_mc",
        help="Nombre lógico del benchmark para reportes.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/freeze_reports"),
        help="Directorio donde guardar reportes.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    report, warnings, errors = validate_benchmark(
        questions_path=args.questions_path,
        summary_path=args.summary_path,
        expected_n=args.expected_n,
        benchmark_name=args.benchmark_name,
        output_dir=args.output_dir,
    )

    print("\nFreeze benchmark report")
    print("-----------------------")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Status: {report['status']}")
    print(f"Rows: {report['n_rows']} / expected {report['expected_n']}")
    print(f"Questions SHA256: {report['questions_sha256']}")
    print(f"JSON report: {report['json_report_path']}")
    print(f"TXT report: {report['txt_report_path']}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\nOK: benchmark validado y congelado.")


if __name__ == "__main__":
    main()

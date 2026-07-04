#!/usr/bin/env python3
"""
export_musique_robustness_qualitative_examples.py

Exporta ejemplos cualitativos del experimento de robustez MuSiQue.

Lee:
  outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_question_matrix.csv

Opcionalmente cruza con:
  outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_interesting_cases.csv

Genera:
  robustness_qualitative_examples.csv
  robustness_qualitative_examples.md

No llama a OpenAI.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEMS = ["s1", "s2", "s3_mc"]
CONDITIONS = ["clean", "noisy", "adversarial"]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return bool(int(x))
    text = str(x).strip().lower()
    return text in {"true", "1", "yes", "y", "correct", "ok"}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def correct_col(condition: str, system: str) -> str | None:
    candidates = [
        f"{condition}_{system}_correct",
        f"{condition}_{system}_mc_correct",
        f"{condition}_{system}_eval_correct",
        f"{system}_{condition}_correct",
    ]
    return candidates


def answer_col(condition: str, system: str) -> list[str]:
    return [
        f"{condition}_{system}_answer",
        f"{condition}_{system}_parsed_answer",
        f"{condition}_{system}_mc_pred",
        f"{system}_{condition}_answer",
    ]


def get_bool(row: pd.Series, cols: list[str]) -> bool:
    for col in cols:
        if col in row.index:
            return to_bool(row[col])
    return False


def get_value(row: pd.Series, cols: list[str]) -> str:
    for col in cols:
        if col in row.index:
            return clean_text(row[col])
    return ""


def first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def classify_patterns(row: pd.Series) -> list[str]:
    labels: list[str] = []

    # Regresiones por ruido/adversarial para cada sistema.
    for system in SYSTEMS:
        clean_ok = get_bool(row, correct_col("clean", system))
        noisy_ok = get_bool(row, correct_col("noisy", system))
        adv_ok = get_bool(row, correct_col("adversarial", system))

        if clean_ok and not noisy_ok:
            labels.append(f"{system}_regresses_noisy")
        if clean_ok and not adv_ok:
            labels.append(f"{system}_regresses_adversarial")
        if not clean_ok and noisy_ok:
            labels.append(f"{system}_recovers_noisy")
        if not clean_ok and adv_ok:
            labels.append(f"{system}_recovers_adversarial")

    # Casos donde un sistema es el único que acierta por condición.
    for condition in CONDITIONS:
        ok = {system: get_bool(row, correct_col(condition, system)) for system in SYSTEMS}
        if ok["s1"] and not ok["s2"] and not ok["s3_mc"]:
            labels.append(f"{condition}_only_s1_correct")
        if ok["s2"] and not ok["s1"] and not ok["s3_mc"]:
            labels.append(f"{condition}_only_s2_correct")
        if ok["s3_mc"] and not ok["s1"] and not ok["s2"]:
            labels.append(f"{condition}_only_s3_mc_correct")
        if ok["s1"] and ok["s2"] and ok["s3_mc"]:
            labels.append(f"{condition}_all_rag_correct")
        if not ok["s1"] and not ok["s2"] and not ok["s3_mc"]:
            labels.append(f"{condition}_all_rag_wrong")

    # Casos útiles para la historia S1 vs S2.
    for condition in CONDITIONS:
        s1_ok = get_bool(row, correct_col(condition, "s1"))
        s2_ok = get_bool(row, correct_col(condition, "s2"))
        if s1_ok and not s2_ok:
            labels.append(f"{condition}_s1_beats_s2")
        if s2_ok and not s1_ok:
            labels.append(f"{condition}_s2_beats_s1")

    return labels


def choose_examples(df: pd.DataFrame, max_per_pattern: int) -> pd.DataFrame:
    df = df.copy()
    df["patterns"] = df.apply(classify_patterns, axis=1)

    priority_patterns = [
        "noisy_s1_beats_s2",
        "adversarial_s1_beats_s2",
        "s2_regresses_noisy",
        "s2_regresses_adversarial",
        "s3_mc_regresses_noisy",
        "s3_mc_regresses_adversarial",
        "adversarial_only_s3_mc_correct",
        "adversarial_only_s1_correct",
        "noisy_all_rag_wrong",
        "adversarial_all_rag_wrong",
    ]

    selected_rows = []
    used_ids = set()

    id_col = first_existing(df, ["id", "question_id", "example_id"])
    if id_col is None:
        raise ValueError("No encontré columna id/question_id/example_id.")

    for pattern in priority_patterns:
        subset = df[df["patterns"].apply(lambda xs: pattern in xs)].copy()
        subset = subset.sort_values(id_col)

        count = 0
        for _, row in subset.iterrows():
            rid = clean_text(row[id_col])
            if rid in used_ids:
                continue
            row_dict = row.to_dict()
            row_dict["selected_pattern"] = pattern
            selected_rows.append(row_dict)
            used_ids.add(rid)
            count += 1
            if count >= max_per_pattern:
                break

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows)


def option_block(row: pd.Series) -> str:
    parts = []
    for opt in ["A", "B", "C", "D"]:
        val = clean_text(row.get(opt, ""))
        if val:
            parts.append(f"{opt}. {val}")
    return "\n".join(parts)


def write_markdown(selected: pd.DataFrame, out_md: Path) -> None:
    id_col = first_existing(selected, ["id", "question_id", "example_id"]) or "id"
    question_col = first_existing(
        selected,
        ["original_question", "question_clean", "question", "retrieval_query"],
    )
    gold_col = first_existing(selected, ["gold_answer", "mc_gold", "correct_answer"])

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# MuSiQue robustness qualitative examples\n\n")
        f.write("Selección automática de casos útiles para interpretar clean/noisy/adversarial.\n\n")

        for pattern, sub in selected.groupby("selected_pattern", dropna=False):
            f.write(f"## {pattern}\n\n")

            for _, row in sub.iterrows():
                qid = clean_text(row.get(id_col, ""))
                question = clean_text(row.get(question_col, "")) if question_col else ""
                gold = clean_text(row.get(gold_col, "")) if gold_col else ""

                f.write(f"### {qid}\n\n")

                if question:
                    f.write(f"**Question:** {question}\n\n")

                opts = option_block(row)
                if opts:
                    f.write("**Options:**\n\n")
                    f.write("```text\n")
                    f.write(opts + "\n")
                    f.write("```\n\n")

                if gold:
                    f.write(f"**Gold:** {gold}\n\n")

                f.write("| Condition | S1 | S2 | S3-MC |\n")
                f.write("|---|---:|---:|---:|\n")

                for condition in CONDITIONS:
                    cells = []
                    for system in SYSTEMS:
                        ok = get_bool(row, correct_col(condition, system))
                        ans = get_value(row, answer_col(condition, system))
                        mark = "✓" if ok else "✗"
                        cell = f"{ans} {mark}".strip()
                        cells.append(cell)
                    f.write(f"| {condition} | {cells[0]} | {cells[1]} | {cells[2]} |\n")

                f.write("\n")

                patterns = row.get("patterns", [])
                if isinstance(patterns, list):
                    f.write("**Detected patterns:** " + ", ".join(patterns[:8]) + "\n\n")

                f.write("---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini"),
    )
    parser.add_argument("--max-per-pattern", type=int, default=3)
    args = parser.parse_args()

    analysis_dir = args.base_dir / "analysis"
    matrix_path = analysis_dir / "robustness_deep_question_matrix.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    df = pd.read_csv(matrix_path)

    selected = choose_examples(df, args.max_per_pattern)

    if selected.empty:
        raise ValueError("No se seleccionaron ejemplos. Revisar columnas/patrones.")

    out_csv = analysis_dir / "robustness_qualitative_examples.csv"
    out_md = analysis_dir / "robustness_qualitative_examples.md"

    # Convertir lista de patrones a string para CSV.
    csv_out = selected.copy()
    if "patterns" in csv_out.columns:
        csv_out["patterns"] = csv_out["patterns"].apply(
            lambda xs: ";".join(xs) if isinstance(xs, list) else str(xs)
        )

    csv_out.to_csv(out_csv, index=False)
    write_markdown(selected, out_md)

    print("Qualitative examples exported")
    print("=============================")
    print(f"Input: {matrix_path}")
    print(f"Rows selected: {len(selected)}")
    print(f"CSV: {out_csv}")
    print(f"MD: {out_md}")
    print()
    print("Selected patterns:")
    print(selected["selected_pattern"].value_counts().to_string())


if __name__ == "__main__":
    main()

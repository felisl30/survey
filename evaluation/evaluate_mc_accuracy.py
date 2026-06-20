#!/usr/bin/env python3
"""Calcula accuracy exacta A/B/C/D para outputs parseados MC."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

VALID = {"A", "B", "C", "D"}


def extract_letter(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    if text in VALID:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else ""


def evaluate(input_path: Path, output_path: Path, summary_path: Path) -> dict:
    df = pd.read_csv(input_path)
    if "gold_answer" not in df.columns:
        raise ValueError("El input debe tener columna gold_answer.")

    answer_col = "parsed_answer"
    if answer_col not in df.columns:
        raise ValueError("El input debe tener columna parsed_answer.")

    out = df.copy()
    out["mc_pred"] = out[answer_col].map(extract_letter)
    out["mc_gold"] = out["gold_answer"].map(extract_letter)
    out["mc_valid_prediction"] = out["mc_pred"].isin(VALID)
    out["mc_correct"] = out["mc_valid_prediction"] & (out["mc_pred"] == out["mc_gold"])

    rows = []
    rows.append({
        "group_type": "overall",
        "group": "all",
        "n": int(len(out)),
        "accuracy": float(out["mc_correct"].mean()) if len(out) else None,
        "valid_prediction_rate": float(out["mc_valid_prediction"].mean()) if len(out) else None,
    })
    if "dataset" in out.columns:
        for dataset, subset in out.groupby("dataset", dropna=False):
            rows.append({
                "group_type": "dataset",
                "group": str(dataset),
                "n": int(len(subset)),
                "accuracy": float(subset["mc_correct"].mean()) if len(subset) else None,
                "valid_prediction_rate": float(subset["mc_valid_prediction"].mean()) if len(subset) else None,
            })

    summary_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate(args.input_path, args.output_path, args.summary_path)
    print(f"MC accuracy: {summary['accuracy']:.3f}")
    print(f"Valid prediction rate: {summary['valid_prediction_rate']:.3f}")
    print(f"Results: {args.output_path}")
    print(f"Summary: {args.summary_path}")


if __name__ == "__main__":
    main()

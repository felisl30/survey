#!/usr/bin/env python3
"""
summarize_musique_robustness_s0_s3.py

Aggregates robustness outputs produced by scripts/run_musique_robustness_s0_s3.sh.

Usage from repo root:
  python evaluation/summarize_musique_robustness_s0_s3.py \
    --base-dir outputs/eval_mc/robustness_musique/gpt_5_4_mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEMS = ["s0", "s1", "s2", "s3_mc"]
CONDITIONS = ["clean", "noisy", "adversarial"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_overall(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall", {})
    if not isinstance(overall, dict):
        return {}
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    base = args.base_dir
    if args.output_dir is None:
        out_dir = base / "analysis"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # S0 is condition-independent.
    s0_summary_path = base / "s0_summary.json"
    if s0_summary_path.exists():
        overall = get_overall(read_json(s0_summary_path))
        rows.append({
            "condition": "baseline",
            "system": "s0",
            "summary_path": str(s0_summary_path),
            "n": overall.get("n"),
            "accuracy": overall.get("accuracy"),
            "valid_format_rate": overall.get("valid_format_rate"),
            "run_error_rate": overall.get("run_error_rate"),
            "avg_confidence": overall.get("avg_confidence"),
            "avg_latency_seconds": overall.get("avg_latency_seconds"),
            "avg_input_tokens": overall.get("avg_input_tokens"),
            "avg_output_tokens": overall.get("avg_output_tokens"),
            "avg_total_tokens": overall.get("avg_total_tokens"),
        })

    for condition in CONDITIONS:
        for system in ["s1", "s2", "s3_mc"]:
            summary_path = base / condition / f"{system}_summary.json"
            if not summary_path.exists():
                rows.append({
                    "condition": condition,
                    "system": system,
                    "summary_path": str(summary_path),
                    "missing": True,
                })
                continue

            overall = get_overall(read_json(summary_path))
            rows.append({
                "condition": condition,
                "system": system,
                "summary_path": str(summary_path),
                "missing": False,
                "n": overall.get("n"),
                "accuracy": overall.get("accuracy"),
                "valid_format_rate": overall.get("valid_format_rate"),
                "run_error_rate": overall.get("run_error_rate"),
                "avg_confidence": overall.get("avg_confidence"),
                "avg_latency_seconds": overall.get("avg_latency_seconds"),
                "avg_input_tokens": overall.get("avg_input_tokens"),
                "avg_output_tokens": overall.get("avg_output_tokens"),
                "avg_total_tokens": overall.get("avg_total_tokens"),
            })

    df = pd.DataFrame(rows)

    # Add deltas vs clean for S1/S2/S3.
    df["delta_accuracy_vs_clean"] = None
    df["delta_tokens_vs_clean"] = None

    for system in ["s1", "s2", "s3_mc"]:
        clean_row = df[(df["condition"] == "clean") & (df["system"] == system)]
        if clean_row.empty:
            continue
        clean_acc = pd.to_numeric(clean_row.iloc[0].get("accuracy"), errors="coerce")
        clean_tok = pd.to_numeric(clean_row.iloc[0].get("avg_total_tokens"), errors="coerce")

        mask = df["system"].eq(system) & df["condition"].isin(CONDITIONS)
        df.loc[mask, "delta_accuracy_vs_clean"] = pd.to_numeric(df.loc[mask, "accuracy"], errors="coerce") - clean_acc
        df.loc[mask, "delta_tokens_vs_clean"] = pd.to_numeric(df.loc[mask, "avg_total_tokens"], errors="coerce") - clean_tok

    summary_csv = out_dir / "robustness_s0_s3_summary.csv"
    df.to_csv(summary_csv, index=False)

    report_txt = out_dir / "robustness_s0_s3_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("MuSiQue robustness S0-S3 summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Base dir: {base}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        if not df.empty and "accuracy" in df.columns:
            pivot = df[df["condition"].isin(CONDITIONS)].pivot(
                index="condition",
                columns="system",
                values="accuracy",
            )
            f.write("Accuracy pivot:\n")
            f.write(pivot.to_string())
            f.write("\n\n")

    print("Guardado:")
    print(summary_csv)
    print(report_txt)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

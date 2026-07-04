#!/usr/bin/env python3
"""
summarize_s4_robustness_focus.py

Resume la corrida S4 focalizada para robustez.

No llama a OpenAI.
Lee:
- input focus con expected_s4_audit / expected_s4_suspicious
- uno o más raw CSV de run_s4_fire_like.py

Genera:
- CSV combinado con categorías de auditoría
- TXT legible
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


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
    return text


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    text = clean_text(x).lower()
    return text in {"true", "1", "yes", "y", "si", "sí"}


def to_int(x: Any) -> int:
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return 0


def infer_s4_suspicious(row: pd.Series) -> bool:
    decision = clean_text(row.get("s4_final_decision", "")).lower()
    n_refuted = to_int(row.get("s4_num_refuted_claims", 0))
    n_nei = to_int(row.get("s4_num_nei_claims", 0))

    if decision in {"abstained", "corrected", "no_claims"}:
        return True
    if n_refuted > 0:
        return True
    if n_nei > 0:
        return True
    return False


def classify(row: pd.Series) -> str:
    expected_suspicious = to_bool(row.get("expected_s4_suspicious", False))
    actual_suspicious = to_bool(row.get("s4_suspicious", False))
    s3_correct = to_bool(row.get("s3_mc_correct", False))

    if expected_suspicious and actual_suspicious and not s3_correct:
        return "expected_error_detected"

    if expected_suspicious and not actual_suspicious and not s3_correct:
        return "expected_error_missed"

    if not expected_suspicious and actual_suspicious and s3_correct:
        return "false_rejection_of_correct_s3"

    if not expected_suspicious and not actual_suspicious and s3_correct:
        return "correct_s3_preserved"

    if actual_suspicious:
        return "suspicious_other"

    return "not_suspicious_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-focus-path", type=Path, required=True)
    parser.add_argument("--raw-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="core5_rules")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    focus = pd.read_csv(args.input_focus_path)
    raw_parts = []

    for path in args.raw_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["_raw_path"] = str(path)
        raw_parts.append(df)

    raw = pd.concat(raw_parts, ignore_index=True)

    keep_focus_cols = [
        "id",
        "original_id",
        "question_id",
        "source_condition",
        "source_system",
        "source_system_for_s4",
        "s4_focus_case_type",
        "expected_s4_audit",
        "expected_s4_suspicious",
        "focus_reason",
        "gold_answer",
        "gold_answer_text",
        "s4_mc_gold_choice",
        "s4_mc_final_choice",
        "s4_mc_final_option_text",
        "s3_mc_correct",
    ]
    keep_focus_cols = [c for c in keep_focus_cols if c in focus.columns]

    merged = raw.merge(
        focus[keep_focus_cols],
        on="id",
        how="left",
        suffixes=("", "_focus"),
    )

    merged["s4_suspicious"] = merged.apply(infer_s4_suspicious, axis=1)
    merged["s4_audit_category"] = merged.apply(classify, axis=1)
    merged["expected_match"] = (
        merged["s4_suspicious"].map(bool)
        == merged["expected_s4_suspicious"].map(to_bool)
    )

    selected_cols = [
        "id",
        "source_condition",
        "s4_focus_case_type",
        "expected_s4_audit",
        "expected_s4_suspicious",
        "s3_mc_correct",
        "gold_answer",
        "s4_mc_final_choice",
        "s4_final_decision",
        "s4_suspicious",
        "s4_audit_category",
        "expected_match",
        "s4_num_claims",
        "s4_num_supported_claims",
        "s4_num_refuted_claims",
        "s4_num_nei_claims",
        "s4_num_chunks_retrieved_total",
        "s4_initial_evidence_count",
        "s4_claim_verdicts",
        "s4_answer",
    ]
    selected_cols = [c for c in selected_cols if c in merged.columns]

    out_csv = args.output_dir / f"{args.prefix}_combined.csv"
    out_txt = args.output_dir / f"{args.prefix}_report.txt"

    merged.to_csv(out_csv, index=False)

    summary = {
        "n": int(len(merged)),
        "expected_match_rate": float(merged["expected_match"].mean()) if len(merged) else None,
        "s4_suspicious_rate": float(merged["s4_suspicious"].mean()) if len(merged) else None,
        "abstention_rate": float((merged["s4_final_decision"].astype(str).str.lower() == "abstained").mean())
        if "s4_final_decision" in merged.columns and len(merged)
        else None,
        "avg_claims": float(pd.to_numeric(merged.get("s4_num_claims", 0), errors="coerce").mean())
        if len(merged)
        else None,
        "avg_supported_claims": float(pd.to_numeric(merged.get("s4_num_supported_claims", 0), errors="coerce").mean())
        if len(merged)
        else None,
        "avg_refuted_claims": float(pd.to_numeric(merged.get("s4_num_refuted_claims", 0), errors="coerce").mean())
        if len(merged)
        else None,
        "avg_nei_claims": float(pd.to_numeric(merged.get("s4_num_nei_claims", 0), errors="coerce").mean())
        if len(merged)
        else None,
        "avg_chunks_retrieved": float(pd.to_numeric(merged.get("s4_num_chunks_retrieved_total", 0), errors="coerce").mean())
        if len(merged)
        else None,
    }

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("S4 robustness focus summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input focus: {args.input_focus_path}\n")
        f.write("Raw paths:\n")
        for path in args.raw_paths:
            f.write(f"- {path}\n")
        f.write(f"\nCombined CSV: {out_csv}\n")
        f.write(f"Report TXT:   {out_txt}\n\n")

        f.write("Overall summary\n")
        f.write("-" * 80 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\nAudit categories\n")
        f.write("-" * 80 + "\n")
        f.write(merged["s4_audit_category"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        if "s4_focus_case_type" in merged.columns:
            f.write("By focus case type\n")
            f.write("-" * 80 + "\n")
            by_case = (
                merged.groupby("s4_focus_case_type", dropna=False)
                .agg(
                    n=("id", "count"),
                    expected_match_rate=("expected_match", "mean"),
                    s4_suspicious_rate=("s4_suspicious", "mean"),
                    abstention_rate=("s4_final_decision", lambda s: (s.astype(str).str.lower() == "abstained").mean()),
                    avg_supported_claims=("s4_num_supported_claims", "mean"),
                    avg_refuted_claims=("s4_num_refuted_claims", "mean"),
                    avg_nei_claims=("s4_num_nei_claims", "mean"),
                    avg_chunks=("s4_num_chunks_retrieved_total", "mean"),
                )
                .reset_index()
            )
            f.write(by_case.to_string(index=False))
            f.write("\n\n")

        f.write("Rows\n")
        f.write("-" * 80 + "\n")
        f.write(merged[selected_cols].to_string(index=False))
        f.write("\n")

    print("S4 robustness focus summary generado")
    print("------------------------------------")
    print(f"Rows: {len(merged)}")
    print(f"Expected match rate: {summary['expected_match_rate']}")
    print(f"CSV: {out_csv}")
    print(f"TXT: {out_txt}")


if __name__ == "__main__":
    main()

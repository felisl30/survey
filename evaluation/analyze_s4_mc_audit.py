#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PARSED_PATH = Path("outputs/eval_mc/musique_mc_rag/s4_mc/generation/fire_like_s4_mc_parsed_balanced_test11_v2_enriched.csv")
DEFAULT_CLAIMS_PATH = Path("outputs/eval_mc/musique_mc_rag/s4_mc/evaluation/fire_like_s4_mc_claim_results_balanced_test11_v2_enriched.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/eval_mc/musique_mc_rag/s4_mc/audit")


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def norm_choice(x) -> str:
    text = clean_text(x).upper()
    return text if text in {"A", "B", "C", "D"} else ""


def infer_s3_correct(row: pd.Series) -> bool:
    final_choice = norm_choice(row.get("s4_mc_final_choice", ""))
    gold_choice = norm_choice(row.get("s4_mc_gold_choice", ""))
    return bool(final_choice and gold_choice and final_choice == gold_choice)


def infer_s4_suspicious(row: pd.Series) -> bool:
    decision = clean_text(row.get("s4_final_decision", "")).lower()
    n_refuted = int(float(row.get("s4_num_refuted_claims", 0) or 0))
    n_nei = int(float(row.get("s4_num_nei_claims", 0) or 0))

    if decision in {"abstained", "no_claims"}:
        return True
    if n_refuted > 0:
        return True
    if n_nei > 0:
        return True
    return False


def classify_audit(row: pd.Series) -> str:
    case_type = clean_text(row.get("s4_mc_case_type", ""))
    s3_correct = infer_s3_correct(row)
    s4_suspicious = infer_s4_suspicious(row)
    decision = clean_text(row.get("s4_final_decision", "")).lower()

    if not s3_correct and s4_suspicious:
        return "true_error_detected"

    if not s3_correct and not s4_suspicious:
        return "false_pass_error"

    if s3_correct and s4_suspicious:
        if case_type == "retrieval_corrected_candidate":
            return "false_rejection_corrected_case"
        return "false_rejection_correct_case"

    if s3_correct and not s4_suspicious:
        if decision == "unchanged":
            return "correct_preserved"
        return "correct_not_suspicious"

    return "unknown"


def summarize_bool(df: pd.DataFrame, mask_col: str, value_col: str) -> float | None:
    subset = df[df[mask_col]]
    if subset.empty:
        return None
    return float(subset[value_col].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-path", type=Path, default=DEFAULT_PARSED_PATH)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="balanced_test11_v2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed = pd.read_csv(args.parsed_path)
    claims = pd.read_csv(args.claims_path)

    parsed["s3_mc_correct"] = parsed.apply(infer_s3_correct, axis=1)
    parsed["s4_suspicious"] = parsed.apply(infer_s4_suspicious, axis=1)
    parsed["s4_mc_audit_category"] = parsed.apply(classify_audit, axis=1)

    parsed["is_s3_error"] = ~parsed["s3_mc_correct"]
    parsed["is_s3_correct"] = parsed["s3_mc_correct"]

    audit_counts = parsed["s4_mc_audit_category"].value_counts().to_dict()

    by_case = (
        parsed.groupby("s4_mc_case_type", dropna=False)
        .agg(
            n=("id", "count"),
            s3_accuracy=("s3_mc_correct", "mean"),
            s4_suspicious_rate=("s4_suspicious", "mean"),
            abstention_rate=("s4_final_decision", lambda s: (s.astype(str).str.lower() == "abstained").mean()),
            unchanged_rate=("s4_final_decision", lambda s: (s.astype(str).str.lower() == "unchanged").mean()),
            avg_claims=("s4_num_claims", "mean"),
            avg_supported_claims=("s4_num_supported_claims", "mean"),
            avg_refuted_claims=("s4_num_refuted_claims", "mean"),
            avg_nei_claims=("s4_num_nei_claims", "mean"),
            avg_chunks=("s4_num_chunks_retrieved_total", "mean"),
        )
        .reset_index()
    )

    n_s3_errors = int(parsed["is_s3_error"].sum())
    n_s3_correct = int(parsed["is_s3_correct"].sum())

    error_detection_rate = (
        float(parsed.loc[parsed["is_s3_error"], "s4_suspicious"].mean())
        if n_s3_errors else None
    )

    correct_preservation_rate = (
        float((~parsed.loc[parsed["is_s3_correct"], "s4_suspicious"]).mean())
        if n_s3_correct else None
    )

    false_rejection_rate = (
        float(parsed.loc[parsed["is_s3_correct"], "s4_suspicious"].mean())
        if n_s3_correct else None
    )

    false_pass_rate = (
        float((~parsed.loc[parsed["is_s3_error"], "s4_suspicious"]).mean())
        if n_s3_errors else None
    )

    claim_summary = {}
    if not claims.empty and "claim_verdict" in claims.columns:
        claim_summary = {
            "n_claim_rows": int(len(claims)),
            "claim_verdict_counts": claims["claim_verdict"].value_counts(dropna=False).to_dict(),
            "claim_type_counts": claims["claim_type"].value_counts(dropna=False).to_dict()
            if "claim_type" in claims.columns else {},
            "claim_manual_review_rate": float(claims["claim_needs_manual_review"].astype(bool).mean())
            if "claim_needs_manual_review" in claims.columns else None,
        }

    report = {
        "n_questions": int(len(parsed)),
        "n_s3_errors": n_s3_errors,
        "n_s3_correct": n_s3_correct,
        "error_detection_rate": error_detection_rate,
        "correct_preservation_rate": correct_preservation_rate,
        "false_rejection_rate": false_rejection_rate,
        "false_pass_rate": false_pass_rate,
        "audit_counts": audit_counts,
        "claim_summary": claim_summary,
    }

    parsed_out = args.output_dir / f"{args.prefix}_question_audit.csv"
    by_case_out = args.output_dir / f"{args.prefix}_by_case_summary.csv"
    report_json_out = args.output_dir / f"{args.prefix}_audit_summary.json"
    report_txt_out = args.output_dir / f"{args.prefix}_audit_summary.txt"

    parsed.to_csv(parsed_out, index=False)
    by_case.to_csv(by_case_out, index=False)

    report_json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with report_txt_out.open("w", encoding="utf-8") as f:
        f.write("S4-MC audit summary\n")
        f.write("=" * 80 + "\n\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print("S4-MC audit summary")
    print("-------------------")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\nBy case:")
    print(by_case.to_string(index=False))

    print("\nAudit rows:")
    cols = [
        "id",
        "s4_mc_case_type",
        "s4_mc_final_choice",
        "s4_mc_gold_choice",
        "s3_mc_correct",
        "s4_final_decision",
        "s4_num_supported_claims",
        "s4_num_refuted_claims",
        "s4_num_nei_claims",
        "s4_suspicious",
        "s4_mc_audit_category",
    ]
    cols = [c for c in cols if c in parsed.columns]
    print(parsed[cols].to_string(index=False))

    print("\nArchivos generados:")
    print(parsed_out)
    print(by_case_out)
    print(report_json_out)
    print(report_txt_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_s2_mc_real_errors.py

Analiza errores y mejoras de S2 real contra S0 y S1 en MuSiQue-100 MC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    return None


def mean_bool(s: pd.Series) -> float | None:
    vals = s.map(coerce_bool).dropna()
    return float(vals.mean()) if len(vals) else None


def mean_num(s: pd.Series) -> float | None:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s0-results-path",
        type=Path,
        default=Path("outputs/eval_mc/s0_gpt_5_mini_musique_100_mc_results.csv"),
    )
    parser.add_argument(
        "--s1-results-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s1/s1_gpt_5_mini_top5_evaluated.csv"),
    )
    parser.add_argument(
        "--s2-results-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_real/s2_gpt_5_mini_policy_top1_045_gap_005_evaluated.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_real/error_analysis"),
    )

    args = parser.parse_args()

    for path in [args.s0_results_path, args.s1_results_path, args.s2_results_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    s0 = pd.read_csv(args.s0_results_path)
    s1 = pd.read_csv(args.s1_results_path)
    s2 = pd.read_csv(args.s2_results_path)

    s0_keep = s0[[
        c for c in [
            "id",
            "original_question",
            "question",
            "A", "B", "C", "D",
            "mc_gold",
            "mc_pred",
            "mc_correct",
            "total_tokens",
            "latency_seconds",
        ]
        if c in s0.columns
    ]].copy()

    s1_keep = s1[[
        c for c in [
            "id",
            "parsed_answer",
            "eval_correct",
            "total_tokens",
            "latency_seconds",
            "n_docs_retrieved",
            "retrieved_doc_ids_json",
            "retrieved_titles_json",
            "retrieved_scores_json",
        ]
        if c in s1.columns
    ]].copy()

    s2_keep = s2[[
        c for c in [
            "id",
            "gold_answer",
            "predicted_route",
            "parsed_answer",
            "eval_correct",
            "total_tokens",
            "latency_seconds",
            "top1_score",
            "top1_top2_gap",
            "n_docs_retrieved",
            "retrieved_doc_ids_json",
            "retrieved_titles_json",
            "retrieved_scores_json",
            "router_reason",
        ]
        if c in s2.columns
    ]].copy()

    s0_keep = s0_keep.rename(columns={
        "mc_gold": "gold_answer_s0",
        "mc_pred": "s0_pred",
        "mc_correct": "s0_correct",
        "total_tokens": "s0_total_tokens",
        "latency_seconds": "s0_latency_seconds",
    })

    s1_keep = s1_keep.rename(columns={
        "parsed_answer": "s1_pred",
        "eval_correct": "s1_correct",
        "total_tokens": "s1_total_tokens",
        "latency_seconds": "s1_latency_seconds",
        "n_docs_retrieved": "s1_n_docs_retrieved",
        "retrieved_doc_ids_json": "s1_retrieved_doc_ids_json",
        "retrieved_titles_json": "s1_retrieved_titles_json",
        "retrieved_scores_json": "s1_retrieved_scores_json",
    })

    s2_keep = s2_keep.rename(columns={
        "gold_answer": "gold_answer_s2",
        "parsed_answer": "s2_pred",
        "eval_correct": "s2_correct",
        "total_tokens": "s2_total_tokens",
        "latency_seconds": "s2_latency_seconds",
        "n_docs_retrieved": "s2_n_docs_retrieved",
        "retrieved_doc_ids_json": "s2_retrieved_doc_ids_json",
        "retrieved_titles_json": "s2_retrieved_titles_json",
        "retrieved_scores_json": "s2_retrieved_scores_json",
    })

    merged = s0_keep.merge(s1_keep, on="id", how="inner").merge(s2_keep, on="id", how="inner")

    merged["s0_correct_bool"] = merged["s0_correct"].map(coerce_bool)
    merged["s1_correct_bool"] = merged["s1_correct"].map(coerce_bool)
    merged["s2_correct_bool"] = merged["s2_correct"].map(coerce_bool)

    def category(row: pd.Series) -> str:
        s0_ok = bool(row["s0_correct_bool"])
        s1_ok = bool(row["s1_correct_bool"])
        s2_ok = bool(row["s2_correct_bool"])

        if s0_ok and s1_ok and s2_ok:
            return "all_correct"
        if not s0_ok and not s1_ok and not s2_ok:
            return "all_wrong"
        if s2_ok and not s0_ok and not s1_ok:
            return "only_s2_correct"
        if not s2_ok and s0_ok and s1_ok:
            return "only_s2_wrong"
        if s2_ok and not s0_ok:
            return "s2_corrected_s0"
        if not s2_ok and s0_ok:
            return "s2_regressed_vs_s0"
        if s2_ok and not s1_ok:
            return "s2_corrected_s1"
        if not s2_ok and s1_ok:
            return "s2_regressed_vs_s1"
        return "mixed"

    merged["error_category"] = merged.apply(category, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_path = args.output_dir / "s0_s1_s2_real_full_error_analysis.csv"
    merged.to_csv(full_path, index=False)

    exports = {
        "s2_corrected_s0_errors.csv": merged[
            (merged["s0_correct_bool"] == False) & (merged["s2_correct_bool"] == True)
        ],
        "s2_regressed_s0_correct.csv": merged[
            (merged["s0_correct_bool"] == True) & (merged["s2_correct_bool"] == False)
        ],
        "s2_corrected_s1_errors.csv": merged[
            (merged["s1_correct_bool"] == False) & (merged["s2_correct_bool"] == True)
        ],
        "s2_regressed_s1_correct.csv": merged[
            (merged["s1_correct_bool"] == True) & (merged["s2_correct_bool"] == False)
        ],
        "only_s2_correct.csv": merged[
            (merged["s0_correct_bool"] == False)
            & (merged["s1_correct_bool"] == False)
            & (merged["s2_correct_bool"] == True)
        ],
        "only_s2_wrong.csv": merged[
            (merged["s0_correct_bool"] == True)
            & (merged["s1_correct_bool"] == True)
            & (merged["s2_correct_bool"] == False)
        ],
        "all_wrong.csv": merged[
            (merged["s0_correct_bool"] == False)
            & (merged["s1_correct_bool"] == False)
            & (merged["s2_correct_bool"] == False)
        ],
    }

    for filename, subset in exports.items():
        subset.to_csv(args.output_dir / filename, index=False)

    by_route = []
    for route, sub in merged.groupby("predicted_route", dropna=False):
        by_route.append({
            "route": str(route),
            "n": int(len(sub)),
            "accuracy": mean_bool(sub["s2_correct"]),
            "avg_tokens": mean_num(sub["s2_total_tokens"]),
            "avg_latency_seconds": mean_num(sub["s2_latency_seconds"]),
            "avg_top1_score": mean_num(sub["top1_score"]),
            "avg_top1_top2_gap": mean_num(sub["top1_top2_gap"]),
        })

    by_route_df = pd.DataFrame(by_route)
    by_route_df.to_csv(args.output_dir / "s2_real_by_route_summary.csv", index=False)

    summary = {
        "n": int(len(merged)),
        "s0_accuracy": mean_bool(merged["s0_correct"]),
        "s1_accuracy": mean_bool(merged["s1_correct"]),
        "s2_accuracy": mean_bool(merged["s2_correct"]),
        "s2_retrieve_rate": float((merged["predicted_route"].astype(str) == "retrieve").mean()),
        "s2_direct_rate": float((merged["predicted_route"].astype(str) == "direct").mean()),
        "counts_by_error_category": merged["error_category"].value_counts().to_dict(),
        "s2_corrected_s0_errors": int(len(exports["s2_corrected_s0_errors.csv"])),
        "s2_regressed_s0_correct": int(len(exports["s2_regressed_s0_correct.csv"])),
        "s2_corrected_s1_errors": int(len(exports["s2_corrected_s1_errors.csv"])),
        "s2_regressed_s1_correct": int(len(exports["s2_regressed_s1_correct.csv"])),
        "only_s2_correct": int(len(exports["only_s2_correct.csv"])),
        "only_s2_wrong": int(len(exports["only_s2_wrong.csv"])),
        "all_wrong": int(len(exports["all_wrong.csv"])),
        "by_route": by_route,
    }

    summary_json_path = args.output_dir / "s2_real_error_analysis_summary.json"
    summary_txt_path = args.output_dir / "s2_real_error_analysis_summary.txt"

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("S2 real error analysis summary\n")
        f.write("=" * 80 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nS2 real error analysis")
    print("----------------------")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nBy route:")
    print(by_route_df.to_string(index=False))

    print("\nArchivos generados en:")
    print(args.output_dir)


if __name__ == "__main__":
    main()

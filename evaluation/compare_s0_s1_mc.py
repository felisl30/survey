#!/usr/bin/env python3
"""
compare_s0_s1_mc.py

Compara S0 directo vs S1 RAG clásico sobre el mismo benchmark MC.
Genera CSVs de casos corregidos, regresiones y resumen global.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


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


def mean_bool(series: pd.Series) -> float:
    cleaned = series.dropna().map(coerce_bool).dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.mean())


def mean_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


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
        "--retrieval-metrics-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/retrieval/retrieval_metrics_by_question.csv"),
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/comparison_s0_s1"),
    )

    args = parser.parse_args()

    if not args.s0_results_path.exists():
        raise FileNotFoundError(args.s0_results_path)
    if not args.s1_results_path.exists():
        raise FileNotFoundError(args.s1_results_path)

    s0 = pd.read_csv(args.s0_results_path)
    s1 = pd.read_csv(args.s1_results_path)

    required_s0 = {"id", "mc_pred", "mc_gold", "mc_correct"}
    required_s1 = {"id", "parsed_answer", "gold_answer", "eval_correct"}

    missing_s0 = required_s0 - set(s0.columns)
    missing_s1 = required_s1 - set(s1.columns)

    if missing_s0:
        raise ValueError(f"Faltan columnas en S0: {sorted(missing_s0)}")
    if missing_s1:
        raise ValueError(f"Faltan columnas en S1: {sorted(missing_s1)}")

    s0_keep = s0[[
        c for c in [
            "id",
            "question",
            "original_question",
            "A",
            "B",
            "C",
            "D",
            "mc_pred",
            "mc_gold",
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
            "gold_answer",
            "eval_correct",
            "retrieval_latency_seconds",
            "generation_latency_seconds",
            "latency_seconds",
            "total_tokens",
            "n_docs_retrieved",
            "retrieved_doc_ids_json",
            "retrieved_titles_json",
            "retrieved_scores_json",
            "retrieved_context_json",
            "raw_output",
        ]
        if c in s1.columns
    ]].copy()

    rename_s0 = {
        "mc_pred": "s0_pred",
        "mc_gold": "s0_gold",
        "mc_correct": "s0_correct",
        "total_tokens": "s0_total_tokens",
        "latency_seconds": "s0_latency_seconds",
    }

    rename_s1 = {
        "parsed_answer": "s1_pred",
        "gold_answer": "s1_gold",
        "eval_correct": "s1_correct",
        "total_tokens": "s1_total_tokens",
        "latency_seconds": "s1_latency_seconds",
    }

    s0_keep = s0_keep.rename(columns=rename_s0)
    s1_keep = s1_keep.rename(columns=rename_s1)

    merged = s0_keep.merge(s1_keep, on="id", how="inner")

    merged["s0_correct_bool"] = merged["s0_correct"].map(coerce_bool)
    merged["s1_correct_bool"] = merged["s1_correct"].map(coerce_bool)

    if args.retrieval_metrics_path.exists():
        retrieval = pd.read_csv(args.retrieval_metrics_path)
        retrieval_k = retrieval[retrieval["k"] == args.retrieval_k].copy()

        retrieval_k = retrieval_k.rename(columns={
            "question_id": "id",
            "hit_at_k": f"retrieval_hit_at_{args.retrieval_k}",
            "recall_at_k": f"retrieval_recall_at_{args.retrieval_k}",
            "same_question_rate_at_k": f"same_question_rate_at_{args.retrieval_k}",
            "mrr_at_k": f"mrr_at_{args.retrieval_k}",
        })

        keep_cols = [
            "id",
            f"retrieval_hit_at_{args.retrieval_k}",
            f"retrieval_recall_at_{args.retrieval_k}",
            f"same_question_rate_at_{args.retrieval_k}",
            f"mrr_at_{args.retrieval_k}",
        ]

        merged = merged.merge(retrieval_k[keep_cols], on="id", how="left")

    def label_row(row: pd.Series) -> str:
        s0_ok = bool(row["s0_correct_bool"])
        s1_ok = bool(row["s1_correct_bool"])

        if s0_ok and s1_ok:
            return "both_correct"
        if not s0_ok and s1_ok:
            return "corrected_by_s1"
        if s0_ok and not s1_ok:
            return "regressed_by_s1"
        return "both_wrong"

    merged["comparison_label"] = merged.apply(label_row, axis=1)

    n = len(merged)
    s0_accuracy = mean_bool(merged["s0_correct_bool"])
    s1_accuracy = mean_bool(merged["s1_correct_bool"])

    s0_errors = int((merged["s0_correct_bool"] == False).sum())
    s1_errors = int((merged["s1_correct_bool"] == False).sum())

    summary = {
        "n": n,
        "s0_accuracy": s0_accuracy,
        "s1_accuracy": s1_accuracy,
        "absolute_delta_accuracy": s1_accuracy - s0_accuracy,
        "s0_errors": s0_errors,
        "s1_errors": s1_errors,
        "errors_reduced": s0_errors - s1_errors,
        "relative_error_reduction": (s0_errors - s1_errors) / s0_errors if s0_errors else None,
        "counts_by_label": merged["comparison_label"].value_counts().to_dict(),
        "s0_avg_total_tokens": mean_numeric(merged["s0_total_tokens"]) if "s0_total_tokens" in merged else None,
        "s1_avg_total_tokens": mean_numeric(merged["s1_total_tokens"]) if "s1_total_tokens" in merged else None,
        "s0_avg_latency_seconds": mean_numeric(merged["s0_latency_seconds"]) if "s0_latency_seconds" in merged else None,
        "s1_avg_latency_seconds": mean_numeric(merged["s1_latency_seconds"]) if "s1_latency_seconds" in merged else None,
    }

    hit_col = f"retrieval_hit_at_{args.retrieval_k}"
    if hit_col in merged.columns:
        by_hit = {}
        for hit_value, subset in merged.groupby(hit_col, dropna=False):
            by_hit[str(hit_value)] = {
                "n": int(len(subset)),
                "s1_accuracy": mean_bool(subset["s1_correct_bool"]),
                "s0_accuracy": mean_bool(subset["s0_correct_bool"]),
            }
        summary[f"by_retrieval_hit_at_{args.retrieval_k}"] = by_hit

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = args.output_dir / "s0_vs_s1_full_comparison.csv"
    corrected_path = args.output_dir / "s1_corrected_s0_errors.csv"
    regressed_path = args.output_dir / "s1_regressed_s0_correct.csv"
    both_wrong_path = args.output_dir / "both_wrong.csv"
    summary_json_path = args.output_dir / "s0_vs_s1_summary.json"
    summary_txt_path = args.output_dir / "s0_vs_s1_summary.txt"

    merged.to_csv(comparison_path, index=False)
    merged[merged["comparison_label"] == "corrected_by_s1"].to_csv(corrected_path, index=False)
    merged[merged["comparison_label"] == "regressed_by_s1"].to_csv(regressed_path, index=False)
    merged[merged["comparison_label"] == "both_wrong"].to_csv(both_wrong_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("S0 vs S1 comparison summary\n")
        f.write("=" * 80 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\nS0 vs S1 comparison")
    print("-------------------")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nCasos por categoría:")
    print(merged["comparison_label"].value_counts().to_string())

    print("\nArchivos generados:")
    print(f"- {comparison_path}")
    print(f"- {corrected_path}")
    print(f"- {regressed_path}")
    print(f"- {both_wrong_path}")
    print(f"- {summary_json_path}")
    print(f"- {summary_txt_path}")


if __name__ == "__main__":
    main()

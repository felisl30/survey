#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_raw.csv")
EVAL_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_evaluated.csv")
SUMMARY_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_summary.json")
OUT_DIR = Path("outputs/eval_mc/musique_mc_rag/s3_mc/analysis")


def norm_answer(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().upper()
    return x if x in {"A", "B", "C", "D"} else ""


def bool_col(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "si", "sí"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_PATH)
    ev = pd.read_csv(EVAL_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    df = raw.copy()

    df["gold_norm"] = df["gold_answer"].map(norm_answer)
    df["candidate_norm"] = df["candidate_answer"].map(norm_answer)
    df["final_norm"] = df["final_answer"].map(norm_answer)
    df["candidate_correct"] = df["candidate_norm"] == df["gold_norm"]
    df["final_correct"] = df["final_norm"] == df["gold_norm"]
    df["used_retrieval"] = df["predicted_route"].astype(str).eq("retrieve")

    def category(row):
        cand = bool(row["candidate_correct"])
        final = bool(row["final_correct"])
        ret = bool(row["used_retrieval"])

        if not ret:
            if final:
                return "direct_correct"
            return "direct_wrong"

        if not cand and final:
            return "retrieval_corrected_candidate"
        if cand and not final:
            return "retrieval_regressed_candidate"
        if cand and final:
            return "retrieval_kept_correct"
        return "retrieval_kept_wrong"

    df["s3_mc_impact_category"] = df.apply(category, axis=1)

    impact_counts = df["s3_mc_impact_category"].value_counts().to_dict()

    by_route = (
        df.groupby("predicted_route", dropna=False)
        .agg(
            n=("id", "count"),
            accuracy=("final_correct", "mean"),
            candidate_accuracy=("candidate_correct", "mean"),
            avg_tokens=("total_tokens", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            avg_top1_score=("top1_score", "mean"),
            avg_gap=("top1_top2_gap", "mean"),
        )
        .reset_index()
    )

    changed = df[df["candidate_norm"] != df["final_norm"]].copy()
    corrected = df[(df["candidate_correct"] == False) & (df["final_correct"] == True)].copy()
    regressed = df[(df["candidate_correct"] == True) & (df["final_correct"] == False)].copy()
    wrong = df[df["final_correct"] == False].copy()

    df.to_csv(OUT_DIR / "s3_mc_full_with_impact_categories.csv", index=False)
    by_route.to_csv(OUT_DIR / "s3_mc_by_route_summary.csv", index=False)
    changed.to_csv(OUT_DIR / "s3_mc_candidate_changed_cases.csv", index=False)
    corrected.to_csv(OUT_DIR / "s3_mc_retrieval_corrected_cases.csv", index=False)
    regressed.to_csv(OUT_DIR / "s3_mc_retrieval_regressed_cases.csv", index=False)
    wrong.to_csv(OUT_DIR / "s3_mc_wrong_cases.csv", index=False)

    overall = summary.get("overall", summary)

    report = {
        "n": int(len(df)),
        "official_accuracy": overall.get("accuracy"),
        "candidate_accuracy": float(df["candidate_correct"].mean()),
        "final_accuracy_recomputed": float(df["final_correct"].mean()),
        "retrieve_rate": float(df["used_retrieval"].mean()),
        "direct_rate": float((~df["used_retrieval"]).mean()),
        "avg_total_tokens": float(pd.to_numeric(df["total_tokens"], errors="coerce").mean()),
        "avg_latency_seconds": float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean()),
        "impact_counts": impact_counts,
        "n_candidate_changed": int(len(changed)),
        "n_retrieval_corrected_candidate": int(len(corrected)),
        "n_retrieval_regressed_candidate": int(len(regressed)),
        "n_final_wrong": int(len(wrong)),
    }

    (OUT_DIR / "s3_mc_analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUT_DIR / "s3_mc_analysis_summary.txt").open("w", encoding="utf-8") as f:
        f.write("S3-MC final analysis\n")
        f.write("=" * 80 + "\n\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print("\nS3-MC final analysis")
    print("--------------------")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nBy route:")
    print(by_route.to_string(index=False))

    print("\nArchivos generados en:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()

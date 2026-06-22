#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("outputs/eval_mc/musique_mc_rag/final_comparison_s0_s1_s2_s3")
S3_SUMMARY_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_summary.json")
S3_ANALYSIS_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/analysis/s3_mc_analysis_summary.json")


def pct(x):
    return round(float(x) * 100, 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s3_summary = json.loads(S3_SUMMARY_PATH.read_text(encoding="utf-8"))
    s3_overall = s3_summary.get("overall", s3_summary)

    s3_analysis = json.loads(S3_ANALYSIS_PATH.read_text(encoding="utf-8"))

    rows = [
        {
            "system": "S0 direct",
            "description": "LLM directo sin retrieval",
            "accuracy": 0.69,
            "accuracy_pct": 69.0,
            "retrieval_rate": 0.0,
            "retrieval_rate_pct": 0.0,
            "avg_total_tokens": 914.26,
            "avg_latency_seconds": 6.16877,
            "main_strength": "baseline simple y barato",
        },
        {
            "system": "S1 classic RAG",
            "description": "RAG fijo top-5",
            "accuracy": 0.78,
            "accuracy_pct": 78.0,
            "retrieval_rate": 1.0,
            "retrieval_rate_pct": 100.0,
            "avg_total_tokens": 1362.82,
            "avg_latency_seconds": 5.22724,
            "main_strength": "mejora accuracy con memoria externa fija",
        },
        {
            "system": "S2 real adaptive",
            "description": "Adaptive-RAG pre-routing",
            "accuracy": 0.81,
            "accuracy_pct": 81.0,
            "retrieval_rate": 0.60,
            "retrieval_rate_pct": 60.0,
            "avg_total_tokens": 1094.62,
            "avg_latency_seconds": 6.89337,
            "main_strength": "mejor balance accuracy/costo",
        },
        {
            "system": "S3-MC FLARE-like",
            "description": "retrieval activo durante generación MC",
            "accuracy": float(s3_overall["accuracy"]),
            "accuracy_pct": pct(s3_overall["accuracy"]),
            "retrieval_rate": float(s3_analysis["retrieve_rate"]),
            "retrieval_rate_pct": pct(s3_analysis["retrieve_rate"]),
            "avg_total_tokens": float(s3_overall["avg_total_tokens"]),
            "avg_latency_seconds": float(s3_overall["avg_latency_seconds"]),
            "main_strength": "mejor accuracy; corrige hipótesis iniciales con retrieval",
        },
    ]

    df = pd.DataFrame(rows)

    # Métricas derivadas
    s2_tokens = float(df.loc[df["system"] == "S2 real adaptive", "avg_total_tokens"].iloc[0])
    s3_tokens = float(df.loc[df["system"] == "S3-MC FLARE-like", "avg_total_tokens"].iloc[0])

    derived = {
        "s3_accuracy_gain_vs_s0": float(s3_overall["accuracy"]) - 0.69,
        "s3_accuracy_gain_vs_s1": float(s3_overall["accuracy"]) - 0.78,
        "s3_accuracy_gain_vs_s2": float(s3_overall["accuracy"]) - 0.81,
        "s3_extra_tokens_vs_s2": s3_tokens - s2_tokens,
        "s3_token_ratio_vs_s2": s3_tokens / s2_tokens,
        "s3_candidate_accuracy": s3_analysis["candidate_accuracy"],
        "s3_final_accuracy": s3_analysis["final_accuracy_recomputed"],
        "s3_retrieval_corrected_candidate": s3_analysis["n_retrieval_corrected_candidate"],
        "s3_retrieval_regressed_candidate": s3_analysis["n_retrieval_regressed_candidate"],
        "s3_final_wrong": s3_analysis["n_final_wrong"],
        "s3_impact_counts": s3_analysis["impact_counts"],
    }

    df.to_csv(OUT_DIR / "s0_s1_s2_s3_mc_final_summary.csv", index=False)

    with (OUT_DIR / "s0_s1_s2_s3_mc_final_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"systems": rows, "derived": derived}, f, ensure_ascii=False, indent=2)

    # Markdown manual para evitar dependencia tabulate
    lines = []
    lines.append("# Final comparison: S0 / S1 / S2 / S3-MC on MuSiQue-MC")
    lines.append("")
    lines.append("| System | Accuracy | Retrieval rate | Avg tokens | Avg latency | Main strength |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['accuracy_pct']:.2f}% | "
            f"{row['retrieval_rate_pct']:.2f}% | {row['avg_total_tokens']:.2f} | "
            f"{row['avg_latency_seconds']:.2f}s | {row['main_strength']} |"
        )

    lines.append("")
    lines.append("## S3-MC impact analysis")
    lines.append("")
    lines.append(f"- Candidate accuracy before retrieval/regeneration: {pct(s3_analysis['candidate_accuracy']):.2f}%")
    lines.append(f"- Final S3-MC accuracy: {pct(s3_analysis['final_accuracy_recomputed']):.2f}%")
    lines.append(f"- Retrieval corrected candidate errors: {s3_analysis['n_retrieval_corrected_candidate']}")
    lines.append(f"- Retrieval regressed candidate answers: {s3_analysis['n_retrieval_regressed_candidate']}")
    lines.append(f"- Final wrong answers: {s3_analysis['n_final_wrong']}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "S3-MC achieves the best accuracy on MuSiQue-MC, improving over S2 real adaptive "
        "by 4 absolute accuracy points. The improvement is not incidental: the initial "
        "candidate accuracy was 68%, while the final answer accuracy after active retrieval "
        "and regeneration reached 85%. Retrieval corrected 18 initially wrong candidate "
        "answers and regressed only 1 initially correct candidate."
    )
    lines.append("")
    lines.append(
        "However, S3-MC is substantially more expensive than S2. Its average token usage is "
        f"{s3_tokens:.2f} tokens/question compared with {s2_tokens:.2f} for S2. "
        "Therefore, S3-MC is the strongest accuracy-oriented system, while S2 remains the "
        "best efficiency-oriented adaptive system."
    )

    md = "\n".join(lines) + "\n"
    (OUT_DIR / "s0_s1_s2_s3_mc_final_summary.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "s0_s1_s2_s3_mc_final_summary.txt").write_text(md, encoding="utf-8")

    print(md)
    print(f"\nArchivos generados en: {OUT_DIR}")


if __name__ == "__main__":
    main()

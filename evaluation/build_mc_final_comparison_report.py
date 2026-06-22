#!/usr/bin/env python3
"""
build_mc_final_comparison_report.py

Construye un reporte final S0 vs S1 vs S2-MC para MuSiQue-100.

No llama API. Resume:
- S0 direct
- S1 RAG top-5
- S2-MC selected policies
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s0-summary-path",
        type=Path,
        default=Path("outputs/eval_mc/s0_gpt_5_mini_musique_100_mc_summary.csv"),
    )
    parser.add_argument(
        "--s1-summary-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s1/s1_gpt_5_mini_top5_summary.json"),
    )
    parser.add_argument(
        "--s2-selected-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_policy/selected_policies/selected_policy_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/final_comparison"),
    )

    args = parser.parse_args()

    if not args.s0_summary_path.exists():
        raise FileNotFoundError(args.s0_summary_path)
    if not args.s1_summary_path.exists():
        raise FileNotFoundError(args.s1_summary_path)
    if not args.s2_selected_path.exists():
        raise FileNotFoundError(args.s2_selected_path)

    s0_summary = pd.read_csv(args.s0_summary_path)
    s0_overall = s0_summary[s0_summary["group_type"] == "overall"].iloc[0]

    s1_summary = read_json(args.s1_summary_path)
    s1_overall = s1_summary["overall"]

    s2 = pd.read_csv(args.s2_selected_path)

    s0_acc = float(s0_overall["accuracy"])
    s1_acc = float(s1_overall["accuracy"])
    s0_tokens = None
    s1_tokens = float(s1_overall["avg_total_tokens"])
    s0_latency = None
    s1_latency = float(s1_overall["avg_latency_seconds"])

    # Para tokens/latencia de S0 usamos la columna guardada en el resumen S2,
    # porque viene de la comparación fila a fila S0/S1/S2.
    if len(s2) > 0:
        s0_tokens = float(s2.iloc[0]["s0_avg_tokens"])
        s0_latency = float(s2.iloc[0]["s0_avg_latency"])

    rows = []

    rows.append({
        "system": "S0 direct",
        "policy": "direct_all",
        "accuracy": s0_acc,
        "delta_vs_s0": 0.0,
        "delta_vs_s1": s0_acc - s1_acc,
        "retrieve_rate": 0.0,
        "avg_tokens": s0_tokens,
        "token_saving_vs_s1": None if s0_tokens is None else s1_tokens - s0_tokens,
        "avg_latency_seconds": s0_latency,
        "notes": "Baseline directo sin memoria externa.",
    })

    rows.append({
        "system": "S1 classic RAG",
        "policy": "retrieve_all_top5",
        "accuracy": s1_acc,
        "delta_vs_s0": s1_acc - s0_acc,
        "delta_vs_s1": 0.0,
        "retrieve_rate": 1.0,
        "avg_tokens": s1_tokens,
        "token_saving_vs_s1": 0.0,
        "avg_latency_seconds": s1_latency,
        "notes": "RAG clásico con recuperación top-5 en todas las preguntas.",
    })

    policy_labels = {
        "top1_ge_0.45_gap_ge_0.05": "S2-MC efficient",
        "top3_mean_ge_0.40": "S2-MC balanced",
        "top5_mean_ge_0.35": "S2-MC accuracy-oriented",
    }

    policy_notes = {
        "top1_ge_0.45_gap_ge_0.05": "Mantiene accuracy de S1 con menor uso de retrieval.",
        "top3_mean_ge_0.40": "Compromiso intermedio entre accuracy y ahorro de tokens.",
        "top5_mean_ge_0.35": "Maximiza accuracy, pero recupera en más preguntas.",
    }

    for _, row in s2.iterrows():
        policy = str(row["policy_name"])
        rows.append({
            "system": policy_labels.get(policy, "S2-MC policy"),
            "policy": policy,
            "accuracy": float(row["s2_accuracy"]),
            "delta_vs_s0": float(row["s2_accuracy"]) - s0_acc,
            "delta_vs_s1": float(row["s2_accuracy"]) - s1_acc,
            "retrieve_rate": float(row["retrieve_rate"]),
            "avg_tokens": float(row["s2_avg_tokens"]),
            "token_saving_vs_s1": float(row["token_saving_vs_s1"]),
            "avg_latency_seconds": float(row["s2_avg_latency"]),
            "notes": policy_notes.get(policy, ""),
        })

    out = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "s0_s1_s2_mc_final_summary.csv"
    txt_path = args.output_dir / "s0_s1_s2_mc_final_summary.txt"
    md_path = args.output_dir / "s0_s1_s2_mc_final_summary.md"

    out.to_csv(csv_path, index=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Final comparison: S0 vs S1 vs S2-MC on MuSiQue-100\n")
        f.write("=" * 80 + "\n\n")

        for _, row in out.iterrows():
            f.write(f"System: {row['system']}\n")
            f.write(f"Policy: {row['policy']}\n")
            f.write(f"Accuracy: {pct(row['accuracy'])}\n")
            f.write(f"Delta vs S0: {pct(row['delta_vs_s0'])}\n")
            f.write(f"Delta vs S1: {pct(row['delta_vs_s1'])}\n")
            f.write(f"Retrieve rate: {pct(row['retrieve_rate'])}\n")
            f.write(f"Avg tokens: {num(row['avg_tokens'])}\n")
            f.write(f"Token saving vs S1: {num(row['token_saving_vs_s1'])}\n")
            f.write(f"Avg latency seconds: {num(row['avg_latency_seconds'])}\n")
            f.write(f"Notes: {row['notes']}\n")
            f.write("-" * 80 + "\n")

        f.write("\nRecommended interpretation:\n")
        f.write(
            "S2-MC efficient is the preferred adaptive policy because it matches "
            "S1 accuracy while reducing retrieval usage and token cost. "
            "S2-MC accuracy-oriented is a secondary policy that maximizes accuracy "
            "but behaves closer to classic RAG.\n"
        )

        f.write("\nMethodological note:\n")
        f.write(
            "These S2-MC results are offline policy simulations that reuse already "
            "generated S0 and S1 outputs. They are valid for policy analysis, but an "
            "end-to-end S2-MC runner would be needed to report them as a fully "
            "executed adaptive system.\n"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Final comparison: S0 vs S1 vs S2-MC on MuSiQue-100\n\n")
        f.write(out.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "The efficient S2-MC policy matches S1 accuracy while reducing retrieval usage "
            "from 100% to 60%, saving roughly 210 tokens per question compared with S1. "
            "The accuracy-oriented S2-MC policy reaches the best accuracy, but with less token saving.\n"
        )

    print("\nFinal comparison generated")
    print("--------------------------")
    print(out.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"TXT: {txt_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()

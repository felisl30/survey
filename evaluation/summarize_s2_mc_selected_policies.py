#!/usr/bin/env python3
"""
summarize_s2_mc_selected_policies.py

Resume políticas S2-MC offline seleccionadas y exporta casos por política.

No llama API. Usa:
- outputs/eval_mc/musique_mc_rag/s2_policy/s2_policy_grid_summary.csv
- outputs/eval_mc/musique_mc_rag/s2_policy/s2_policy_all_routes.csv
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


def mean_bool(series: pd.Series) -> float | None:
    vals = series.map(coerce_bool).dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def mean_num(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def summarize_cases(df: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    s0 = df["s0_correct"].map(coerce_bool)
    s1 = df["s1_correct"].map(coerce_bool)
    s2 = df["s2_correct"].map(coerce_bool)

    return {
        "policy_name": policy_name,
        "n": int(len(df)),
        "s0_accuracy": mean_bool(df["s0_correct"]),
        "s1_accuracy": mean_bool(df["s1_correct"]),
        "s2_accuracy": mean_bool(df["s2_correct"]),
        "retrieve_rate": float((df["s2_route"] == "retrieve").mean()),
        "direct_rate": float((df["s2_route"] == "direct").mean()),
        "s0_avg_tokens": mean_num(df["s0_total_tokens"]),
        "s1_avg_tokens": mean_num(df["s1_total_tokens"]),
        "s2_avg_tokens": mean_num(df["s2_total_tokens"]),
        "token_saving_vs_s1": mean_num(df["s1_total_tokens"]) - mean_num(df["s2_total_tokens"]),
        "token_increase_vs_s0": mean_num(df["s2_total_tokens"]) - mean_num(df["s0_total_tokens"]),
        "s0_avg_latency": mean_num(df["s0_latency_seconds"]),
        "s1_avg_latency": mean_num(df["s1_latency_seconds"]),
        "s2_avg_latency": mean_num(df["s2_latency_seconds"]),
        "both_s0_s1_s2_correct": int((s0 & s1 & s2).sum()),
        "s2_correct_s0_wrong": int((s2 & ~s0).sum()),
        "s2_wrong_s0_correct": int((~s2 & s0).sum()),
        "s2_correct_s1_wrong": int((s2 & ~s1).sum()),
        "s2_wrong_s1_correct": int((~s2 & s1).sum()),
        "s2_uses_direct_and_correct": int(((df["s2_route"] == "direct") & s2).sum()),
        "s2_uses_retrieve_and_correct": int(((df["s2_route"] == "retrieve") & s2).sum()),
        "s2_uses_direct_and_wrong": int(((df["s2_route"] == "direct") & ~s2).sum()),
        "s2_uses_retrieve_and_wrong": int(((df["s2_route"] == "retrieve") & ~s2).sum()),
    }


def safe_filename(name: str) -> str:
    return (
        name
        .replace(".", "_")
        .replace("-", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("=", "eq")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_policy/s2_policy_grid_summary.csv"),
    )
    parser.add_argument(
        "--all-routes-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_policy/s2_policy_all_routes.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_policy/selected_policies"),
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="top1_ge_0.45_gap_ge_0.05,top3_mean_ge_0.40,top5_mean_ge_0.35",
    )

    args = parser.parse_args()

    if not args.summary_path.exists():
        raise FileNotFoundError(args.summary_path)
    if not args.all_routes_path.exists():
        raise FileNotFoundError(args.all_routes_path)

    grid = pd.read_csv(args.summary_path)
    routes = pd.read_csv(args.all_routes_path)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_summaries = []
    missing = []

    for policy in policies:
        sub = routes[routes["s2_policy_name"] == policy].copy()

        if sub.empty:
            missing.append(policy)
            continue

        selected_summaries.append(summarize_cases(sub, policy))

        prefix = safe_filename(policy)

        sub.to_csv(args.output_dir / f"{prefix}_cases.csv", index=False)

        # Casos donde la política corrige S0.
        sub[
            (sub["s0_correct"].map(coerce_bool) == False)
            & (sub["s2_correct"].map(coerce_bool) == True)
        ].to_csv(args.output_dir / f"{prefix}_corrected_s0_errors.csv", index=False)

        # Casos donde la política empeora S0.
        sub[
            (sub["s0_correct"].map(coerce_bool) == True)
            & (sub["s2_correct"].map(coerce_bool) == False)
        ].to_csv(args.output_dir / f"{prefix}_regressed_s0_correct.csv", index=False)

        # Casos donde S2 difiere de S1.
        sub[
            sub["s2_pred"].astype(str) != sub["s1_pred"].astype(str)
        ].to_csv(args.output_dir / f"{prefix}_differs_from_s1.csv", index=False)

    selected_df = pd.DataFrame(selected_summaries)
    selected_summary_path = args.output_dir / "selected_policy_summary.csv"
    selected_json_path = args.output_dir / "selected_policy_summary.json"
    selected_txt_path = args.output_dir / "selected_policy_summary.txt"

    selected_df.to_csv(selected_summary_path, index=False)

    with selected_json_path.open("w", encoding="utf-8") as f:
        json.dump(selected_summaries, f, ensure_ascii=False, indent=2)

    with selected_txt_path.open("w", encoding="utf-8") as f:
        f.write("Selected S2-MC policy summary\n")
        f.write("=" * 80 + "\n\n")

        for item in selected_summaries:
            f.write(f"Policy: {item['policy_name']}\n")
            f.write("-" * 80 + "\n")
            for key, value in item.items():
                if key != "policy_name":
                    f.write(f"{key}: {value}\n")
            f.write("\n")

        if missing:
            f.write("Missing policies:\n")
            for policy in missing:
                f.write(f"- {policy}\n")

    print("\nSelected S2-MC policies")
    print("-----------------------")
    print(selected_df.to_string(index=False))

    if missing:
        print("\nPolíticas no encontradas:")
        for policy in missing:
            print(f"- {policy}")

    print("\nArchivos generados:")
    print(f"- {selected_summary_path}")
    print(f"- {selected_json_path}")
    print(f"- {selected_txt_path}")
    print(f"- CSVs por política en {args.output_dir}")


if __name__ == "__main__":
    main()

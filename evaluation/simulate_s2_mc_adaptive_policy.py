#!/usr/bin/env python3
"""
simulate_s2_mc_adaptive_policy.py

Simula S2 Adaptive-RAG sobre MuSiQue-100 MC sin llamar a la API.

Usa:
- S0 ya evaluado como rama direct.
- S1 ya evaluado como rama retrieve.
- Scores del retriever como señal de routing.

Política básica:
    retrieve si una señal del retriever supera cierto umbral;
    direct en caso contrario.

También incluye una política oracle_hit_at_5 solo como techo diagnóstico:
usa retrieval_hit_at_5, que depende de qrels, por lo tanto NO es usable en
inferencia real. Sirve para estimar el máximo esperable de una política que
detectara perfectamente cuándo retrieval recuperó evidencia.
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
    vals = series.map(coerce_bool).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def mean_numeric(series: pd.Series) -> float | None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def parse_json_list(value: Any) -> list[Any]:
    text = clean_str(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def build_retrieval_features(s1: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in s1.iterrows():
        qid = clean_str(row["id"])
        scores = parse_json_list(row.get("retrieved_scores_json", ""))

        scores_float = []
        for x in scores:
            try:
                scores_float.append(float(x))
            except Exception:
                pass

        top1 = scores_float[0] if len(scores_float) >= 1 else None
        top2 = scores_float[1] if len(scores_float) >= 2 else None
        top3 = scores_float[2] if len(scores_float) >= 3 else None

        rows.append({
            "id": qid,
            "top1_score": top1,
            "top2_score": top2,
            "top3_score": top3,
            "top1_top2_gap": None if top1 is None or top2 is None else top1 - top2,
            "top3_mean_score": None if len(scores_float) < 3 else sum(scores_float[:3]) / 3,
            "top5_mean_score": None if len(scores_float) < 5 else sum(scores_float[:5]) / 5,
        })

    return pd.DataFrame(rows)


def summarize_policy(df: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    n = len(df)

    retrieve_rate = float((df["s2_route"] == "retrieve").mean()) if n else None
    direct_rate = float((df["s2_route"] == "direct").mean()) if n else None

    s0_acc = mean_bool(df["s0_correct"])
    s1_acc = mean_bool(df["s1_correct"])
    s2_acc = mean_bool(df["s2_correct"])

    s0_tokens = mean_numeric(df["s0_total_tokens"])
    s1_tokens = mean_numeric(df["s1_total_tokens"])
    s2_tokens = mean_numeric(df["s2_total_tokens"])

    s0_latency = mean_numeric(df["s0_latency_seconds"])
    s1_latency = mean_numeric(df["s1_latency_seconds"])
    s2_latency = mean_numeric(df["s2_latency_seconds"])

    s0_errors = int((df["s0_correct"].map(coerce_bool) == False).sum())
    s1_errors = int((df["s1_correct"].map(coerce_bool) == False).sum())
    s2_errors = int((df["s2_correct"].map(coerce_bool) == False).sum())

    route_quality = {}
    for route, subset in df.groupby("s2_route", dropna=False):
        route_quality[str(route)] = {
            "n": int(len(subset)),
            "accuracy": mean_bool(subset["s2_correct"]),
            "avg_total_tokens": mean_numeric(subset["s2_total_tokens"]),
            "avg_latency_seconds": mean_numeric(subset["s2_latency_seconds"]),
        }

    return {
        "policy_name": policy_name,
        "n": n,
        "s0_accuracy": s0_acc,
        "s1_accuracy": s1_acc,
        "s2_accuracy": s2_acc,
        "delta_vs_s0": None if s2_acc is None else s2_acc - s0_acc,
        "delta_vs_s1": None if s2_acc is None else s2_acc - s1_acc,
        "s0_errors": s0_errors,
        "s1_errors": s1_errors,
        "s2_errors": s2_errors,
        "retrieve_rate": retrieve_rate,
        "direct_rate": direct_rate,
        "counts_by_route": df["s2_route"].value_counts().to_dict(),
        "s0_avg_total_tokens": s0_tokens,
        "s1_avg_total_tokens": s1_tokens,
        "s2_avg_total_tokens": s2_tokens,
        "token_saving_vs_s1": None if s1_tokens is None or s2_tokens is None else s1_tokens - s2_tokens,
        "token_increase_vs_s0": None if s0_tokens is None or s2_tokens is None else s2_tokens - s0_tokens,
        "s0_avg_latency_seconds": s0_latency,
        "s1_avg_latency_seconds": s1_latency,
        "s2_avg_latency_seconds": s2_latency,
        "latency_saving_vs_s1": None if s1_latency is None or s2_latency is None else s1_latency - s2_latency,
        "route_quality": route_quality,
    }


def build_s2_rows(
    *,
    merged: pd.DataFrame,
    policy_name: str,
    threshold: float | None,
    mode: str,
) -> pd.DataFrame:
    df = merged.copy()

    if mode == "direct_all":
        df["s2_route"] = "direct"

    elif mode == "retrieve_all":
        df["s2_route"] = "retrieve"

    elif mode == "threshold_top1":
        if threshold is None:
            raise ValueError("threshold requerido para threshold_top1")
        df["s2_route"] = df["top1_score"].apply(
            lambda x: "retrieve" if pd.notna(x) and float(x) >= threshold else "direct"
        )

    elif mode == "threshold_top3_mean":
        if threshold is None:
            raise ValueError("threshold requerido para threshold_top3_mean")
        df["s2_route"] = df["top3_mean_score"].apply(
            lambda x: "retrieve" if pd.notna(x) and float(x) >= threshold else "direct"
        )

    elif mode == "threshold_top5_mean":
        if threshold is None:
            raise ValueError("threshold requerido para threshold_top5_mean")
        df["s2_route"] = df["top5_mean_score"].apply(
            lambda x: "retrieve" if pd.notna(x) and float(x) >= threshold else "direct"
        )

    elif mode == "threshold_top1_gap":
        if threshold is None:
            raise ValueError("threshold requerido para threshold_top1_gap")
        df["s2_route"] = df.apply(
            lambda r: "retrieve"
            if pd.notna(r.get("top1_score"))
            and pd.notna(r.get("top1_top2_gap"))
            and float(r["top1_score"]) >= threshold
            and float(r["top1_top2_gap"]) >= 0.05
            else "direct",
            axis=1,
        )

    elif mode == "oracle_hit_at_5":
        if "retrieval_hit_at_5" not in df.columns:
            raise ValueError("oracle_hit_at_5 requiere retrieval_hit_at_5")
        df["s2_route"] = df["retrieval_hit_at_5"].map(coerce_bool).apply(
            lambda x: "retrieve" if x is True else "direct"
        )

    else:
        raise ValueError(f"Modo desconocido: {mode}")

    use_retrieve = df["s2_route"] == "retrieve"

    df["s2_pred"] = df["s0_pred"]
    df.loc[use_retrieve, "s2_pred"] = df.loc[use_retrieve, "s1_pred"]

    df["s2_correct"] = df["s0_correct"]
    df.loc[use_retrieve, "s2_correct"] = df.loc[use_retrieve, "s1_correct"]

    df["s2_total_tokens"] = df["s0_total_tokens"]
    df.loc[use_retrieve, "s2_total_tokens"] = df.loc[use_retrieve, "s1_total_tokens"]

    df["s2_latency_seconds"] = df["s0_latency_seconds"]
    df.loc[use_retrieve, "s2_latency_seconds"] = df.loc[use_retrieve, "s1_latency_seconds"]

    df["s2_policy_name"] = policy_name
    df["s2_threshold"] = threshold

    return df


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
        default=Path("outputs/eval_mc/musique_mc_rag/s2_policy"),
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )

    args = parser.parse_args()

    if not args.s0_results_path.exists():
        raise FileNotFoundError(args.s0_results_path)
    if not args.s1_results_path.exists():
        raise FileNotFoundError(args.s1_results_path)

    s0 = pd.read_csv(args.s0_results_path)
    s1 = pd.read_csv(args.s1_results_path)

    s0_keep = s0[[
        c for c in [
            "id",
            "question",
            "original_question",
            "A", "B", "C", "D",
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
            "total_tokens",
            "latency_seconds",
            "retrieval_latency_seconds",
            "generation_latency_seconds",
            "n_docs_retrieved",
            "retrieved_doc_ids_json",
            "retrieved_titles_json",
            "retrieved_scores_json",
        ]
        if c in s1.columns
    ]].copy()

    s0_keep = s0_keep.rename(columns={
        "mc_pred": "s0_pred",
        "mc_gold": "gold_answer",
        "mc_correct": "s0_correct",
        "total_tokens": "s0_total_tokens",
        "latency_seconds": "s0_latency_seconds",
    })

    s1_keep = s1_keep.rename(columns={
        "parsed_answer": "s1_pred",
        "gold_answer": "s1_gold_answer",
        "eval_correct": "s1_correct",
        "total_tokens": "s1_total_tokens",
        "latency_seconds": "s1_latency_seconds",
    })

    features = build_retrieval_features(s1)

    merged = s0_keep.merge(s1_keep, on="id", how="inner")
    merged = merged.merge(features, on="id", how="left")

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

    thresholds = [
        float(x.strip())
        for x in args.thresholds.split(",")
        if x.strip()
    ]

    all_policies: list[tuple[str, str, float | None]] = [
        ("direct_all", "direct_all", None),
        ("retrieve_all", "retrieve_all", None),
        ("oracle_hit_at_5_diagnostic_only", "oracle_hit_at_5", None),
    ]

    for t in thresholds:
        all_policies.append((f"top1_ge_{t:.2f}", "threshold_top1", t))
    for t in thresholds:
        all_policies.append((f"top3_mean_ge_{t:.2f}", "threshold_top3_mean", t))
    for t in thresholds:
        all_policies.append((f"top5_mean_ge_{t:.2f}", "threshold_top5_mean", t))
    for t in thresholds:
        all_policies.append((f"top1_ge_{t:.2f}_gap_ge_0.05", "threshold_top1_gap", t))

    policy_outputs = []
    summaries = []

    for policy_name, mode, threshold in all_policies:
        policy_df = build_s2_rows(
            merged=merged,
            policy_name=policy_name,
            threshold=threshold,
            mode=mode,
        )

        summary = summarize_policy(policy_df, policy_name)
        summary["mode"] = mode
        summary["threshold"] = threshold
        summary["diagnostic_only"] = bool(mode == "oracle_hit_at_5")

        summaries.append(summary)
        policy_outputs.append(policy_df)

    summary_df = pd.DataFrame([
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in s.items()
        }
        for s in summaries
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "s2_policy_grid_summary.csv"
    summary_json_path = args.output_dir / "s2_policy_grid_summary.json"
    all_routes_path = args.output_dir / "s2_policy_all_routes.csv"

    summary_df.to_csv(summary_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    all_routes = pd.concat(policy_outputs, ignore_index=True)
    all_routes.to_csv(all_routes_path, index=False)

    usable = summary_df[summary_df["diagnostic_only"] == False].copy()

    s1_acc = float(summary_df[summary_df["policy_name"] == "retrieve_all"]["s2_accuracy"].iloc[0])
    s0_acc = float(summary_df[summary_df["policy_name"] == "direct_all"]["s2_accuracy"].iloc[0])

    cols = [
        "policy_name",
        "s2_accuracy",
        "delta_vs_s0",
        "delta_vs_s1",
        "retrieve_rate",
        "s2_avg_total_tokens",
        "token_saving_vs_s1",
        "s2_avg_latency_seconds",
        "diagnostic_only",
    ]

    best_accuracy = usable.sort_values(
        ["s2_accuracy", "s2_avg_total_tokens"],
        ascending=[False, True],
    ).head(10)

    near_s1 = usable[usable["s2_accuracy"] >= s1_acc - 0.02].sort_values(
        ["s2_avg_total_tokens", "s2_accuracy"],
        ascending=[True, False],
    ).head(10)

    above_s0 = usable[usable["s2_accuracy"] >= s0_acc + 0.05].sort_values(
        ["s2_avg_total_tokens", "s2_accuracy"],
        ascending=[True, False],
    ).head(10)

    print("\nS2-MC adaptive policy grid")
    print("--------------------------")

    print("\nBaseline y techo diagnóstico:")
    print(summary_df[summary_df["policy_name"].isin([
        "direct_all",
        "retrieve_all",
        "oracle_hit_at_5_diagnostic_only",
    ])][cols].to_string(index=False))

    print("\nTop 10 políticas usables por accuracy:")
    print(best_accuracy[cols].to_string(index=False))

    print("\nMás baratas con accuracy >= S1 - 0.02:")
    if near_s1.empty:
        print("Ninguna.")
    else:
        print(near_s1[cols].to_string(index=False))

    print("\nMás baratas con accuracy >= S0 + 0.05:")
    if above_s0.empty:
        print("Ninguna.")
    else:
        print(above_s0[cols].to_string(index=False))

    print("\nArchivos generados:")
    print(f"- {summary_path}")
    print(f"- {summary_json_path}")
    print(f"- {all_routes_path}")


if __name__ == "__main__":
    main()

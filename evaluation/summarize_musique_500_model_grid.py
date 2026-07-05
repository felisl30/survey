#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_LABELS = {
    "s0": "S0 direct",
    "s1": "S1 RAG top-5",
    "s2": "S2 adaptive RAG",
    "s3": "S3 MC FLARE-like",
}


def model_tag(model: str) -> str:
    return model.replace(".", "_").replace("-", "_").replace("/", "_")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_s0_row(base_out: Path, model: str, expected_n: int | None) -> dict[str, Any] | None:
    tag = model_tag(model)
    prefix = f"s0_{tag}_musique_500"
    summary_path = base_out / f"{prefix}_mc_summary.csv"
    raw_path = base_out / f"{prefix}_raw.csv"
    results_path = base_out / f"{prefix}_mc_results.csv"

    if not summary_path.exists() or not raw_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    overall = summary[summary["group_type"].eq("overall")]
    if overall.empty:
        overall = summary.head(1)
    raw = pd.read_csv(raw_path)

    n = int(overall["n"].iloc[0]) if "n" in overall else len(raw)
    return {
        "model": model,
        "model_tag": tag,
        "system": "s0",
        "system_label": SYSTEM_LABELS["s0"],
        "n": n,
        "expected_n": expected_n,
        "complete": bool(expected_n is None or n >= expected_n),
        "accuracy": float(overall["accuracy"].iloc[0]),
        "valid_format_rate": float(overall.get("valid_prediction_rate", pd.Series([float("nan")])).iloc[0]),
        "run_error_rate": float(raw["error"].notna().mean()) if "error" in raw else float("nan"),
        "avg_input_tokens": pd.to_numeric(raw.get("input_tokens"), errors="coerce").mean(),
        "avg_output_tokens": pd.to_numeric(raw.get("output_tokens"), errors="coerce").mean(),
        "avg_total_tokens": pd.to_numeric(raw.get("total_tokens"), errors="coerce").mean(),
        "avg_latency_seconds": pd.to_numeric(raw.get("latency_seconds"), errors="coerce").mean(),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
    }


def load_rag_row(root: Path, model: str, system: str, expected_n: int | None) -> dict[str, Any] | None:
    tag = model_tag(model)
    model_dir = root / tag
    summary_path = model_dir / f"{system}_summary.json"
    raw_path = model_dir / f"{system}_raw.csv"
    evaluated_path = model_dir / f"{system}_evaluated.csv"

    if not summary_path.exists() or not raw_path.exists():
        return None

    summary = read_json(summary_path)
    overall = summary.get("overall", summary)
    raw = pd.read_csv(raw_path)
    n = int(overall.get("n", len(raw)))

    row = {
        "model": model,
        "model_tag": tag,
        "system": system,
        "system_label": SYSTEM_LABELS[system],
        "n": n,
        "expected_n": expected_n,
        "complete": bool(expected_n is None or n >= expected_n),
        "accuracy": overall.get("accuracy"),
        "valid_format_rate": overall.get("valid_format_rate"),
        "run_error_rate": overall.get("run_error_rate"),
        "avg_input_tokens": overall.get("avg_input_tokens"),
        "avg_output_tokens": overall.get("avg_output_tokens"),
        "avg_total_tokens": overall.get("avg_total_tokens"),
        "avg_latency_seconds": overall.get("avg_latency_seconds"),
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "results_path": str(evaluated_path),
    }

    if system == "s2" and "predicted_route" in raw.columns:
        row["retrieve_rate"] = raw["predicted_route"].astype(str).str.lower().str.contains("retrieve").mean()
    elif system == "s3" and "s3_mc_active_retrieval" in raw.columns:
        row["retrieve_rate"] = raw["s3_mc_active_retrieval"].astype(str).str.lower().isin(["true", "1", "yes"]).mean()
    elif "n_docs_retrieved" in raw.columns:
        row["retrieve_rate"] = (pd.to_numeric(raw["n_docs_retrieved"], errors="coerce").fillna(0) > 0).mean()
    else:
        row["retrieve_rate"] = 0.0 if system == "s0" else float("nan")

    return row


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["accuracy_pct"] = df["accuracy"].astype(float) * 100
    df["accuracy_per_1000_tokens"] = df["accuracy"].astype(float) * 1000 / df["avg_total_tokens"].astype(float)

    baseline_rows = df[df["system"].eq("s0")][["model", "accuracy", "avg_total_tokens"]]
    baseline_rows = baseline_rows.rename(
        columns={"accuracy": "s0_accuracy", "avg_total_tokens": "s0_avg_total_tokens"}
    )
    df = df.merge(baseline_rows, on="model", how="left")
    df["delta_accuracy_vs_s0"] = df["accuracy"].astype(float) - df["s0_accuracy"].astype(float)
    df["token_ratio_vs_s0"] = df["avg_total_tokens"].astype(float) / df["s0_avg_total_tokens"].astype(float)
    return df


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# MuSiQue-500 Model Grid Summary",
        "",
        "| Model | System | n | Complete | Accuracy | Avg tokens | Acc/1k tok | Avg latency | Retrieval rate | Delta acc vs S0 | Token ratio vs S0 |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.sort_values(["model", "system"]).iterrows():
        retrieval = row.get("retrieve_rate")
        retrieval_text = "" if pd.isna(retrieval) else f"{float(retrieval):.3f}"
        lines.append(
            "| {model} | {system} | {n} | {complete} | {acc:.3f} | {tok:.2f} | {acc_tok:.3f} | {lat:.2f}s | {ret} | {delta:.3f} | {ratio:.2f} |".format(
                model=row["model"],
                system=row["system_label"],
                n=int(row["n"]),
                complete="yes" if bool(row["complete"]) else "no",
                acc=float(row["accuracy"]),
                tok=float(row["avg_total_tokens"]),
                acc_tok=float(row["accuracy_per_1000_tokens"]),
                lat=float(row["avg_latency_seconds"]),
                ret=retrieval_text,
                delta=float(row["delta_accuracy_vs_s0"]) if not pd.isna(row["delta_accuracy_vs_s0"]) else float("nan"),
                ratio=float(row["token_ratio_vs_s0"]) if not pd.isna(row["token_ratio_vs_s0"]) else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `Accuracy` compares answer correctness.",
            "- `Avg tokens` is the average total token usage per question.",
            "- `Acc/1k tok` is an efficiency metric: higher means more correct answers per token budget.",
            "- `Delta acc vs S0` measures the gain from adding RAG/adaptive/FLARE over the direct baseline for the same model.",
            "- `Token ratio vs S0` shows how much more/less expensive each system is relative to S0 for the same model.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gpt-5-nano,gpt-5-mini,gpt-4.1-mini")
    parser.add_argument("--systems", default="s0,s1,s2,s3")
    parser.add_argument("--expected-n", type=int, default=500)
    parser.add_argument("--s0-out-dir", type=Path, default=Path("outputs/eval_mc"))
    parser.add_argument("--rag-root", type=Path, default=Path("outputs/eval_mc/musique_mc_rag_500"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_mc/musique_mc_rag_500/model_grid_summary"))
    args = parser.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    systems = [x.strip() for x in args.systems.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for model in models:
        for system in systems:
            if system == "s0":
                row = load_s0_row(args.s0_out_dir, model, args.expected_n)
            else:
                row = load_rag_row(args.rag_root, model, system, args.expected_n)
            if row is None:
                missing.append({"model": model, "system": system})
            else:
                rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_df = pd.DataFrame(missing)
    missing_df.to_csv(args.output_dir / "missing_runs.csv", index=False)

    if not rows:
        raise SystemExit("No encontre ningun resultado para resumir.")

    df = add_derived_metrics(pd.DataFrame(rows))
    df.to_csv(args.output_dir / "model_grid_metrics.csv", index=False)
    write_markdown(df, args.output_dir / "model_grid_metrics.md")

    print(f"Metrics: {args.output_dir / 'model_grid_metrics.csv'}")
    print(f"Report:  {args.output_dir / 'model_grid_metrics.md'}")
    print(f"Missing: {args.output_dir / 'missing_runs.csv'}")


if __name__ == "__main__":
    main()

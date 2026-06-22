#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("outputs/eval_mc/model_grid_musique")
SYSTEMS = ["s0", "s1", "s2", "s3_mc"]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_overall(summary: dict) -> dict:
    if not summary:
        return {}
    return summary.get("overall", summary)


def mean_numeric(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def retrieve_rate(raw: pd.DataFrame, system: str):
    if raw.empty:
        return None
    if system == "s0":
        return 0.0
    if system == "s1":
        return 1.0
    if "predicted_route" in raw.columns:
        return float((raw["predicted_route"].astype(str) == "retrieve").mean())
    if "active_retrieval_triggered" in raw.columns:
        return float(raw["active_retrieval_triggered"].astype(str).str.lower().isin(["true", "1", "yes"]).mean())
    return None


def model_name_from_raw(raw: pd.DataFrame, fallback: str) -> str:
    if "model" in raw.columns and not raw.empty:
        vals = raw["model"].dropna().astype(str)
        vals = vals[vals.str.strip() != ""]
        if not vals.empty:
            return vals.iloc[0]
    return fallback


rows = []

for model_dir in sorted(BASE.iterdir()):
    if not model_dir.is_dir():
        continue

    tag = model_dir.name

    for system in SYSTEMS:
        raw_path = model_dir / f"{system}_raw.csv"
        summary_path = model_dir / f"{system}_summary.json"

        raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
        summary = get_overall(load_json(summary_path))

        rows.append({
            "model_tag": tag,
            "model": model_name_from_raw(raw, tag),
            "system": system,
            "n_raw": int(len(raw)),
            "n_summary": summary.get("n"),
            "accuracy": summary.get("accuracy", summary.get("final_accuracy")),
            "valid_format_rate": summary.get("valid_format_rate", summary.get("valid_answer_format_rate")),
            "run_error_rate": summary.get("run_error_rate"),
            "retrieval_rate": retrieve_rate(raw, system),
            "avg_total_tokens": summary.get("avg_total_tokens", mean_numeric(raw, "total_tokens")),
            "avg_latency_seconds": summary.get("avg_latency_seconds", mean_numeric(raw, "latency_seconds")),
        })

out = pd.DataFrame(rows)

BASE.mkdir(parents=True, exist_ok=True)
out_path = BASE / "model_grid_s0_s3_summary.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print()
print("Guardado en:", out_path)

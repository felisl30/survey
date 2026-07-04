#!/usr/bin/env python3
"""
analyze_musique_robustness_deep.py

Deep analysis for MuSiQue robustness S0-S3 outputs.

Reads the already generated files under:
    outputs/eval_mc/robustness_musique/gpt_5_4_mini

Produces:
    analysis/robustness_deep_system_summary.csv
    analysis/robustness_deep_condition_deltas.csv
    analysis/robustness_deep_question_matrix.csv
    analysis/robustness_deep_patterns_summary.csv
    analysis/robustness_deep_interesting_cases.csv
    analysis/robustness_deep_report.txt

This script does not call any API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VALID_BOOL_TRUE = {"true", "1", "yes", "y", "correct"}
VALID_BOOL_FALSE = {"false", "0", "no", "n", "incorrect"}

SYSTEMS = ["s1", "s2", "s3_mc"]
CONDITIONS = ["clean", "noisy", "adversarial"]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)
    norm = s.astype(str).str.strip().str.lower()
    return norm.map(lambda x: True if x in VALID_BOOL_TRUE else False if x in VALID_BOOL_FALSE else False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary_row(base: Path, condition: str, system: str) -> dict[str, Any]:
    if system == "s0":
        summary_path = base / "s0_summary.json"
    else:
        summary_path = base / condition / f"{system}_summary.json"

    data = read_json(summary_path)
    overall = data.get("overall", {}) if isinstance(data, dict) else {}

    return {
        "condition": condition,
        "system": system,
        "summary_path": str(summary_path),
        "n": overall.get("n"),
        "accuracy": overall.get("accuracy"),
        "valid_format_rate": overall.get("valid_format_rate"),
        "run_error_rate": overall.get("run_error_rate"),
        "avg_confidence": overall.get("avg_confidence"),
        "avg_latency_seconds": overall.get("avg_latency_seconds"),
        "avg_input_tokens": overall.get("avg_input_tokens"),
        "avg_output_tokens": overall.get("avg_output_tokens"),
        "avg_total_tokens": overall.get("avg_total_tokens"),
        "exists": summary_path.exists(),
    }


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_eval_file(base: Path, condition: str, system: str) -> pd.DataFrame:
    if system == "s0":
        path = base / "s0_evaluated.csv"
    else:
        path = base / condition / f"{system}_evaluated.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["condition"] = condition
    df["system"] = system
    df["evaluated_path"] = str(path)
    return df


def read_raw_file(base: Path, condition: str, system: str) -> pd.DataFrame | None:
    if system == "s0":
        path = base / "s0_raw.csv"
    else:
        path = base / condition / f"{system}_raw.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["condition"] = condition
    df["system"] = system
    df["raw_path"] = str(path)
    return df


def normalize_eval(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    id_col = find_col(df, ["id", "question_id", "example_id"])
    if id_col is None:
        raise KeyError(f"No id column in {prefix}: {list(df.columns)}")

    correct_col = find_col(df, ["eval_correct", "mc_correct", "correct", "is_correct"])
    pred_col = find_col(df, ["parsed_answer", "mc_pred", "answer", "final_answer", "predicted_answer"])
    gold_col = find_col(df, ["gold_answer", "mc_gold", "correct_answer", "expected_answer"])

    keep = [id_col]
    rename = {id_col: "id"}

    for col in [correct_col, pred_col, gold_col]:
        if col is not None and col not in keep:
            keep.append(col)

    optional_meta = ["original_question", "question", "A", "B", "C", "D", "dataset"]
    for col in optional_meta:
        if col in df.columns and col not in keep:
            keep.append(col)

    out = df[keep].copy().rename(columns=rename)

    if correct_col is not None:
        out[f"{prefix}_correct"] = to_bool_series(df[correct_col])
    else:
        out[f"{prefix}_correct"] = False

    if pred_col is not None:
        out[f"{prefix}_answer"] = df[pred_col].map(clean_text)

    if gold_col is not None:
        out["gold_answer"] = df[gold_col].map(clean_text)

    return out.drop(columns=[c for c in [correct_col, pred_col, gold_col] if c in out.columns], errors="ignore")


def raw_route_summary(raw: pd.DataFrame | None) -> dict[str, Any]:
    if raw is None or raw.empty:
        return {}

    out: dict[str, Any] = {}

    route_col = find_col(raw, ["predicted_route", "route", "selected_route"])
    if route_col:
        counts = raw[route_col].fillna("missing").astype(str).value_counts(dropna=False).to_dict()
        out["route_counts"] = counts
        total = len(raw)
        for key, val in counts.items():
            out[f"route_rate_{key}"] = val / total if total else None

    retrieval_cols = [
        "retrieval_used",
        "retrieval_triggered",
        "active_retrieval_triggered",
        "used_retrieval",
    ]
    for col in retrieval_cols:
        if col in raw.columns:
            vals = to_bool_series(raw[col])
            out[f"{col}_rate"] = float(vals.mean()) if len(vals) else None

    numeric_candidates = [
        "top1_score",
        "top2_score",
        "score_gap",
        "retrieval_score_top1",
        "retrieval_gap",
        "candidate_confidence",
        "final_confidence",
        "confidence",
        "total_tokens",
        "latency_seconds",
    ]
    for col in numeric_candidates:
        if col in raw.columns:
            nums = pd.to_numeric(raw[col], errors="coerce")
            if nums.notna().any():
                out[f"avg_{col}"] = float(nums.mean())

    return out


def build_system_summary(base: Path) -> pd.DataFrame:
    rows = []

    rows.append(read_summary_row(base, "baseline", "s0"))

    for cond in CONDITIONS:
        for sys in SYSTEMS:
            row = read_summary_row(base, cond, sys)
            raw = read_raw_file(base, cond, sys)
            row.update(raw_route_summary(raw))
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_condition_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sys in SYSTEMS:
        clean = summary[(summary["condition"] == "clean") & (summary["system"] == sys)]
        if clean.empty:
            continue
        clean_row = clean.iloc[0]

        for cond in ["noisy", "adversarial"]:
            target = summary[(summary["condition"] == cond) & (summary["system"] == sys)]
            if target.empty:
                continue
            row = target.iloc[0]

            rows.append({
                "system": sys,
                "condition": cond,
                "clean_accuracy": clean_row.get("accuracy"),
                "condition_accuracy": row.get("accuracy"),
                "delta_accuracy_vs_clean": row.get("accuracy") - clean_row.get("accuracy")
                    if pd.notna(row.get("accuracy")) and pd.notna(clean_row.get("accuracy")) else None,
                "clean_avg_total_tokens": clean_row.get("avg_total_tokens"),
                "condition_avg_total_tokens": row.get("avg_total_tokens"),
                "delta_tokens_vs_clean": row.get("avg_total_tokens") - clean_row.get("avg_total_tokens")
                    if pd.notna(row.get("avg_total_tokens")) and pd.notna(clean_row.get("avg_total_tokens")) else None,
                "clean_avg_latency_seconds": clean_row.get("avg_latency_seconds"),
                "condition_avg_latency_seconds": row.get("avg_latency_seconds"),
                "delta_latency_vs_clean": row.get("avg_latency_seconds") - clean_row.get("avg_latency_seconds")
                    if pd.notna(row.get("avg_latency_seconds")) and pd.notna(clean_row.get("avg_latency_seconds")) else None,
            })

    return pd.DataFrame(rows)


def build_question_matrix(base: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    # Baseline S0 metadata.
    s0_eval = read_eval_file(base, "baseline", "s0")
    s0_norm = normalize_eval(s0_eval, "s0")
    merged = s0_norm

    for cond in CONDITIONS:
        for sys in SYSTEMS:
            df = read_eval_file(base, cond, sys)
            prefix = f"{cond}_{sys}"
            norm = normalize_eval(df, prefix)

            # Avoid duplicating shared metadata after first merge.
            drop_meta = [c for c in ["original_question", "question", "A", "B", "C", "D", "dataset", "gold_answer"] if c in norm.columns]
            norm = norm.drop(columns=drop_meta, errors="ignore")

            merged = merged.merge(norm, on="id", how="outer")

    assert merged is not None

    # Per-system condition transitions.
    for sys in SYSTEMS:
        clean_col = f"clean_{sys}_correct"
        if clean_col not in merged.columns:
            continue
        for cond in ["noisy", "adversarial"]:
            cond_col = f"{cond}_{sys}_correct"
            if cond_col not in merged.columns:
                continue
            merged[f"{sys}_{cond}_regressed_vs_clean"] = merged[clean_col].fillna(False) & (~merged[cond_col].fillna(False))
            merged[f"{sys}_{cond}_recovered_vs_clean"] = (~merged[clean_col].fillna(False)) & merged[cond_col].fillna(False)
            merged[f"{sys}_{cond}_same_correctness_as_clean"] = merged[clean_col].fillna(False) == merged[cond_col].fillna(False)

    # Cross-system cases by condition.
    for cond in CONDITIONS:
        cols = [f"{cond}_{sys}_correct" for sys in SYSTEMS]
        if all(c in merged.columns for c in cols):
            merged[f"{cond}_all_rag_correct"] = merged[cols].fillna(False).all(axis=1)
            merged[f"{cond}_all_rag_wrong"] = (~merged[cols].fillna(False)).all(axis=1)
            merged[f"{cond}_only_s1_correct"] = merged[f"{cond}_s1_correct"].fillna(False) & (~merged[f"{cond}_s2_correct"].fillna(False)) & (~merged[f"{cond}_s3_mc_correct"].fillna(False))
            merged[f"{cond}_only_s2_correct"] = (~merged[f"{cond}_s1_correct"].fillna(False)) & merged[f"{cond}_s2_correct"].fillna(False) & (~merged[f"{cond}_s3_mc_correct"].fillna(False))
            merged[f"{cond}_only_s3_mc_correct"] = (~merged[f"{cond}_s1_correct"].fillna(False)) & (~merged[f"{cond}_s2_correct"].fillna(False)) & merged[f"{cond}_s3_mc_correct"].fillna(False)
            merged[f"{cond}_s2_beats_s1"] = (~merged[f"{cond}_s1_correct"].fillna(False)) & merged[f"{cond}_s2_correct"].fillna(False)
            merged[f"{cond}_s1_beats_s2"] = merged[f"{cond}_s1_correct"].fillna(False) & (~merged[f"{cond}_s2_correct"].fillna(False))
            merged[f"{cond}_s3_beats_s1_s2"] = (~merged[f"{cond}_s1_correct"].fillna(False)) & (~merged[f"{cond}_s2_correct"].fillna(False)) & merged[f"{cond}_s3_mc_correct"].fillna(False)

    return merged


def build_patterns_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []

    bool_cols = [c for c in matrix.columns if matrix[c].dtype == bool or c.endswith((
        "_regressed_vs_clean",
        "_recovered_vs_clean",
        "_all_rag_correct",
        "_all_rag_wrong",
        "_only_s1_correct",
        "_only_s2_correct",
        "_only_s3_mc_correct",
        "_s2_beats_s1",
        "_s1_beats_s2",
        "_s3_beats_s1_s2",
    ))]

    for col in bool_cols:
        try:
            vals = matrix[col].fillna(False).astype(bool)
        except Exception:
            continue
        rows.append({
            "pattern": col,
            "count": int(vals.sum()),
            "rate": float(vals.mean()) if len(vals) else None,
        })

    out = pd.DataFrame(rows).sort_values(["count", "pattern"], ascending=[False, True])
    return out


def build_interesting_cases(matrix: pd.DataFrame) -> pd.DataFrame:
    pattern_cols = [c for c in matrix.columns if any(token in c for token in [
        "regressed_vs_clean",
        "recovered_vs_clean",
        "only_s",
        "beats",
        "all_rag_wrong",
    ])]

    if not pattern_cols:
        return matrix.head(0).copy()

    mask = pd.Series(False, index=matrix.index)
    for col in pattern_cols:
        try:
            mask |= matrix[col].fillna(False).astype(bool)
        except Exception:
            pass

    out = matrix[mask].copy()

    # Put compact useful columns first.
    first_cols = [
        "id", "dataset", "original_question", "question", "A", "B", "C", "D", "gold_answer",
        "s0_correct", "s0_answer",
    ]
    correctness_cols = [c for c in out.columns if c.endswith("_correct") or c.endswith("_answer")]
    pattern_cols_present = [c for c in pattern_cols if c in out.columns]

    ordered = []
    for c in first_cols + correctness_cols + pattern_cols_present:
        if c in out.columns and c not in ordered:
            ordered.append(c)
    ordered += [c for c in out.columns if c not in ordered]

    return out[ordered]


def write_report(
    *,
    out_path: Path,
    base: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    patterns: pd.DataFrame,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("MuSiQue robustness deep analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Base dir: {base}\n\n")

        f.write("System summary\n")
        f.write("-" * 80 + "\n")
        show_cols = [
            "condition", "system", "n", "accuracy", "valid_format_rate", "run_error_rate",
            "avg_total_tokens", "avg_latency_seconds",
        ]
        route_cols = [c for c in summary.columns if c.startswith("route_rate_") or c.endswith("retrieval_used_rate") or c.endswith("retrieval_triggered_rate") or c.endswith("active_retrieval_triggered_rate")]
        show_cols += route_cols
        show_cols = [c for c in show_cols if c in summary.columns]
        f.write(summary[show_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Condition deltas vs clean\n")
        f.write("-" * 80 + "\n")
        if deltas.empty:
            f.write("No deltas available.\n")
        else:
            f.write(deltas.to_string(index=False))
        f.write("\n\n")

        f.write("Top robustness/question patterns\n")
        f.write("-" * 80 + "\n")
        if patterns.empty:
            f.write("No patterns available.\n")
        else:
            f.write(patterns.head(80).to_string(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    base = args.base_dir
    out_dir = args.output_dir or base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_system_summary(base)
    deltas = build_condition_deltas(summary)
    matrix = build_question_matrix(base)
    patterns = build_patterns_summary(matrix)
    interesting = build_interesting_cases(matrix)

    summary_path = out_dir / "robustness_deep_system_summary.csv"
    deltas_path = out_dir / "robustness_deep_condition_deltas.csv"
    matrix_path = out_dir / "robustness_deep_question_matrix.csv"
    patterns_path = out_dir / "robustness_deep_patterns_summary.csv"
    interesting_path = out_dir / "robustness_deep_interesting_cases.csv"
    report_path = out_dir / "robustness_deep_report.txt"

    summary.to_csv(summary_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    patterns.to_csv(patterns_path, index=False)
    interesting.to_csv(interesting_path, index=False)

    write_report(
        out_path=report_path,
        base=base,
        summary=summary,
        deltas=deltas,
        patterns=patterns,
    )

    print("Deep robustness analysis written")
    print("=" * 80)
    print(summary_path)
    print(deltas_path)
    print(matrix_path)
    print(patterns_path)
    print(interesting_path)
    print(report_path)
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

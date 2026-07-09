#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 5B — Reconstruir resumen HotpotQA-MC-500 con accuracy
#
# Este paso NO llama al LLM.
#
# Objetivo:
#   Leer los *_evaluated.csv y *_summary.json ya generados para
#   S0/S1/S2/S3 y crear una tabla completa con accuracy, tokens,
#   latencia, valid_format y run_error.
#
# Salidas:
#   outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini/hotpotqa_mc500_s0_s3_summary_fixed.csv
#   outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini/hotpotqa_mc500_s0_s3_summary_fixed.md
#   outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini/hotpotqa_mc500_columns_audit.txt
# ============================================================

ROOT="${ROOT:-outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini}"
DATASET="${DATASET:-hotpotqa_mc_500}"
MODEL="${MODEL:-gpt-5-mini}"

echo "== 0. Configuración =="
echo "ROOT=$ROOT"
echo "DATASET=$DATASET"
echo "MODEL=$MODEL"

echo
echo "== 1. Precheck de outputs =="
for s in s0 s1 s2 s3; do
  test -f "$ROOT/${s}_evaluated.csv"
  test -f "$ROOT/${s}_summary.json"
  echo "OK: $s evaluated + summary"
done

echo
echo "== 2. Reconstruir tabla completa =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini")
dataset = "hotpotqa_mc_500"
model = "gpt-5-mini"

def as_num(series):
    return pd.to_numeric(series, errors="coerce")

def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

def boolish_mean(series):
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "yes": 1, "no": 0,
            "correct": 1, "incorrect": 0,
        })
        fallback = pd.to_numeric(series, errors="coerce")
        s = mapped.fillna(fallback)
    else:
        s = pd.to_numeric(series, errors="coerce")
    return float(s.mean())

def normalize_choice(x):
    if pd.isna(x):
        return ""
    t = str(x).strip().upper()
    if len(t) >= 1 and t[0] in {"A", "B", "C", "D"}:
        return t[0]
    return t

rows = []
audit_lines = []

for system in ["s0", "s1", "s2", "s3"]:
    evaluated_path = root / f"{system}_evaluated.csv"
    parsed_path = root / f"{system}_parsed.csv"
    raw_path = root / f"{system}_raw.csv"
    summary_path = root / f"{system}_summary.json"

    df = pd.read_csv(evaluated_path)
    summary = {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        audit_lines.append(f"[{system}] WARNING: no se pudo leer summary JSON: {e}")

    audit_lines.append("")
    audit_lines.append(f"===== {system} =====")
    audit_lines.append(f"evaluated_path={evaluated_path}")
    audit_lines.append(f"shape={df.shape}")
    audit_lines.append("columns=" + ", ".join(df.columns.astype(str).tolist()))
    audit_lines.append("summary_keys=" + ", ".join(sorted(summary.keys())))

    row = {
        "dataset": dataset,
        "model": model,
        "system": system,
        "n": len(df),
        "raw_exists": raw_path.exists(),
        "parsed_exists": parsed_path.exists(),
        "evaluated_exists": evaluated_path.exists(),
        "summary_exists": summary_path.exists(),
    }

    acc_source = None

    for key in ["accuracy", "correct_rate", "accuracy_decided", "correct_rate_decided", "auto_correct_rate", "mc_accuracy"]:
        if key in summary:
            try:
                row["accuracy"] = float(summary[key])
                acc_source = f"summary.{key}"
                break
            except Exception:
                pass

    if "accuracy" not in row:
        col = first_existing(df, [
            "is_correct", "correct", "is_answer_correct", "answer_correct",
            "auto_correct", "exact_match", "mc_correct", "is_correct_auto", "correct_auto"
        ])
        if col is not None:
            row["accuracy"] = boolish_mean(df[col])
            acc_source = f"evaluated.{col}"

    if "accuracy" not in row:
        gold_col = first_existing(df, ["gold_answer", "expected_answer", "correct_answer", "label", "gold"])
        pred_col = first_existing(df, [
            "parsed_answer", "predicted_answer", "prediction", "model_answer",
            "selected_answer", "final_answer", "answer", "choice",
            "auto_decision", "decision"
        ])
        if gold_col and pred_col:
            gold = df[gold_col].map(normalize_choice)
            pred = df[pred_col].map(normalize_choice)
            valid = gold.isin(["A", "B", "C", "D"]) & pred.isin(["A", "B", "C", "D"])
            if valid.any():
                row["accuracy"] = float((gold[valid] == pred[valid]).mean())
                row["accuracy_reconstructed_n"] = int(valid.sum())
                acc_source = f"compare evaluated.{pred_col} vs evaluated.{gold_col}"

    row["accuracy_source"] = acc_source or "NOT_FOUND"

    for out_key in ["valid_format_rate", "run_error_rate", "auto_decision_rate", "manual_review_rate", "abstention_rate"]:
        if out_key in summary:
            try:
                row[out_key] = float(summary[out_key])
            except Exception:
                pass

    if "valid_format_rate" not in row:
        col = first_existing(df, ["valid_format"])
        if col:
            row["valid_format_rate"] = boolish_mean(df[col])

    if "run_error_rate" not in row:
        col = first_existing(df, ["run_error"])
        if col:
            row["run_error_rate"] = boolish_mean(df[col])

    if "avg_total_tokens" not in row:
        for key in ["avg_total_tokens", "total_tokens_avg", "mean_total_tokens"]:
            if key in summary:
                try:
                    row["avg_total_tokens"] = float(summary[key])
                    break
                except Exception:
                    pass
    if "avg_total_tokens" not in row:
        col = first_existing(df, ["total_tokens"])
        if col:
            row["avg_total_tokens"] = float(as_num(df[col]).mean())

    if "avg_latency_seconds" not in row:
        for key in ["avg_latency_seconds", "latency_seconds_avg", "mean_latency_seconds"]:
            if key in summary:
                try:
                    row["avg_latency_seconds"] = float(summary[key])
                    break
                except Exception:
                    pass
    if "avg_latency_seconds" not in row:
        col = first_existing(df, ["latency_seconds"])
        if col:
            row["avg_latency_seconds"] = float(as_num(df[col]).mean())

    for out_key, candidates in [
        ("retrieve_rate", ["retrieved", "used_retrieval", "s2_retrieved"]),
        ("active_retrieval_rate", ["active_retrieval", "s3_active_retrieval"]),
    ]:
        if out_key in summary:
            try:
                row[out_key] = float(summary[out_key])
                continue
            except Exception:
                pass
        col = first_existing(df, candidates)
        if col:
            row[out_key] = boolish_mean(df[col])

    rows.append(row)

summary_df = pd.DataFrame(rows)

if (summary_df["system"] == "s0").any() and "accuracy" in summary_df.columns:
    s0_acc = summary_df.loc[summary_df["system"] == "s0", "accuracy"].iloc[0]
    if pd.notna(s0_acc):
        summary_df["delta_vs_s0"] = summary_df["accuracy"] - float(s0_acc)

if (summary_df["system"] == "s0").any() and "avg_total_tokens" in summary_df.columns:
    s0_tok = summary_df.loc[summary_df["system"] == "s0", "avg_total_tokens"].iloc[0]
    if pd.notna(s0_tok) and float(s0_tok) != 0:
        summary_df["token_ratio_vs_s0"] = summary_df["avg_total_tokens"] / float(s0_tok)

display_cols = [c for c in [
    "dataset", "model", "system", "n", "accuracy", "delta_vs_s0",
    "valid_format_rate", "run_error_rate", "avg_total_tokens",
    "token_ratio_vs_s0", "avg_latency_seconds", "retrieve_rate",
    "active_retrieval_rate", "accuracy_source"
] if c in summary_df.columns]

out_csv = root / "hotpotqa_mc500_s0_s3_summary_fixed.csv"
out_md = root / "hotpotqa_mc500_s0_s3_summary_fixed.md"
audit_txt = root / "hotpotqa_mc500_columns_audit.txt"

summary_df.to_csv(out_csv, index=False)
try:
    md = summary_df[display_cols].to_markdown(index=False)
except Exception:
    md = summary_df[display_cols].to_csv(index=False)
out_md.write_text(md, encoding="utf-8")
audit_txt.write_text("\n".join(audit_lines), encoding="utf-8")

print(summary_df[display_cols].to_string(index=False))
print()
print("summary_fixed_csv:", out_csv)
print("summary_fixed_md:", out_md)
print("columns_audit:", audit_txt)

bad = []
if "accuracy" not in summary_df.columns or summary_df["accuracy"].isna().any():
    bad.append("falta accuracy en al menos un sistema")
for _, r in summary_df.iterrows():
    if int(r.get("n", 0)) != 500:
        bad.append(f"{r['system']} n={r.get('n')}")
    if float(r.get("valid_format_rate", 1.0)) < 0.95:
        bad.append(f"{r['system']} valid_format_rate={r.get('valid_format_rate')}")
    if float(r.get("run_error_rate", 0.0)) > 0.05:
        bad.append(f"{r['system']} run_error_rate={r.get('run_error_rate')}")

if bad:
    print()
    print("WARNING:", "; ".join(bad))
    print("Revisar columns_audit si falta accuracy.")
else:
    print()
    print("OK: resumen completo con accuracy reconstruido.")
PY

echo
echo "== 3. Estado Git resumido =="
git status --short

echo
echo "LISTO: resumen HotpotQA-MC-500 corregido."

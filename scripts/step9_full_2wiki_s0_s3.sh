#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 9 — Corrida completa 2Wiki-MC-500 con S0/S1/S2/S3
#
# Este paso SÍ llama al LLM.
#
# Por defecto:
#   MODEL=gpt-5-mini
#
# Salidas:
#   outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini/
#
# La corrida usa --resume, así que si se corta puede relanzarse.
# Al final reconstruye una tabla completa con accuracy.
# ============================================================

MODEL="${MODEL:-gpt-5-mini}"

model_tag () {
  echo "$1" | tr '.-' '__'
}

TAG="$(model_tag "$MODEL")"
OUT="outputs/eval_mc/2wiki_mc_rag_500/${TAG}"
QUESTIONS="data/eval_mc/2wiki_mc_rag_500/questions.csv"
INDEX_DIR="indexes/eval_mc/2wiki_mc_rag_500"

echo "== 0. Configuración =="
echo "MODEL=$MODEL"
echo "TAG=$TAG"
echo "QUESTIONS=$QUESTIONS"
echo "INDEX_DIR=$INDEX_DIR"
echo "OUT=$OUT"

echo
echo "== 1. Precheck de archivos =="
test -f "$QUESTIONS"
test -d "$INDEX_DIR"
test -f "$INDEX_DIR/chunks.csv"
test -f "$INDEX_DIR/embeddings.npy"
test -f "$INDEX_DIR/metadata.json"
test -f modelos/s0/run_s0_direct.py
test -f modelos/s0/parse_s0_outputs.py
test -f modelos/s0/evaluate_s0.py
test -f modelos/s1/run_s1_mc_rag.py
test -f modelos/s2/run_s2_mc_real_adaptive.py
test -f modelos/s3/run_s3_mc_flare_like.py
echo "OK: archivos necesarios encontrados."

mkdir -p "$OUT"

echo
echo "== 2. Preflight del modelo =="
python - "$MODEL" <<'PY'
import sys
from direct_llm import ask_direct_llm_with_metadata

model = sys.argv[1]
result = ask_direct_llm_with_metadata(
    'Return exactly this JSON: {"answer":"A","confidence":1.0}',
    model=model,
    max_retries=0,
)
print(f"MODEL_OK {model} total_tokens={result.get('total_tokens')}")
PY

run_eval_mc () {
  local RAW="$1"
  local PREFIX="$2"

  python modelos/s0/parse_s0_outputs.py \
    --input-path "$RAW" \
    --output-path "${PREFIX}_parsed.csv"

  python modelos/s0/evaluate_s0.py \
    --input-path "${PREFIX}_parsed.csv" \
    --output-path "${PREFIX}_evaluated.csv" \
    --summary-path "${PREFIX}_summary.json" \
    --group-summary-path "${PREFIX}_group_summary.csv"
}

echo
echo "== 3. S0 direct completo =="
python modelos/s0/run_s0_direct.py \
  --input-path "$QUESTIONS" \
  --output-path "$OUT/s0_raw.csv" \
  --model "$MODEL" \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s0_raw.csv" "$OUT/s0"

echo
echo "== 4. S1 classic RAG top-5 completo =="
python modelos/s1/run_s1_mc_rag.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s1_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s1_raw.csv" "$OUT/s1"

echo
echo "== 5. S2 real adaptive completo =="
python modelos/s2/run_s2_mc_real_adaptive.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s2_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --threshold 0.45 \
  --min-gap 0.05 \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s2_raw.csv" "$OUT/s2"

echo
echo "== 6. S3-MC FLARE-like completo =="
python modelos/s3/run_s3_mc_flare_like.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s3_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --confidence-threshold 0.78 \
  --score-threshold 0.45 \
  --min-gap 0.05 \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s3_raw.csv" "$OUT/s3"

echo
echo "== 7. Resumen completo 2Wiki-MC-500 con accuracy =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini")
dataset = "2wiki_mc_500"
model = "gpt-5-mini"

def normalize_choice(x):
    if pd.isna(x):
        return ""
    t = str(x).strip().upper()
    if t and t[0] in {"A", "B", "C", "D"}:
        return t[0]
    return t

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

def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

rows = []
audit_lines = []

for system in ["s0", "s1", "s2", "s3"]:
    evaluated_path = root / f"{system}_evaluated.csv"
    parsed_path = root / f"{system}_parsed.csv"
    raw_path = root / f"{system}_raw.csv"
    summary_path = root / f"{system}_summary.json"

    row = {
        "dataset": dataset,
        "model": model,
        "system": system,
        "raw_exists": raw_path.exists(),
        "parsed_exists": parsed_path.exists(),
        "evaluated_exists": evaluated_path.exists(),
        "summary_exists": summary_path.exists(),
    }

    if not evaluated_path.exists():
        rows.append(row)
        continue

    df = pd.read_csv(evaluated_path)
    row["n"] = len(df)

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            row["summary_error"] = str(e)

    audit_lines.append("")
    audit_lines.append(f"===== {system} =====")
    audit_lines.append(f"evaluated_path={evaluated_path}")
    audit_lines.append(f"shape={df.shape}")
    audit_lines.append("columns=" + ", ".join(df.columns.astype(str).tolist()))
    audit_lines.append("summary_keys=" + ", ".join(sorted(summary.keys())))

    # Accuracy: preferir comparación explícita parsed_answer vs gold_answer.
    if "parsed_answer" in df.columns and "gold_answer" in df.columns:
        pred = df["parsed_answer"].map(normalize_choice)
        gold = df["gold_answer"].map(normalize_choice)
        valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
        row["accuracy"] = float((pred[valid] == gold[valid]).mean())
        row["accuracy_n"] = int(valid.sum())
        row["accuracy_source"] = "compare evaluated.parsed_answer vs evaluated.gold_answer"
    else:
        for key in ["accuracy", "correct_rate", "accuracy_decided", "correct_rate_decided"]:
            if key in summary:
                try:
                    row["accuracy"] = float(summary[key])
                    row["accuracy_source"] = f"summary.{key}"
                    break
                except Exception:
                    pass

    if "valid_format" in df.columns:
        row["valid_format_rate"] = boolish_mean(df["valid_format"])
    elif "valid_format_rate" in summary:
        row["valid_format_rate"] = float(summary["valid_format_rate"])

    if "run_error" in df.columns:
        row["run_error_rate"] = boolish_mean(df["run_error"])
    elif "run_error_rate" in summary:
        row["run_error_rate"] = float(summary["run_error_rate"])

    if "total_tokens" in df.columns:
        row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())
    elif "avg_total_tokens" in summary:
        row["avg_total_tokens"] = float(summary["avg_total_tokens"])

    if "latency_seconds" in df.columns:
        row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())
    elif "avg_latency_seconds" in summary:
        row["avg_latency_seconds"] = float(summary["avg_latency_seconds"])

    # Tasas específicas si existen.
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

if "accuracy" in summary_df.columns and (summary_df["system"] == "s0").any():
    s0_acc = summary_df.loc[summary_df["system"] == "s0", "accuracy"].iloc[0]
    if pd.notna(s0_acc):
        summary_df["delta_vs_s0"] = summary_df["accuracy"] - float(s0_acc)

if "avg_total_tokens" in summary_df.columns and (summary_df["system"] == "s0").any():
    s0_tok = summary_df.loc[summary_df["system"] == "s0", "avg_total_tokens"].iloc[0]
    if pd.notna(s0_tok) and float(s0_tok) != 0:
        summary_df["token_ratio_vs_s0"] = summary_df["avg_total_tokens"] / float(s0_tok)

display_cols = [c for c in [
    "dataset",
    "model",
    "system",
    "n",
    "accuracy",
    "delta_vs_s0",
    "valid_format_rate",
    "run_error_rate",
    "avg_total_tokens",
    "token_ratio_vs_s0",
    "avg_latency_seconds",
    "retrieve_rate",
    "active_retrieval_rate",
    "accuracy_source",
] if c in summary_df.columns]

out_csv = root / "2wiki_mc500_s0_s3_summary.csv"
out_md = root / "2wiki_mc500_s0_s3_summary.md"
audit_txt = root / "2wiki_mc500_columns_audit.txt"

summary_df.to_csv(out_csv, index=False)
try:
    out_md.write_text(summary_df[display_cols].to_markdown(index=False), encoding="utf-8")
except Exception:
    out_md.write_text(summary_df[display_cols].to_csv(index=False), encoding="utf-8")
audit_txt.write_text("\n".join(audit_lines), encoding="utf-8")

print(summary_df[display_cols].to_string(index=False))
print()
print("summary_csv:", out_csv)
print("summary_md:", out_md)
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
else:
    print()
    print("OK: corrida completa consistente.")
PY

echo
echo "== 8. Estado Git resumido =="
git status --short

echo
echo "LISTO: corrida completa 2Wiki-MC-500 S0/S1/S2/S3 terminada."

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 8 — Smoke test 2Wiki-MC-500 con S0/S1/S2/S3
#
# Este paso SÍ llama al LLM.
#
# Por defecto:
#   MODEL=gpt-5-mini
#   LIMIT=20
#
# Salidas:
#   outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini_smoke20/
# ============================================================

MODEL="${MODEL:-gpt-5-mini}"
LIMIT="${LIMIT:-20}"

model_tag () {
  echo "$1" | tr '.-' '__'
}

TAG="$(model_tag "$MODEL")"
OUT="outputs/eval_mc/2wiki_mc_rag_500/${TAG}_smoke${LIMIT}"
QUESTIONS="data/eval_mc/2wiki_mc_rag_500/questions.csv"
INDEX_DIR="indexes/eval_mc/2wiki_mc_rag_500"

echo "== 0. Configuración =="
echo "MODEL=$MODEL"
echo "TAG=$TAG"
echo "LIMIT=$LIMIT"
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
test -f run_s0_direct.py
test -f parse_s0_outputs.py
test -f evaluate_s0.py
test -f evaluation/run_s1_mc_rag.py
test -f evaluation/run_s2_mc_real_adaptive.py
test -f evaluation/run_s3_mc_flare_like.py
echo "OK: archivos necesarios encontrados."

mkdir -p "$OUT"

echo
echo "== 2. Preflight del modelo =="
python - "$MODEL" <<'PY'
import sys
from direct_llm import ask_direct_llm_with_metadata

model = sys.argv[1]
try:
    result = ask_direct_llm_with_metadata(
        'Return exactly this JSON: {"answer":"A","confidence":1.0}',
        model=model,
        max_retries=0,
    )
    print(f"MODEL_OK {model} total_tokens={result.get('total_tokens')}")
except Exception as e:
    print(f"MODEL_NOT_AVAILABLE {model}: {e}")
    raise
PY

run_eval_mc () {
  local RAW="$1"
  local PREFIX="$2"

  python parse_s0_outputs.py \
    --input-path "$RAW" \
    --output-path "${PREFIX}_parsed.csv"

  python evaluate_s0.py \
    --input-path "${PREFIX}_parsed.csv" \
    --output-path "${PREFIX}_evaluated.csv" \
    --summary-path "${PREFIX}_summary.json" \
    --group-summary-path "${PREFIX}_group_summary.csv"
}

echo
echo "== 3. S0 direct =="
python run_s0_direct.py \
  --input-path "$QUESTIONS" \
  --output-path "$OUT/s0_raw.csv" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s0_raw.csv" "$OUT/s0"

echo
echo "== 4. S1 classic RAG top-5 =="
python evaluation/run_s1_mc_rag.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s1_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --limit "$LIMIT" \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s1_raw.csv" "$OUT/s1"

echo
echo "== 5. S2 real adaptive =="
python evaluation/run_s2_mc_real_adaptive.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s2_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --threshold 0.45 \
  --min-gap 0.05 \
  --limit "$LIMIT" \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s2_raw.csv" "$OUT/s2"

echo
echo "== 6. S3-MC FLARE-like =="
python evaluation/run_s3_mc_flare_like.py \
  --questions-path "$QUESTIONS" \
  --index-dir "$INDEX_DIR" \
  --output-path "$OUT/s3_raw.csv" \
  --model "$MODEL" \
  --top-k 5 \
  --confidence-threshold 0.78 \
  --score-threshold 0.45 \
  --min-gap 0.05 \
  --limit "$LIMIT" \
  --save-every 1 \
  --resume

run_eval_mc "$OUT/s3_raw.csv" "$OUT/s3"

echo
echo "== 7. Resumen del smoke con accuracy =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini_smoke20")
if not root.exists():
    candidates = sorted(Path("outputs/eval_mc/2wiki_mc_rag_500").glob("*_smoke*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("ERROR: no encontré carpeta smoke.")
    root = candidates[-1]

print("smoke_dir:", root)

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
            "true": 1, "false": 0, "1": 1, "0": 0,
            "yes": 1, "no": 0,
        })
        fallback = pd.to_numeric(series, errors="coerce")
        s = mapped.fillna(fallback)
    else:
        s = pd.to_numeric(series, errors="coerce")
    return float(s.mean())

rows = []

for system in ["s0", "s1", "s2", "s3"]:
    summary_path = root / f"{system}_summary.json"
    evaluated_path = root / f"{system}_evaluated.csv"
    raw_path = root / f"{system}_raw.csv"

    row = {
        "system": system,
        "raw_exists": raw_path.exists(),
        "evaluated_exists": evaluated_path.exists(),
        "summary_exists": summary_path.exists(),
    }

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            row["summary_error"] = str(e)

    if evaluated_path.exists():
        df = pd.read_csv(evaluated_path)
        row["n"] = len(df)

        if "parsed_answer" in df.columns and "gold_answer" in df.columns:
            pred = df["parsed_answer"].map(normalize_choice)
            gold = df["gold_answer"].map(normalize_choice)
            valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
            row["accuracy"] = float((pred[valid] == gold[valid]).mean())
            row["accuracy_n"] = int(valid.sum())

        if "valid_format" in df.columns:
            row["valid_format_rate"] = boolish_mean(df["valid_format"])

        if "run_error" in df.columns:
            row["run_error_rate"] = boolish_mean(df["run_error"])

        if "total_tokens" in df.columns:
            row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())

        if "latency_seconds" in df.columns:
            row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())

    for key in ["valid_format_rate", "run_error_rate", "avg_total_tokens", "avg_latency_seconds", "active_retrieval_rate", "retrieve_rate"]:
        if key not in row and key in summary:
            try:
                row[key] = float(summary[key])
            except Exception:
                pass

    rows.append(row)

summary_df = pd.DataFrame(rows)

if "accuracy" in summary_df.columns and (summary_df["system"] == "s0").any():
    s0_acc = float(summary_df.loc[summary_df["system"] == "s0", "accuracy"].iloc[0])
    summary_df["delta_vs_s0"] = summary_df["accuracy"] - s0_acc

if "avg_total_tokens" in summary_df.columns and (summary_df["system"] == "s0").any():
    s0_tok = float(summary_df.loc[summary_df["system"] == "s0", "avg_total_tokens"].iloc[0])
    if s0_tok:
        summary_df["token_ratio_vs_s0"] = summary_df["avg_total_tokens"] / s0_tok

display_cols = [c for c in [
    "system",
    "n",
    "accuracy",
    "delta_vs_s0",
    "valid_format_rate",
    "run_error_rate",
    "avg_total_tokens",
    "token_ratio_vs_s0",
    "avg_latency_seconds",
    "active_retrieval_rate",
    "retrieve_rate",
    "summary_exists",
    "evaluated_exists",
    "raw_exists",
] if c in summary_df.columns]

print(summary_df[display_cols].to_string(index=False))

summary_csv = root / "smoke_summary_table.csv"
summary_md = root / "smoke_summary_table.md"
summary_df.to_csv(summary_csv, index=False)
try:
    summary_md.write_text(summary_df[display_cols].to_markdown(index=False), encoding="utf-8")
except Exception:
    summary_md.write_text(summary_df[display_cols].to_csv(index=False), encoding="utf-8")

print("smoke_summary_table_csv:", summary_csv)
print("smoke_summary_table_md:", summary_md)

bad = []
for _, r in summary_df.iterrows():
    if int(r.get("n", 0)) != 20:
        bad.append(f"{r['system']} n={r.get('n')}")
    if float(r.get("valid_format_rate", 1.0)) < 0.95:
        bad.append(f"{r['system']} valid_format_rate={r.get('valid_format_rate')}")
    if float(r.get("run_error_rate", 0.0)) > 0.05:
        bad.append(f"{r['system']} run_error_rate={r.get('run_error_rate')}")

if bad:
    print("WARNING: revisar smoke:", "; ".join(bad))
else:
    print("OK: smoke básico consistente.")
PY

echo
echo "== 8. Estado Git resumido =="
git status --short

echo
echo "LISTO: smoke 2Wiki-MC-500 S0/S1/S2/S3 terminado."

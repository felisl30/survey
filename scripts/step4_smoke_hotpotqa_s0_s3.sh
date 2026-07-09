#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 4 — Smoke test HotpotQA-MC-500 con S0/S1/S2/S3
#
# Este paso SÍ llama al LLM.
#
# Por defecto:
#   MODEL=gpt-5-mini
#   LIMIT=20
#
# Salidas:
#   outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini_smoke20/
# ============================================================

MODEL="${MODEL:-gpt-5-mini}"
LIMIT="${LIMIT:-20}"

model_tag () {
  echo "$1" | tr '.-' '__'
}

TAG="$(model_tag "$MODEL")"
OUT="outputs/eval_mc/hotpotqa_mc_rag_500/${TAG}_smoke${LIMIT}"
QUESTIONS="data/eval_mc/hotpotqa_mc_rag_500/questions.csv"
INDEX_DIR="indexes/eval_mc/hotpotqa_mc_rag_500"

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
echo "== 7. Resumen del smoke =="
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

out = Path("outputs/eval_mc/hotpotqa_mc_rag_500")
smoke_dirs = sorted(out.glob("*_smoke*"), key=lambda p: p.stat().st_mtime)
if not smoke_dirs:
    raise SystemExit("ERROR: no encontré carpeta smoke.")
root = smoke_dirs[-1]
print("smoke_dir:", root)

rows = []
for system in ["s0", "s1", "s2", "s3"]:
    summary_path = root / f"{system}_summary.json"
    evaluated_path = root / f"{system}_evaluated.csv"
    raw_path = root / f"{system}_raw.csv"

    row = {"system": system, "summary_exists": summary_path.exists(), "evaluated_exists": evaluated_path.exists(), "raw_exists": raw_path.exists()}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for k, v in summary.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
        except Exception as e:
            row["summary_error"] = str(e)

    if evaluated_path.exists():
        df = pd.read_csv(evaluated_path)
        row["evaluated_rows"] = len(df)
        if "is_correct" in df.columns:
            row["accuracy_from_evaluated"] = float(df["is_correct"].mean())
        if "valid_format" in df.columns:
            row["valid_format_rate_from_evaluated"] = float(df["valid_format"].mean())
        if "run_error" in df.columns:
            row["run_error_rate_from_evaluated"] = float(df["run_error"].mean())
        if "total_tokens" in df.columns:
            row["avg_total_tokens_from_evaluated"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())

    rows.append(row)

summary_df = pd.DataFrame(rows)
display_cols = [c for c in [
    "system",
    "evaluated_rows",
    "accuracy",
    "accuracy_from_evaluated",
    "valid_format_rate",
    "valid_format_rate_from_evaluated",
    "run_error_rate",
    "run_error_rate_from_evaluated",
    "avg_total_tokens",
    "avg_total_tokens_from_evaluated",
    "summary_exists",
    "evaluated_exists",
    "raw_exists",
] if c in summary_df.columns]

print(summary_df[display_cols].to_string(index=False))
summary_df.to_csv(root / "smoke_summary_table.csv", index=False)
print("smoke_summary_table:", root / "smoke_summary_table.csv")

bad = []
for _, r in summary_df.iterrows():
    if r.get("evaluated_rows", 0) != 20:
        bad.append(f"{r['system']} rows={r.get('evaluated_rows')}")
    vf = r.get("valid_format_rate", r.get("valid_format_rate_from_evaluated", 1.0))
    if pd.notna(vf) and float(vf) < 0.95:
        bad.append(f"{r['system']} valid_format_rate={vf}")
    re = r.get("run_error_rate", r.get("run_error_rate_from_evaluated", 0.0))
    if pd.notna(re) and float(re) > 0.05:
        bad.append(f"{r['system']} run_error_rate={re}")

if bad:
    print("WARNING: revisar smoke:", "; ".join(bad))
else:
    print("OK: smoke básico consistente.")
PY

echo
echo "== 8. Estado Git resumido =="
git status --short

echo
echo "LISTO: smoke HotpotQA-MC-500 S0/S1/S2/S3 terminado."

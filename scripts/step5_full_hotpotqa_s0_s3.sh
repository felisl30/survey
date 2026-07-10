#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 5 — Corrida completa HotpotQA-MC-500 con S0/S1/S2/S3
#
# Este paso SÍ llama al LLM.
#
# Por defecto:
#   MODEL=gpt-5-mini
#
# Salidas:
#   outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini/
#
# La corrida usa --resume, así que si se corta puede relanzarse.
# ============================================================

MODEL="${MODEL:-gpt-5-mini}"

model_tag () {
  echo "$1" | tr '.-' '__'
}

TAG="$(model_tag "$MODEL")"
OUT="outputs/eval_mc/hotpotqa_mc_rag_500/${TAG}"
QUESTIONS="data/eval_mc/hotpotqa_mc_rag_500/questions.csv"
INDEX_DIR="indexes/eval_mc/hotpotqa_mc_rag_500"

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
echo "== 7. Resumen completo HotpotQA-MC-500 =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini")
if not root.exists():
    candidates = sorted(Path("outputs/eval_mc/hotpotqa_mc_rag_500").glob("gpt*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("ERROR: no encontré carpeta de outputs.")
    root = candidates[-1]

print("output_dir:", root)

rows = []
for system in ["s0", "s1", "s2", "s3"]:
    evaluated_path = root / f"{system}_evaluated.csv"
    summary_path = root / f"{system}_summary.json"
    raw_path = root / f"{system}_raw.csv"

    row = {
        "dataset": "hotpotqa_mc_500",
        "model": "gpt-5-mini",
        "system": system,
        "raw_exists": raw_path.exists(),
        "evaluated_exists": evaluated_path.exists(),
        "summary_exists": summary_path.exists(),
    }

    if evaluated_path.exists():
        df = pd.read_csv(evaluated_path)
        row["n"] = len(df)

        for col in ["is_correct", "correct"]:
            if col in df.columns:
                row["accuracy"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                break

        for col in ["valid_format"]:
            if col in df.columns:
                row["valid_format_rate"] = float(pd.to_numeric(df[col], errors="coerce").mean())

        for col in ["run_error"]:
            if col in df.columns:
                row["run_error_rate"] = float(pd.to_numeric(df[col], errors="coerce").mean())

        for col in ["total_tokens", "avg_total_tokens"]:
            if col in df.columns:
                row["avg_total_tokens"] = float(pd.to_numeric(df[col], errors="coerce").mean())
                break

        for col in ["latency_seconds"]:
            if col in df.columns:
                row["avg_latency_seconds"] = float(pd.to_numeric(df[col], errors="coerce").mean())

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for src, dst in [
                ("accuracy", "accuracy"),
                ("correct_rate", "accuracy"),
                ("valid_format_rate", "valid_format_rate"),
                ("run_error_rate", "run_error_rate"),
                ("avg_total_tokens", "avg_total_tokens"),
                ("avg_latency_seconds", "avg_latency_seconds"),
                ("auto_decision_rate", "auto_decision_rate"),
                ("manual_review_rate", "manual_review_rate"),
            ]:
                if src in summary and dst not in row:
                    row[dst] = summary[src]
        except Exception as e:
            row["summary_error"] = str(e)

    rows.append(row)

df = pd.DataFrame(rows)

# Deltas vs S0.
if "accuracy" in df.columns and (df["system"] == "s0").any():
    s0_acc = float(df.loc[df["system"] == "s0", "accuracy"].iloc[0])
    df["delta_vs_s0"] = df["accuracy"] - s0_acc

if "avg_total_tokens" in df.columns and (df["system"] == "s0").any():
    s0_tok = float(df.loc[df["system"] == "s0", "avg_total_tokens"].iloc[0])
    df["token_ratio_vs_s0"] = df["avg_total_tokens"] / s0_tok if s0_tok else None

out_csv = root / "hotpotqa_mc500_s0_s3_summary.csv"
out_md = root / "hotpotqa_mc500_s0_s3_summary.md"
df.to_csv(out_csv, index=False)

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
    "auto_decision_rate",
    "manual_review_rate",
] if c in df.columns]

print(df[display_cols].to_string(index=False))

try:
    out_md.write_text(df[display_cols].to_markdown(index=False), encoding="utf-8")
    print("summary_csv:", out_csv)
    print("summary_md:", out_md)
except Exception as e:
    print("summary_csv:", out_csv)
    print("WARNING: no pude escribir markdown:", e)

bad = []
for _, r in df.iterrows():
    if int(r.get("n", 0)) != 500:
        bad.append(f"{r['system']} n={r.get('n')}")
    if float(r.get("valid_format_rate", 1.0)) < 0.95:
        bad.append(f"{r['system']} valid_format_rate={r.get('valid_format_rate')}")
    if float(r.get("run_error_rate", 0.0)) > 0.05:
        bad.append(f"{r['system']} run_error_rate={r.get('run_error_rate')}")

if bad:
    print("WARNING: revisar corrida:", "; ".join(bad))
else:
    print("OK: corrida completa consistente.")
PY

echo
echo "== 8. Estado Git resumido =="
git status --short

echo
echo "LISTO: corrida completa HotpotQA-MC-500 S0/S1/S2/S3 terminada."

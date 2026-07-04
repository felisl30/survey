#!/usr/bin/env bash
set -euo pipefail

# run_musique_robustness_s0_s3.sh
#
# Runs S0 once and S1/S2/S3-MC over clean/noisy/adversarial robustness conditions.
#
# Usage from repo root:
#   LIMIT=3 bash scripts/run_musique_robustness_s0_s3.sh
#   bash scripts/run_musique_robustness_s0_s3.sh
#
# Optional overrides:
#   MODEL="gpt-5.4-mini" bash scripts/run_musique_robustness_s0_s3.sh
#   BASE_OUT="outputs/eval_mc/robustness_musique" bash scripts/run_musique_robustness_s0_s3.sh

MODEL="${MODEL:-gpt-5.4-mini}"
QUESTIONS="data/eval_mc/robustness_musique/questions.csv"
BASE_OUT="${BASE_OUT:-outputs/eval_mc/robustness_musique}"

CONDITIONS=(clean noisy adversarial)

LIMIT_ARG=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

model_tag () {
  echo "$1" | tr '.-' '__'
}

preflight_model () {
  local MODEL_NAME="$1"
  python - "$MODEL_NAME" <<'PY'
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
    sys.exit(17)
PY
}

run_eval_mc () {
  local IN="$1"
  local PREFIX="$2"

  python parse_s0_outputs.py \
    --input-path "${IN}" \
    --output-path "${PREFIX}_parsed.csv"

  python evaluate_s0.py \
    --input-path "${PREFIX}_parsed.csv" \
    --output-path "${PREFIX}_evaluated.csv" \
    --summary-path "${PREFIX}_summary.json" \
    --group-summary-path "${PREFIX}_group_summary.csv"
}

TAG="$(model_tag "$MODEL")"
OUT="${BASE_OUT}/${TAG}"
mkdir -p "$OUT"

echo
echo "================================================================================"
echo "ROBUSTNESS RUN"
echo "MODEL: $MODEL"
echo "TAG:   $TAG"
echo "OUT:   $OUT"
if [ -n "${LIMIT:-}" ]; then
  echo "LIMIT: $LIMIT"
else
  echo "LIMIT: full"
fi
echo "================================================================================"

if ! preflight_model "$MODEL"; then
  echo "SKIP: $MODEL no está disponible o falló el preflight."
  exit 17
fi

echo
echo "---- S0 direct baseline, run once ----"
python run_s0_direct.py \
  --input-path "$QUESTIONS" \
  --output-path "$OUT/s0_raw.csv" \
  --model "$MODEL" \
  --save-every 1 \
  --resume \
  "${LIMIT_ARG[@]}"

run_eval_mc "$OUT/s0_raw.csv" "$OUT/s0"

for COND in "${CONDITIONS[@]}"; do
  INDEX_DIR="indexes/eval_mc/robustness_musique_${COND}"
  COND_OUT="$OUT/${COND}"
  mkdir -p "$COND_OUT"

  echo
  echo "================================================================================"
  echo "CONDITION: $COND"
  echo "INDEX:     $INDEX_DIR"
  echo "OUT:       $COND_OUT"
  echo "================================================================================"

  echo
  echo "---- S1 classic RAG top-5: $COND ----"
  python evaluation/run_s1_mc_rag.py \
    --questions-path "$QUESTIONS" \
    --index-dir "$INDEX_DIR" \
    --output-path "$COND_OUT/s1_raw.csv" \
    --model "$MODEL" \
    --top-k 5 \
    --save-every 1 \
    --resume \
    "${LIMIT_ARG[@]}"

  run_eval_mc "$COND_OUT/s1_raw.csv" "$COND_OUT/s1"

  echo
  echo "---- S2 real adaptive: $COND ----"
  python evaluation/run_s2_mc_real_adaptive.py \
    --questions-path "$QUESTIONS" \
    --index-dir "$INDEX_DIR" \
    --output-path "$COND_OUT/s2_raw.csv" \
    --model "$MODEL" \
    --top-k 5 \
    --threshold 0.45 \
    --min-gap 0.05 \
    --save-every 1 \
    --resume \
    "${LIMIT_ARG[@]}"

  run_eval_mc "$COND_OUT/s2_raw.csv" "$COND_OUT/s2"

  echo
  echo "---- S3-MC FLARE-like: $COND ----"
  python evaluation/run_s3_mc_flare_like.py \
    --questions-path "$QUESTIONS" \
    --index-dir "$INDEX_DIR" \
    --output-path "$COND_OUT/s3_mc_raw.csv" \
    --model "$MODEL" \
    --top-k 5 \
    --confidence-threshold 0.78 \
    --score-threshold 0.45 \
    --min-gap 0.05 \
    --save-every 1 \
    --resume \
    "${LIMIT_ARG[@]}"

  run_eval_mc "$COND_OUT/s3_mc_raw.csv" "$COND_OUT/s3_mc"

  echo
  echo "DONE CONDITION: $COND"
done

echo
echo "Robustness S0-S3 run finished."
echo "Outputs: $OUT"

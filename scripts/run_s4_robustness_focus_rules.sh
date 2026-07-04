#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="outputs/eval_mc/robustness_musique/gpt_5_4_mini"
INPUT_DIR="${BASE_DIR}/s4/input"
GEN_DIR="${BASE_DIR}/s4/generation"
ANALYSIS_DIR="${BASE_DIR}/s4/analysis"

INPUT_CORE="${INPUT_DIR}/s4_robustness_focus_core5.csv"
INPUT_ADV="${INPUT_DIR}/s4_robustness_focus_core5_adversarial.csv"
INPUT_NOISY="${INPUT_DIR}/s4_robustness_focus_core5_noisy.csv"

OUT_ADV="${GEN_DIR}/fire_like_s4_robustness_focus_core5_adversarial_rules_raw.csv"
OUT_NOISY="${GEN_DIR}/fire_like_s4_robustness_focus_core5_noisy_rules_raw.csv"

mkdir -p "$INPUT_DIR" "$GEN_DIR" "$ANALYSIS_DIR"

echo
echo "================================================================================"
echo "Splitting S4 focus input by condition"
echo "================================================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini")
input_core = base / "s4/input/s4_robustness_focus_core5.csv"
input_dir = base / "s4/input"

df = pd.read_csv(input_core)

required = {"id", "source_condition"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {input_core}: {sorted(missing)}")

for cond in ["adversarial", "noisy"]:
    out = df[df["source_condition"].astype(str) == cond].copy()
    out_path = input_dir / f"s4_robustness_focus_core5_{cond}.csv"
    out.to_csv(out_path, index=False)
    print(f"{cond}: {len(out)} rows -> {out_path}")
PY

echo
echo "================================================================================"
echo "Running S4 rules+index on adversarial focus rows"
echo "================================================================================"

python s4_model_code/run_s4_fire_like.py \
  --input-path "$INPUT_ADV" \
  --index-dir indexes/eval_mc/robustness_musique_adversarial \
  --output-path "$OUT_ADV" \
  --limit 4 \
  --use-index \
  --claim-strategy rules \
  --verification-strategy rules \
  --query-strategy rules \
  --repair-strategy rules \
  --initial-evidence-mode auto \
  --max-claims 4 \
  --max-rounds-per-claim 3 \
  --top-k-per-round 5 \
  --max-total-retrievals 8 \
  --max-total-chunks 12

echo
echo "================================================================================"
echo "Running S4 rules+index on noisy focus rows"
echo "================================================================================"

python s4_model_code/run_s4_fire_like.py \
  --input-path "$INPUT_NOISY" \
  --index-dir indexes/eval_mc/robustness_musique_noisy \
  --output-path "$OUT_NOISY" \
  --limit 1 \
  --use-index \
  --claim-strategy rules \
  --verification-strategy rules \
  --query-strategy rules \
  --repair-strategy rules \
  --initial-evidence-mode auto \
  --max-claims 4 \
  --max-rounds-per-claim 3 \
  --top-k-per-round 5 \
  --max-total-retrievals 8 \
  --max-total-chunks 12

echo
echo "================================================================================"
echo "Summarizing S4 robustness focus rules run"
echo "================================================================================"

python evaluation/summarize_s4_robustness_focus.py \
  --input-focus-path "$INPUT_CORE" \
  --raw-paths "$OUT_ADV" "$OUT_NOISY" \
  --output-dir "$ANALYSIS_DIR" \
  --prefix "core5_rules"

echo
echo "DONE"
echo
echo "Report:"
echo "${ANALYSIS_DIR}/core5_rules_report.txt"

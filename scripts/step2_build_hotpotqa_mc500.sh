#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 2 — Construir y validar HotpotQA-MC-500
#
# Este paso puede consumir API porque build_mc_eval_dataset.py
# genera/normaliza opciones multiple-choice.
#
# Salidas principales:
#   data/eval_mc/questions_hotpotqa_mc_500.csv
#   data/eval_mc/build_summary_hotpotqa_mc_500.json
#   outputs/eval_mc/freeze_reports/hotpotqa_mc_500_freeze_report.json
#   outputs/eval_mc/freeze_reports/hotpotqa_mc_500_freeze_report.txt
#   outputs/eval_mc/rag_readiness/hotpotqa_mc_500_rag_readiness.json
#   outputs/eval_mc/rag_readiness/hotpotqa_mc_500_rag_readiness.txt
# ============================================================

echo "== 0. Precheck básico =="
test -f evaluation/build_mc_eval_dataset.py
test -f evaluation/validate_freeze_mc_benchmark.py
test -f evaluation/inspect_mc_rag_readiness.py
test -f data/hotpotqa_distractor/hotpotqa_distractor_validation.jsonl

echo "OK: scripts y raw HotpotQA local encontrados."

echo
echo "== 1. Limpiar temporales del patchcheck anterior =="
rm -rf data/eval_mc/_tmp_hotpotqa_mc_rag_patchcheck
rm -rf data/eval_mc/_tmp_2wiki_mc_rag_patchcheck
echo "OK: temporales eliminados."

echo
echo "== 2. Construir HotpotQA-MC-500 =="
python evaluation/build_mc_eval_dataset.py \
  --mode real \
  --datasets hotpotqa \
  --per-dataset 500 \
  --output-path data/eval_mc/questions_hotpotqa_mc_500.csv \
  --summary-path data/eval_mc/build_summary_hotpotqa_mc_500.json \
  --generator-model gpt-5-mini \
  --split validation \
  --max-scan 2500 \
  --seed 42 \
  --resume \
  --max-retries 2

echo
echo "== 3. Validar y congelar HotpotQA-MC-500 =="
python evaluation/validate_freeze_mc_benchmark.py \
  --questions-path data/eval_mc/questions_hotpotqa_mc_500.csv \
  --summary-path data/eval_mc/build_summary_hotpotqa_mc_500.json \
  --expected-n 500 \
  --benchmark-name hotpotqa_mc_500 \
  --output-dir outputs/eval_mc/freeze_reports

echo
echo "== 4. Readiness de contexto/evidencia para RAG =="
python evaluation/inspect_mc_rag_readiness.py \
  --questions-path data/eval_mc/questions_hotpotqa_mc_500.csv \
  --summary-path data/eval_mc/build_summary_hotpotqa_mc_500.json \
  --output-dir outputs/eval_mc/rag_readiness \
  --benchmark-name hotpotqa_mc_500

echo
echo "== 5. Resumen verificable =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

q_path = Path("data/eval_mc/questions_hotpotqa_mc_500.csv")
s_path = Path("data/eval_mc/build_summary_hotpotqa_mc_500.json")
freeze_txt = Path("outputs/eval_mc/freeze_reports/hotpotqa_mc_500_freeze_report.txt")
readiness_txt = Path("outputs/eval_mc/rag_readiness/hotpotqa_mc_500_rag_readiness.txt")

df = pd.read_csv(q_path)

print("questions_path:", q_path)
print("summary_path:", s_path)
print("shape:", df.shape)
print("dataset values:", sorted(df["dataset"].dropna().astype(str).unique().tolist()))
print("source_dataset values:", sorted(df["source_dataset"].dropna().astype(str).unique().tolist()))
print("source_split values:", sorted(df["source_split"].dropna().astype(str).unique().tolist()))
print("gold_answer distribution:", df["gold_answer"].value_counts().sort_index().to_dict())
print("requires_retrieval distribution:", df["requires_retrieval"].value_counts(dropna=False).to_dict())

for col in ["context_titles_json", "evidence_json"]:
    non_empty = df[col].fillna("").astype(str).str.len().gt(2).sum()
    print(f"{col} non-empty:", int(non_empty), "/", len(df))

try:
    summary = json.loads(s_path.read_text(encoding="utf-8"))
    print("summary keys:", sorted(summary.keys()))
    for k in ["mode", "datasets", "per_dataset", "generated_rows", "n_rows", "output_path"]:
        if k in summary:
            print(f"summary {k}:", summary[k])
except Exception as e:
    print("WARNING: no pude leer summary json:", e)

print("freeze_report_exists:", freeze_txt.exists(), freeze_txt)
print("readiness_report_exists:", readiness_txt.exists(), readiness_txt)
PY

echo
echo "== 6. Estado Git resumido =="
git status --short

echo
echo "LISTO: HotpotQA-MC-500 construido y validado."

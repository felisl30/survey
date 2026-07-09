#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 1 — Parche mínimo para build_mc_rag_dataset.py
# Objetivo:
#   Evitar que HotpotQA y 2Wiki queden etiquetados como "musique"
#   en la columna source_dataset al construir datasets RAG.
#
# Uso:
#   cd ~/Documents/natural_language_processing/trabajo_cientifico
#   source tp_cientifico/bin/activate
#   bash scripts/patch_source_dataset_build_mc_rag.sh
# ============================================================

echo "== 1. Backup del script original =="
BACKUP="evaluation/build_mc_rag_dataset.py.bak_source_dataset_$(date +%Y%m%d_%H%M%S)"
cp evaluation/build_mc_rag_dataset.py "$BACKUP"
echo "Backup creado en: $BACKUP"

echo
echo "== 2. Aplicando parche source_dataset =="
python - <<'PY'
from pathlib import Path

path = Path("evaluation/build_mc_rag_dataset.py")
text = path.read_text(encoding="utf-8")

old_marker = '        dataset = clean_str(row.get("dataset", benchmark_name))\n'
new_marker = (
    '        dataset = clean_str(row.get("dataset", benchmark_name))\n'
    '        source_dataset = clean_str(row.get("source_dataset", "")) or dataset\n'
)

if new_marker in text:
    print("OK: la variable source_dataset ya estaba agregada.")
else:
    if old_marker not in text:
        raise SystemExit("ERROR: no encontré la línea esperada de dataset. No se modificó el archivo.")
    text = text.replace(old_marker, new_marker, 1)
    print("OK: agregada variable source_dataset dentro del loop.")

count_before = text.count('"source_dataset": "musique"')
text = text.replace('"source_dataset": "musique"', '"source_dataset": source_dataset')
count_after = text.count('"source_dataset": "musique"')

path.write_text(text, encoding="utf-8")

print(f"Hardcodes reemplazados: {count_before - count_after}")
print(f"Hardcodes restantes: {count_after}")

if count_after != 0:
    raise SystemExit("ERROR: todavía quedan source_dataset hardcodeados como musique.")
PY

echo
echo "== 3. Compilación sintáctica =="
python -m py_compile evaluation/build_mc_rag_dataset.py
echo "OK: build_mc_rag_dataset.py compila."

echo
echo "== 4. Verificación textual del parche =="
grep -nE 'dataset = clean_str|source_dataset' evaluation/build_mc_rag_dataset.py | sed -n '1,80p'

echo
echo "== 5. Smoke sin API: construir RAG temporal HotpotQA-MC-100 =="
rm -rf data/eval_mc/_tmp_hotpotqa_mc_rag_patchcheck
python evaluation/build_mc_rag_dataset.py \
  --input-path data/eval_mc/questions_hotpotqa_mc_100.csv \
  --output-dir data/eval_mc/_tmp_hotpotqa_mc_rag_patchcheck \
  --benchmark-name hotpotqa_mc_100_patchcheck \
  --expected-n 100

echo
echo "== 6. Smoke sin API: construir RAG temporal 2Wiki-MC-100 =="
rm -rf data/eval_mc/_tmp_2wiki_mc_rag_patchcheck
python evaluation/build_mc_rag_dataset.py \
  --input-path data/eval_mc/questions_2wiki_mc_100.csv \
  --output-dir data/eval_mc/_tmp_2wiki_mc_rag_patchcheck \
  --benchmark-name 2wiki_mc_100_patchcheck \
  --expected-n 100

echo
echo "== 7. Verificación de metadatos generados =="
python - <<'PY'
import pandas as pd
from pathlib import Path

checks = [
    ("hotpotqa", Path("data/eval_mc/_tmp_hotpotqa_mc_rag_patchcheck")),
    ("2wiki", Path("data/eval_mc/_tmp_2wiki_mc_rag_patchcheck")),
]

for expected, root in checks:
    print(f"\n--- {root}")
    q = pd.read_csv(root / "questions.csv")
    c = pd.read_csv(root / "corpus.csv")
    qr = pd.read_csv(root / "qrels.csv")

    print("questions shape:", q.shape)
    print("corpus shape:", c.shape)
    print("qrels shape:", qr.shape)

    print("questions source_dataset:", sorted(q["source_dataset"].dropna().unique().tolist()))
    print("corpus source_dataset:", sorted(c["source_dataset"].dropna().unique().tolist()))
    print("questions benchmark_name:", sorted(q["benchmark_name"].dropna().unique().tolist()))
    print("corpus benchmark_name:", sorted(c["benchmark_name"].dropna().unique().tolist()))

    q_sources = set(q["source_dataset"].dropna().astype(str))
    c_sources = set(c["source_dataset"].dropna().astype(str))

    if not all(expected in s.lower() for s in q_sources):
        raise SystemExit(f"ERROR: questions.csv no quedó etiquetado como {expected}: {q_sources}")
    if not all(expected in s.lower() for s in c_sources):
        raise SystemExit(f"ERROR: corpus.csv no quedó etiquetado como {expected}: {c_sources}")

print("\nOK: parche validado en HotpotQA y 2Wiki.")
PY

echo
echo "== 8. Estado Git =="
git diff -- evaluation/build_mc_rag_dataset.py
git status --short

echo
echo "LISTO: primer paso terminado."

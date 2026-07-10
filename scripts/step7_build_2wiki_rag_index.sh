#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 7 — Construir 2Wiki-MC-500 RAG + índice vectorial
#
# Este paso NO llama al LLM.
# Convierte:
#   data/eval_mc/questions_2wiki_mc_500.csv
# en:
#   data/eval_mc/2wiki_mc_rag_500/questions.csv
#   data/eval_mc/2wiki_mc_rag_500/corpus.csv
#   data/eval_mc/2wiki_mc_rag_500/qrels.csv
#
# Luego construye:
#   indexes/eval_mc/2wiki_mc_rag_500/chunks.csv
#   indexes/eval_mc/2wiki_mc_rag_500/embeddings.npy
#   indexes/eval_mc/2wiki_mc_rag_500/metadata.json
# ============================================================

echo "== 0. Precheck básico =="
test -f evaluation/build_mc_rag_dataset.py
test -f evaluation/build_mc_rag_index.py
test -f data/eval_mc/questions_2wiki_mc_500.csv
test -f data/eval_mc/build_summary_2wiki_mc_500.json

echo "OK: scripts y 2Wiki-MC-500 encontrados."

echo
echo "== 1. Construir dataset RAG 2Wiki-MC-500 =="
rm -rf data/eval_mc/2wiki_mc_rag_500

python evaluation/build_mc_rag_dataset.py \
  --input-path data/eval_mc/questions_2wiki_mc_500.csv \
  --output-dir data/eval_mc/2wiki_mc_rag_500 \
  --benchmark-name 2wiki_mc_500 \
  --expected-n 500

echo
echo "== 2. Construir índice vectorial 2Wiki-MC-500 =="
rm -rf indexes/eval_mc/2wiki_mc_rag_500

python evaluation/build_mc_rag_index.py \
  --corpus-path data/eval_mc/2wiki_mc_rag_500/corpus.csv \
  --output-dir indexes/eval_mc/2wiki_mc_rag_500

echo
echo "== 3. Verificación de dataset RAG e índice =="
python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

rag_dir = Path("data/eval_mc/2wiki_mc_rag_500")
idx_dir = Path("indexes/eval_mc/2wiki_mc_rag_500")

paths = {
    "questions": rag_dir / "questions.csv",
    "corpus": rag_dir / "corpus.csv",
    "qrels": rag_dir / "qrels.csv",
    "rag_summary": rag_dir / "build_summary.json",
    "chunks": idx_dir / "chunks.csv",
    "embeddings": idx_dir / "embeddings.npy",
    "index_metadata": idx_dir / "metadata.json",
}

for name, path in paths.items():
    print(f"{name}_exists:", path.exists(), path)
    if not path.exists():
        raise SystemExit(f"ERROR: falta {name}: {path}")

q = pd.read_csv(paths["questions"])
c = pd.read_csv(paths["corpus"])
qr = pd.read_csv(paths["qrels"])
chunks = pd.read_csv(paths["chunks"])
emb = np.load(paths["embeddings"])

print()
print("questions shape:", q.shape)
print("corpus shape:", c.shape)
print("qrels shape:", qr.shape)
print("chunks shape:", chunks.shape)
print("embeddings shape:", emb.shape)

print()
print("questions dataset:", sorted(q["dataset"].dropna().astype(str).unique().tolist()))
print("questions source_dataset:", sorted(q["source_dataset"].dropna().astype(str).unique().tolist()))
print("corpus source_dataset:", sorted(c["source_dataset"].dropna().astype(str).unique().tolist()))
print("questions benchmark_name:", sorted(q["benchmark_name"].dropna().astype(str).unique().tolist()))
print("corpus benchmark_name:", sorted(c["benchmark_name"].dropna().astype(str).unique().tolist()))

print()
print("qrels relevance distribution:", qr["relevance"].value_counts(dropna=False).sort_index().to_dict())
print("docs per question, min:", int(qr.groupby("question_id")["doc_id"].nunique().min()))
print("docs per question, max:", int(qr.groupby("question_id")["doc_id"].nunique().max()))
print("docs per question, mean:", round(float(qr.groupby("question_id")["doc_id"].nunique().mean()), 2))

missing_context = sorted(set(q["question_id"]) - set(qr["question_id"]))
print("questions without qrels:", len(missing_context))

if len(q) != 500:
    raise SystemExit(f"ERROR: questions debería tener 500 filas y tiene {len(q)}")
if len(c) == 0 or len(qr) == 0:
    raise SystemExit("ERROR: corpus/qrels vacíos")
if emb.shape[0] != len(chunks):
    raise SystemExit(f"ERROR: embeddings rows {emb.shape[0]} != chunks rows {len(chunks)}")
if emb.shape[1] != 384:
    raise SystemExit(f"ERROR: embedding dim esperada 384, obtenida {emb.shape[1]}")
if missing_context:
    raise SystemExit("ERROR: hay preguntas sin qrels/contexto")
if set(q["source_dataset"].dropna().astype(str)) != {"2wiki"}:
    raise SystemExit("ERROR: questions source_dataset no es 2wiki")
if set(c["source_dataset"].dropna().astype(str)) != {"2wiki"}:
    raise SystemExit("ERROR: corpus source_dataset no es 2wiki")

try:
    rag_summary = json.loads(paths["rag_summary"].read_text(encoding="utf-8"))
    print()
    print("rag_summary keys:", sorted(rag_summary.keys()))
    for k in ["benchmark_name", "n_questions", "n_corpus_docs", "n_qrels", "questions_without_context", "avg_contexts_per_question"]:
        if k in rag_summary:
            print(f"rag_summary {k}:", rag_summary[k])
except Exception as e:
    print("WARNING: no pude leer rag_summary:", e)

try:
    metadata = json.loads(paths["index_metadata"].read_text(encoding="utf-8"))
    print()
    print("index_metadata keys:", sorted(metadata.keys()))
    for k in ["embedding_model", "embedding_dim", "n_chunks", "corpus_path", "created_at"]:
        if k in metadata:
            print(f"index_metadata {k}:", metadata[k])
except Exception as e:
    print("WARNING: no pude leer index metadata:", e)

print()
print("OK: 2Wiki-MC-500 RAG + índice verificados.")
PY

echo
echo "== 4. Estado Git resumido =="
git status --short

echo
echo "LISTO: 2Wiki-MC-500 RAG + índice construidos."

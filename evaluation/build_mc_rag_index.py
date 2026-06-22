#!/usr/bin/env python3
"""
build_mc_rag_index.py

Construye un índice vectorial simple para un dataset MC-RAG.

Entrada:
    data/eval_mc/musique_mc_rag/corpus.csv

Salida:
    indexes/eval_mc/musique_mc_rag/chunks.csv
    indexes/eval_mc/musique_mc_rag/embeddings.npy
    indexes/eval_mc/musique_mc_rag/metadata.json

No llama a OpenAI. Usa sentence-transformers local/descargado.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_index(
    *,
    corpus_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    force: bool,
) -> dict:
    if not corpus_path.exists():
        raise FileNotFoundError(f"No existe corpus_path: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.csv"
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"

    if not force:
        existing = [p for p in [chunks_path, embeddings_path, metadata_path] if p.exists()]
        if existing:
            raise FileExistsError(
                "Ya existen archivos de índice. Usá --force para sobrescribir:\n"
                + "\n".join(str(p) for p in existing)
            )

    corpus = pd.read_csv(corpus_path)

    required = {"doc_id", "question_id", "title", "text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en corpus.csv: {sorted(missing)}")

    corpus = corpus.copy()
    corpus["title"] = corpus["title"].fillna("").astype(str)
    corpus["text"] = corpus["text"].fillna("").astype(str)

    corpus["chunk_id"] = corpus["doc_id"].astype(str)

    # Texto que se embebe. Incluye título porque en este benchmark ayuda bastante.
    corpus["chunk_text"] = (
        corpus["title"].str.strip()
        + "\n\n"
        + corpus["text"].str.strip()
    ).str.strip()

    empty_chunks = int(corpus["chunk_text"].eq("").sum())
    if empty_chunks > 0:
        raise ValueError(f"Hay {empty_chunks} chunks vacíos.")

    print(f"Corpus: {corpus_path}")
    print(f"Filas corpus: {len(corpus)}")
    print(f"Modelo embeddings: {model_name}")
    print("Cargando modelo...")

    model = SentenceTransformer(model_name)

    print("Generando embeddings...")
    embeddings = model.encode(
        corpus["chunk_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = embeddings.astype("float32")
    embeddings = normalize_embeddings(embeddings)

    chunks_cols = [
        "chunk_id",
        "doc_id",
        "document_id",
        "question_id",
        "source_question_id",
        "source_dataset",
        "benchmark_name",
        "rank_in_source",
        "title",
        "text",
        "chunk_text",
    ]
    chunks_cols = [c for c in chunks_cols if c in corpus.columns]

    corpus[chunks_cols].to_csv(chunks_path, index=False)
    np.save(embeddings_path, embeddings)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_path": str(corpus_path),
        "output_dir": str(output_dir),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "embedding_model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "n_chunks": int(embeddings.shape[0]),
        "normalized_embeddings": True,
        "batch_size": batch_size,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/corpus.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("indexes/eval_mc/musique_mc_rag"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    metadata = build_index(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        force=args.force,
    )

    print("\nMC RAG index built")
    print("------------------")
    print(f"Chunks: {metadata['n_chunks']}")
    print(f"Embedding dim: {metadata['embedding_dim']}")
    print(f"Chunks path: {metadata['chunks_path']}")
    print(f"Embeddings path: {metadata['embeddings_path']}")
    print(f"Metadata path: {Path(args.output_dir) / 'metadata.json'}")


if __name__ == "__main__":
    main()

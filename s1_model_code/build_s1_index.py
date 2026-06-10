#!/usr/bin/env python3
"""
build_s1_index.py

Construye el índice vectorial de S1 a partir de corpus_s1.csv.

Entrada:
    data/s1/hotpotqa_mini/corpus_s1.csv

Salidas:
    indexes/s1/hotpotqa_mini/chunks.csv
    indexes/s1/hotpotqa_mini/embeddings.npy
    indexes/s1/hotpotqa_mini/metadata.json

Diseño:
    - Usa sentence-transformers para calcular embeddings locales.
    - Combina title + text para representar cada chunk.
    - Normaliza los embeddings para que cosine similarity = dot product.
    - No usa FAISS todavía; con ~200 chunks alcanza numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import S1_CORPUS_PATH, S1_INDEX_DIR


DEFAULT_CORPUS_PATH = S1_CORPUS_PATH
DEFAULT_INDEX_DIR = S1_INDEX_DIR
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_COLUMNS = {
    "chunk_id",
    "title",
    "text",
}


def clean_text(value) -> str:
    """Convierte valores a texto limpio."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_corpus(df: pd.DataFrame) -> None:
    """Valida columnas mínimas del corpus."""
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "El corpus no tiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("El corpus está vacío.")

    if df["chunk_id"].duplicated().any():
        duplicated = df[df["chunk_id"].duplicated()]["chunk_id"].tolist()
        raise ValueError(
            "Hay chunk_id duplicados. Ejemplos: "
            + ", ".join(map(str, duplicated[:10]))
        )


def build_embedding_text(row: pd.Series) -> str:
    """
    Construye el texto usado para el embedding.

    Usar título + texto suele mejorar retrieval porque HotpotQA trabaja con
    entidades de Wikipedia y los títulos son muy informativos.
    """
    title = clean_text(row.get("title", ""))
    text = clean_text(row.get("text", ""))

    if title and text:
        return f"Title: {title}\nText: {text}"

    if text:
        return text

    return title


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza filas de una matriz con norma L2."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def build_index(
    *,
    corpus_path: Path,
    index_dir: Path,
    embedding_model_name: str,
    batch_size: int,
) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(f"No se encontró el corpus: {corpus_path}")

    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo corpus desde: {corpus_path}")
    corpus_df = pd.read_csv(corpus_path)
    validate_corpus(corpus_df)

    corpus_df = corpus_df.copy()
    corpus_df["embedding_text"] = corpus_df.apply(build_embedding_text, axis=1)

    empty_embedding_text = corpus_df["embedding_text"].apply(lambda x: not str(x).strip()).sum()
    if empty_embedding_text > 0:
        raise ValueError(
            f"Hay {empty_embedding_text} chunks sin texto utilizable para embeddings."
        )

    print(f"Chunks a indexar: {len(corpus_df)}")
    print(f"Modelo de embeddings: {embedding_model_name}")

    model = SentenceTransformer(embedding_model_name)

    texts = corpus_df["embedding_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = embeddings.astype("float32")
    embeddings = l2_normalize(embeddings).astype("float32")

    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.json"

    print(f"Guardando chunks en: {chunks_path}")
    corpus_df.to_csv(chunks_path, index=False)

    print(f"Guardando embeddings en: {embeddings_path}")
    np.save(embeddings_path, embeddings)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model_name,
        "n_chunks": int(len(corpus_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_file": str(embeddings_path),
        "chunks_file": str(chunks_path),
        "normalized_embeddings": True,
        "similarity": "cosine_similarity_via_dot_product",
        "text_field": "embedding_text = Title + Text",
        "batch_size": batch_size,
        "columns": list(corpus_df.columns),
    }

    print(f"Guardando metadata en: {metadata_path}")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nÍndice S1 construido correctamente.")
    print(f"Chunks: {len(corpus_df)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Index dir: {index_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye índice vectorial para S1 RAG básico."
    )

    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Ruta al corpus_s1.csv.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio donde guardar el índice.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modelo de sentence-transformers.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size para calcular embeddings.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    build_index(
        corpus_path=args.corpus_path,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

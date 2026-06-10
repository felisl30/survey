#!/usr/bin/env python3
"""
build_s2_index.py

Construye el índice vectorial de S2 Adaptive-RAG a partir de corpus_s2.csv.

Entrada por defecto:
    data/s2/adaptive_rag/corpus_s2.csv

Salidas por defecto:
    indexes/s2/adaptive_rag/chunks.csv
    indexes/s2/adaptive_rag/embeddings.npy
    indexes/s2/adaptive_rag/metadata.json

Diseño:
    - Usa sentence-transformers para calcular embeddings locales.
    - Combina title + text para representar cada chunk.
    - Normaliza los embeddings para que cosine similarity = dot product.
    - No usa FAISS; con corpus chicos/medianos alcanza numpy.
    - Es compatible con la lógica de retriever_s1.py si se le pasa
      --index-dir indexes/s2/adaptive_rag.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_CORPUS_PATH = Path("data/s2/adaptive_rag/corpus_s2.csv")
DEFAULT_INDEX_DIR = Path("indexes/s2/adaptive_rag")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_COLUMNS = {
    "chunk_id",
    "title",
    "text",
}


def clean_text(value: Any) -> str:
    """Convierte valores a texto limpio."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def validate_corpus(df: pd.DataFrame) -> None:
    """Valida columnas y consistencia mínima del corpus."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "El corpus no tiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("El corpus está vacío.")

    if df["chunk_id"].duplicated().any():
        duplicated = df[df["chunk_id"].duplicated()]["chunk_id"].astype(str).tolist()
        raise ValueError(
            "Hay chunk_id duplicados. Ejemplos: "
            + ", ".join(duplicated[:10])
        )

    empty_text = df["text"].apply(lambda x: not clean_text(x)).sum()
    if empty_text > 0:
        raise ValueError(f"Hay {empty_text} chunks con text vacío.")


def build_embedding_text(row: pd.Series) -> str:
    """
    Construye el texto usado para embeddings.

    Usar título + texto suele mejorar retrieval en HotpotQA porque los títulos
    de Wikipedia contienen entidades muy informativas.
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


def write_metadata(
    *,
    metadata_path: Path,
    corpus_path: Path,
    index_dir: Path,
    embedding_model_name: str,
    embeddings: np.ndarray,
    chunks_path: Path,
    embeddings_path: Path,
    corpus_df: pd.DataFrame,
    batch_size: int,
) -> dict[str, Any]:
    metadata = {
        "system": "S2_adaptive_rag",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model_name,
        "n_chunks": int(len(corpus_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_file": str(embeddings_path),
        "chunks_file": str(chunks_path),
        "metadata_file": str(metadata_path),
        "normalized_embeddings": True,
        "similarity": "cosine_similarity_via_dot_product",
        "text_field": "embedding_text = Title + Text",
        "batch_size": int(batch_size),
        "columns": list(corpus_df.columns),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def verify_index(index_dir: Path) -> dict[str, Any]:
    """Verifica que chunks, embeddings y metadata existan y sean coherentes."""
    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.json"

    for path in [chunks_path, embeddings_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se generó el archivo esperado: {path}")

    chunks_df = pd.read_csv(chunks_path)
    embeddings = np.load(embeddings_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(chunks_df) != embeddings.shape[0]:
        raise ValueError(
            f"Inconsistencia: chunks={len(chunks_df)} pero embeddings={embeddings.shape[0]}"
        )

    if int(metadata.get("n_chunks", -1)) != len(chunks_df):
        raise ValueError(
            f"metadata n_chunks={metadata.get('n_chunks')} no coincide con chunks={len(chunks_df)}"
        )

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings.npy debe ser matriz 2D. Shape recibido: {embeddings.shape}")

    # Verificación de normalización aproximada.
    norms = np.linalg.norm(embeddings.astype("float32"), axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0))) if len(norms) else 0.0

    return {
        "chunks": int(len(chunks_df)),
        "embedding_shape": tuple(int(x) for x in embeddings.shape),
        "embedding_dim": int(embeddings.shape[1]),
        "max_norm_error": max_norm_error,
        "metadata_embedding_model": metadata.get("embedding_model", ""),
    }


def build_index(
    *,
    corpus_path: Path,
    index_dir: Path,
    embedding_model_name: str,
    batch_size: int,
    overwrite: bool,
) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(f"No se encontró el corpus S2: {corpus_path}")

    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.json"

    existing_files = [path for path in [chunks_path, embeddings_path, metadata_path] if path.exists()]
    if existing_files and not overwrite:
        raise FileExistsError(
            "Ya existe un índice en el directorio indicado. "
            "Usá --overwrite para regenerarlo. Archivos existentes: "
            + ", ".join(str(path) for path in existing_files)
        )

    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo corpus S2 desde: {corpus_path}")
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
    print(f"Batch size: {batch_size}")

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

    print(f"Guardando chunks en: {chunks_path}")
    corpus_df.to_csv(chunks_path, index=False)

    print(f"Guardando embeddings en: {embeddings_path}")
    np.save(embeddings_path, embeddings)

    print(f"Guardando metadata en: {metadata_path}")
    write_metadata(
        metadata_path=metadata_path,
        corpus_path=corpus_path,
        index_dir=index_dir,
        embedding_model_name=embedding_model_name,
        embeddings=embeddings,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        corpus_df=corpus_df,
        batch_size=batch_size,
    )

    verification = verify_index(index_dir)

    print("\nÍndice S2 construido correctamente.")
    print(f"Chunks: {verification['chunks']}")
    print(f"Embedding shape: {verification['embedding_shape']}")
    print(f"Embedding dim: {verification['embedding_dim']}")
    print(f"Max norm error: {verification['max_norm_error']:.8f}")
    print(f"Index dir: {index_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye índice vectorial para S2 Adaptive-RAG."
    )

    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Ruta al corpus_s2.csv.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio donde guardar el índice S2.",
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenera el índice aunque ya existan chunks.csv/embeddings.npy/metadata.json.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size debe ser mayor que 0.")

    build_index(
        corpus_path=args.corpus_path,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

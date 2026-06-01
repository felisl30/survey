#!/usr/bin/env python3
"""
retriever_s1.py

Retriever reutilizable para S1: RAG básico sobre HotpotQA-mini.

Entradas esperadas:
    indexes/s1/hotpotqa_mini/chunks.csv
    indexes/s1/hotpotqa_mini/embeddings.npy
    indexes/s1/hotpotqa_mini/metadata.json

Diseño:
    - Reutiliza el índice creado por build_s1_index.py.
    - Usa el mismo modelo de embeddings registrado en metadata.json.
    - Calcula similitud coseno como producto punto porque los embeddings del
      índice están normalizados.
    - Devuelve top-k chunks con texto, título, score y metadatos.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_DIR = Path("indexes/s1/hotpotqa_mini")
DEFAULT_SELECTED_TOP_K_PATH = Path("outputs/s1/selected_top_k.txt")
DEFAULT_TOP_K = 5


def is_missing(value: Any) -> bool:
    """Detecta None/NaN/cadenas vacías."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    """Convierte valores a string limpio, evitando 'nan'."""
    if is_missing(value):
        return ""
    return str(value).strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza filas de una matriz con norma L2."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def read_selected_top_k(
    path: Path = DEFAULT_SELECTED_TOP_K_PATH,
    default: int = DEFAULT_TOP_K,
) -> int:
    """Lee el top-k seleccionado para S1; si no existe, usa default."""
    if not path.exists():
        return default

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return default

    try:
        value = int(text)
    except ValueError:
        return default

    if value <= 0:
        return default

    return value


def get_retrieval_query(row: pd.Series) -> str:
    """
    Devuelve la query limpia para retrieval.

    Para S1 conviene usar la pregunta original, no el prompt completo con
    instrucciones de formato. Por eso se prioriza original_question.
    """
    for col in ["original_question", "question", "prompt"]:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value

    raise ValueError(f"Fila con id={row.get('id', '<sin id>')} sin query válida.")


class S1Retriever:
    """Retriever vectorial simple para S1."""

    def __init__(
        self,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embedding_model_name: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.chunks_path = self.index_dir / "chunks.csv"
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.metadata_path = self.index_dir / "metadata.json"

        self._validate_files()

        self.metadata = self._load_metadata()
        self.embedding_model_name = (
            embedding_model_name
            or clean_text(self.metadata.get("embedding_model", ""))
        )

        if not self.embedding_model_name:
            raise ValueError(
                "No se pudo determinar el modelo de embeddings. "
                "Pasalo con --embedding-model o revisá metadata.json."
            )

        self.chunks_df = pd.read_csv(self.chunks_path)
        self.chunk_embeddings = np.load(self.embeddings_path).astype("float32")

        self._validate_index()

        # Carga lazy del modelo: se inicializa al primer retrieve.
        self._model: SentenceTransformer | None = None

    def _validate_files(self) -> None:
        for path in [self.chunks_path, self.embeddings_path, self.metadata_path]:
            if not path.exists():
                raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")

    def _load_metadata(self) -> dict[str, Any]:
        with self.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_index(self) -> None:
        if self.chunks_df.empty:
            raise ValueError(f"El archivo de chunks está vacío: {self.chunks_path}")

        required_columns = {"chunk_id", "title", "text"}
        missing = required_columns - set(self.chunks_df.columns)
        if missing:
            raise ValueError(
                "chunks.csv no tiene las columnas requeridas: "
                + ", ".join(sorted(missing))
            )

        if len(self.chunks_df) != self.chunk_embeddings.shape[0]:
            raise ValueError(
                f"Chunks ({len(self.chunks_df)}) y embeddings "
                f"({self.chunk_embeddings.shape[0]}) no coinciden."
            )

        if self.chunk_embeddings.ndim != 2:
            raise ValueError(
                f"embeddings.npy debería ser una matriz 2D. "
                f"Shape recibido: {self.chunk_embeddings.shape}"
            )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """Encodea y normaliza una query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query vacía o inválida.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype("float32")

        query_embedding = l2_normalize(query_embedding).astype("float32")
        return query_embedding[0]

    def retrieve(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Devuelve los top-k chunks más similares a la query."""
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0.")

        if top_k > len(self.chunks_df):
            top_k = len(self.chunks_df)

        query_embedding = self.encode_query(query)

        scores = self.chunk_embeddings @ query_embedding
        top_indices = np.argsort(-scores)[:top_k]

        retrieved: list[dict[str, Any]] = []

        for rank, idx in enumerate(top_indices, start=1):
            idx_int = int(idx)
            row = self.chunks_df.iloc[idx_int]

            item: dict[str, Any] = {
                "rank": rank,
                "chunk_id": clean_text(row.get("chunk_id", "")),
                "doc_id": clean_text(row.get("doc_id", "")),
                "title": clean_text(row.get("title", "")),
                "text": clean_text(row.get("text", "")),
                "score": float(scores[idx_int]),
                "index_position": idx_int,
            }

            # Conserva metadatos útiles si existen en chunks.csv.
            for col in [
                "source",
                "source_split",
                "topic",
                "question_id",
                "original_hotpotqa_id",
                "paragraph_index",
                "is_gold_evidence",
            ]:
                if col in row.index:
                    value = row.get(col, "")
                    if isinstance(value, (np.bool_, bool)):
                        item[col] = bool(value)
                    elif isinstance(value, (np.integer, int)):
                        item[col] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        item[col] = float(value)
                    else:
                        item[col] = clean_text(value)

            retrieved.append(item)

        return retrieved

    def retrieve_for_row(
        self,
        row: pd.Series,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extrae la query de una fila y recupera top-k chunks."""
        query = get_retrieval_query(row)
        return query, self.retrieve(query, top_k=top_k)


def compact_retrieval_for_print(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce el resultado para imprimirlo por terminal o JSON."""
    compact = []

    for item in items:
        compact.append(
            {
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "score": round(float(item["score"]), 6),
                "text_preview": item["text"][:220],
            }
        )

    return compact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prueba rápida del retriever S1 sobre el índice construido."
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio del índice S1.",
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Pregunta/query para recuperar chunks.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Cantidad de chunks a recuperar. Si se omite, lee outputs/s1/selected_top_k.txt o usa 5.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Modelo de embeddings. Si se omite, usa el registrado en metadata.json.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    top_k = args.top_k
    if top_k is None:
        top_k = read_selected_top_k()

    retriever = S1Retriever(
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
    )

    results = retriever.retrieve(args.query, top_k=top_k)

    print(
        json.dumps(
            {
                "query": args.query,
                "top_k": top_k,
                "embedding_model": retriever.embedding_model_name,
                "results": compact_retrieval_for_print(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

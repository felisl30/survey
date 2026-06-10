#!/usr/bin/env python3
"""
evaluate_s1_retrieval.py

Evalúa el retriever de S1 sin llamar al LLM.

Entradas:
    data/hotpotqa_mini/questions_s1.csv
    data/hotpotqa_mini/qrels_s1.csv

    Cambio esas ultimas dos por: 
    data//s1/hotpotqa_mini/questions_s1.csv
    data/s1/hotpotqa_mini/qrels_s1.csv




    indexes/s1/hotpotqa_mini/chunks.csv
    indexes/s1/hotpotqa_mini/embeddings.npy
    indexes/s1/hotpotqa_mini/metadata.json

Salidas:
    outputs/s1/retrieval/hotpotqa_mini_retrieval_results.csv
    outputs/s1/retrieval/hotpotqa_mini_retrieval_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import S1_INDEX_DIR, S1_QRELS_PATH, S1_QUESTIONS_PATH, S1_RETRIEVAL_DIR


DEFAULT_QUESTIONS_PATH = S1_QUESTIONS_PATH
DEFAULT_QRELS_PATH = S1_QRELS_PATH
DEFAULT_INDEX_DIR = S1_INDEX_DIR
DEFAULT_OUTPUT_DIR = S1_RETRIEVAL_DIR


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def load_metadata(index_dir: Path) -> dict[str, Any]:
    metadata_path = index_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró metadata.json en {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_files(
    questions_path: Path,
    qrels_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
) -> None:
    for path in [questions_path, qrels_path, chunks_path, embeddings_path]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")


def get_query(row: pd.Series) -> str:
    """
    Para retrieval usamos la pregunta original, no el prompt completo.

    Prioridad:
        1. original_question
        2. question
        3. prompt
    """
    for col in ["original_question", "question", "prompt"]:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value

    raise ValueError(f"No hay query válida para id={row.get('id', '<sin id>')}")


def build_gold_map(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Construye:
        question_id -> set(chunk_id relevantes)
    """
    qrels_df = qrels_df.copy()
    qrels_df["relevance"] = pd.to_numeric(
        qrels_df["relevance"],
        errors="coerce"
    ).fillna(0)

    gold_map: dict[str, set[str]] = {}

    positive_qrels = qrels_df[qrels_df["relevance"] > 0]

    for question_id, group in positive_qrels.groupby("question_id"):
        gold_map[str(question_id)] = set(group["chunk_id"].astype(str))

    return gold_map


def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype("float32")

    query_embedding = l2_normalize(query_embedding).astype("float32")

    scores = chunk_embeddings @ query_embedding[0]
    top_indices = np.argsort(-scores)[:top_k]

    retrieved = []

    for rank, idx in enumerate(top_indices, start=1):
        row = chunks_df.iloc[int(idx)]

        retrieved.append(
            {
                "rank": rank,
                "chunk_id": clean_text(row["chunk_id"]),
                "title": clean_text(row.get("title", "")),
                "score": float(scores[int(idx)]),
            }
        )

    return retrieved


def evaluate_query(
    question_id: str,
    query: str,
    gold_chunk_ids: set[str],
    retrieved: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    retrieved_chunk_ids = [item["chunk_id"] for item in retrieved]
    retrieved_titles = [item["title"] for item in retrieved]
    retrieved_scores = [item["score"] for item in retrieved]

    retrieved_set = set(retrieved_chunk_ids)
    matched = retrieved_set & gold_chunk_ids

    hit_at_k = len(matched) > 0
    recall_at_k = len(matched) / len(gold_chunk_ids) if gold_chunk_ids else 0.0
    precision_at_k = len(matched) / top_k if top_k else 0.0

    first_hit_rank = None
    mrr_at_k = 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_chunk_ids:
            first_hit_rank = rank
            mrr_at_k = 1.0 / rank
            break

    return {
        "question_id": question_id,
        "query": query,
        "n_gold": len(gold_chunk_ids),
        "gold_chunk_ids": "|".join(sorted(gold_chunk_ids)),
        "retrieved_chunk_ids": "|".join(retrieved_chunk_ids),
        "retrieved_titles": "|".join(retrieved_titles),
        "retrieved_scores": "|".join(f"{score:.6f}" for score in retrieved_scores),
        "matched_gold_chunk_ids": "|".join(sorted(matched)),
        "hit_at_k": bool(hit_at_k),
        "recall_at_k": float(recall_at_k),
        "precision_at_k": float(precision_at_k),
        "mrr_at_k": float(mrr_at_k),
        "first_hit_rank": first_hit_rank,
    }


def summarize(results_df: pd.DataFrame, top_k: int) -> dict[str, Any]:
    summary = {
        "top_k": top_k,
        "n_questions": int(len(results_df)),
        f"hit@{top_k}": float(results_df["hit_at_k"].astype(bool).mean()),
        f"recall@{top_k}": float(results_df["recall_at_k"].mean()),
        f"precision@{top_k}": float(results_df["precision_at_k"].mean()),
        f"mrr@{top_k}": float(results_df["mrr_at_k"].mean()),
        "avg_n_gold": float(results_df["n_gold"].mean()),
        "questions_without_gold": int((results_df["n_gold"] == 0).sum()),
    }

    if "topic" in results_df.columns:
        summary["by_topic"] = {}
        for topic, group in results_df.groupby("topic", dropna=False):
            summary["by_topic"][str(topic)] = {
                "n": int(len(group)),
                f"hit@{top_k}": float(group["hit_at_k"].astype(bool).mean()),
                f"recall@{top_k}": float(group["recall_at_k"].mean()),
                f"mrr@{top_k}": float(group["mrr_at_k"].mean()),
            }

    if "hotpot_type" in results_df.columns:
        summary["by_hotpot_type"] = {}
        for hotpot_type, group in results_df.groupby("hotpot_type", dropna=False):
            summary["by_hotpot_type"][str(hotpot_type)] = {
                "n": int(len(group)),
                f"hit@{top_k}": float(group["hit_at_k"].astype(bool).mean()),
                f"recall@{top_k}": float(group["recall_at_k"].mean()),
                f"mrr@{top_k}": float(group["mrr_at_k"].mean()),
            }

    return summary


def run_evaluation(
    questions_path: Path,
    qrels_path: Path,
    index_dir: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"

    validate_files(
        questions_path=questions_path,
        qrels_path=qrels_path,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
    )

    metadata = load_metadata(index_dir)
    embedding_model_name = metadata["embedding_model"]

    print(f"Leyendo preguntas desde: {questions_path}")
    questions_df = pd.read_csv(questions_path)

    print(f"Leyendo qrels desde: {qrels_path}")
    qrels_df = pd.read_csv(qrels_path)

    print(f"Leyendo chunks desde: {chunks_path}")
    chunks_df = pd.read_csv(chunks_path)

    print(f"Leyendo embeddings desde: {embeddings_path}")
    chunk_embeddings = np.load(embeddings_path).astype("float32")

    if len(chunks_df) != chunk_embeddings.shape[0]:
        raise ValueError(
            f"Chunks ({len(chunks_df)}) y embeddings ({chunk_embeddings.shape[0]}) "
            "no coinciden."
        )

    gold_map = build_gold_map(qrels_df)

    print(f"Modelo de embeddings: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    results = []

    for _, row in tqdm(
        questions_df.iterrows(),
        total=len(questions_df),
        desc=f"Evaluando retrieval top-{top_k}",
    ):
        question_id = clean_text(row["id"])
        query = get_query(row)
        gold_chunk_ids = gold_map.get(question_id, set())

        retrieved = retrieve_top_k(
            query=query,
            model=model,
            chunk_embeddings=chunk_embeddings,
            chunks_df=chunks_df,
            top_k=top_k,
        )

        result = evaluate_query(
            question_id=question_id,
            query=query,
            gold_chunk_ids=gold_chunk_ids,
            retrieved=retrieved,
            top_k=top_k,
        )

        for col in ["topic", "hotpot_type", "level", "expected_answer", "original_question"]:
            if col in row.index:
                result[col] = clean_text(row.get(col, ""))

        results.append(result)

    results_df = pd.DataFrame(results)
    summary = summarize(results_df, top_k=top_k)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "hotpotqa_mini_retrieval_results.csv"
    summary_path = output_dir / "hotpotqa_mini_retrieval_summary.json"

    results_df.to_csv(results_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nEvaluación terminada.")
    print(f"Resultados por pregunta: {results_path}")
    print(f"Resumen: {summary_path}")

    print("\nResumen principal")
    print("-----------------")
    print(f"Preguntas: {summary['n_questions']}")
    print(f"Hit@{top_k}: {summary[f'hit@{top_k}']:.3f}")
    print(f"Recall@{top_k}: {summary[f'recall@{top_k}']:.3f}")
    print(f"Precision@{top_k}: {summary[f'precision@{top_k}']:.3f}")
    print(f"MRR@{top_k}: {summary[f'mrr@{top_k}']:.3f}")
    print(f"Gold chunks promedio: {summary['avg_n_gold']:.3f}")

    if "by_topic" in summary:
        print("\nPor topic:")
        for topic, metrics in summary["by_topic"].items():
            print(
                f"- {topic}: n={metrics['n']}, "
                f"Hit@{top_k}={metrics[f'hit@{top_k}']:.3f}, "
                f"Recall@{top_k}={metrics[f'recall@{top_k}']:.3f}, "
                f"MRR@{top_k}={metrics[f'mrr@{top_k}']:.3f}"
            )

    if "by_hotpot_type" in summary:
        print("\nPor tipo HotpotQA:")
        for hotpot_type, metrics in summary["by_hotpot_type"].items():
            print(
                f"- {hotpot_type}: n={metrics['n']}, "
                f"Hit@{top_k}={metrics[f'hit@{top_k}']:.3f}, "
                f"Recall@{top_k}={metrics[f'recall@{top_k}']:.3f}, "
                f"MRR@{top_k}={metrics[f'mrr@{top_k}']:.3f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa el retriever de S1 contra qrels_s1.csv."
    )

    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )

    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=DEFAULT_QRELS_PATH,
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k debe ser mayor que 0.")

    run_evaluation(
        questions_path=args.questions_path,
        qrels_path=args.qrels_path,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

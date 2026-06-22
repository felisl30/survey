#!/usr/bin/env python3
"""
evaluate_mc_rag_retrieval.py

Evalúa retrieval sobre el índice MC-RAG.

No llama a OpenAI. Usa el mismo modelo de embeddings usado para construir el índice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    qrels = pd.read_csv(qrels_path)

    required = {"question_id", "doc_id"}
    missing = required - set(qrels.columns)
    if missing:
        raise ValueError(f"Faltan columnas en qrels: {sorted(missing)}")

    out: dict[str, set[str]] = {}

    for _, row in qrels.iterrows():
        qid = clean_str(row["question_id"])
        doc_id = clean_str(row["doc_id"])
        out.setdefault(qid, set()).add(doc_id)

    return out


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/questions.csv"),
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/qrels.csv"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes/eval_mc/musique_mc_rag"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/retrieval"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1,3,5,8,10",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    chunks_path = args.index_dir / "chunks.csv"
    embeddings_path = args.index_dir / "embeddings.npy"

    if not args.questions_path.exists():
        raise FileNotFoundError(args.questions_path)
    if not args.qrels_path.exists():
        raise FileNotFoundError(args.qrels_path)
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    top_k_values = [int(x.strip()) for x in args.top_k_values.split(",") if x.strip()]
    max_k = max(top_k_values)

    questions = pd.read_csv(args.questions_path)
    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    chunks = pd.read_csv(chunks_path)
    embeddings = np.load(embeddings_path).astype("float32")
    qrels = load_qrels(args.qrels_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks y embeddings no coinciden: {len(chunks)} vs {embeddings.shape[0]}"
        )

    if "retrieval_query" not in questions.columns:
        raise ValueError("questions.csv debe tener columna retrieval_query.")

    if "doc_id" not in chunks.columns:
        raise ValueError("chunks.csv debe tener columna doc_id.")

    if "question_id" not in chunks.columns:
        raise ValueError("chunks.csv debe tener columna question_id.")

    print(f"Preguntas: {len(questions)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Top-k values: {top_k_values}")
    print(f"Modelo embeddings: {args.model_name}")
    print("Cargando modelo...")

    model = SentenceTransformer(args.model_name)

    print("Embebiendo queries...")
    start = time.time()
    query_embeddings = model.encode(
        questions["retrieval_query"].astype(str).tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    query_embeddings = normalize_embeddings(query_embeddings)
    query_latency_s = round(time.time() - start, 3)

    print("Calculando similitudes...")
    scores = query_embeddings @ embeddings.T

    detailed_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for q_idx, (_, qrow) in enumerate(questions.iterrows()):
        qid = clean_str(qrow["question_id"])
        relevant = qrels.get(qid, set())

        order = np.argsort(-scores[q_idx])[:max_k]

        retrieved = []
        for rank, chunk_idx in enumerate(order, start=1):
            chunk = chunks.iloc[int(chunk_idx)]
            doc_id = clean_str(chunk["doc_id"])
            chunk_qid = clean_str(chunk["question_id"])
            is_relevant = doc_id in relevant

            retrieved.append(doc_id)

            detailed_rows.append({
                "question_id": qid,
                "rank": rank,
                "retrieved_doc_id": doc_id,
                "retrieved_chunk_question_id": chunk_qid,
                "is_relevant": is_relevant,
                "same_question": chunk_qid == qid,
                "score": float(scores[q_idx, int(chunk_idx)]),
                "query": clean_str(qrow.get("retrieval_query", "")),
                "retrieved_title": clean_str(chunk.get("title", "")),
                "retrieved_text": clean_str(chunk.get("text", ""))[:500],
            })

        for k in top_k_values:
            top_docs = retrieved[:k]
            relevant_in_top_k = sum(1 for d in top_docs if d in relevant)

            top_rows = detailed_rows[-max_k:][:k]
            same_question_count = sum(1 for r in top_rows if r["same_question"])

            metric_rows.append({
                "question_id": qid,
                "k": k,
                "n_relevant_total": len(relevant),
                "n_relevant_retrieved": relevant_in_top_k,
                "hit_at_k": relevant_in_top_k > 0,
                "recall_at_k": relevant_in_top_k / len(relevant) if relevant else 0.0,
                "same_question_rate_at_k": same_question_count / k,
                "mrr_at_k": reciprocal_rank(top_docs, relevant),
            })

    detailed_df = pd.DataFrame(detailed_rows)
    metrics_df = pd.DataFrame(metric_rows)

    summary_rows = []
    for k, subset in metrics_df.groupby("k"):
        summary_rows.append({
            "k": int(k),
            "n_questions": int(len(subset)),
            "hit_at_k": float(subset["hit_at_k"].mean()),
            "recall_at_k": float(subset["recall_at_k"].mean()),
            "same_question_rate_at_k": float(subset["same_question_rate_at_k"].mean()),
            "mrr_at_k": float(subset["mrr_at_k"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("k")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = args.output_dir / "retrieval_detailed.csv"
    metrics_path = args.output_dir / "retrieval_metrics_by_question.csv"
    summary_path = args.output_dir / "retrieval_summary.csv"
    metadata_path = args.output_dir / "retrieval_metadata.json"

    detailed_df.to_csv(detailed_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "questions_path": str(args.questions_path),
        "qrels_path": str(args.qrels_path),
        "index_dir": str(args.index_dir),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "model_name": args.model_name,
        "n_questions": int(len(questions)),
        "n_chunks": int(len(chunks)),
        "embedding_dim": int(embeddings.shape[1]),
        "query_latency_s": query_latency_s,
        "top_k_values": top_k_values,
        "detailed_path": str(detailed_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nRetrieval evaluation")
    print("--------------------")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed: {detailed_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

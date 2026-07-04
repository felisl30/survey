#!/usr/bin/env python3
"""
build_mc_robustness_dataset.py

Construye un benchmark de robustez a partir de un dataset MC-RAG existente.

Entrada esperada:
    data/eval_mc/musique_mc_rag/questions.csv
    data/eval_mc/musique_mc_rag/corpus.csv
    data/eval_mc/musique_mc_rag/qrels.csv

Salida:
    data/eval_mc/robustness_musique/questions.csv
    data/eval_mc/robustness_musique/qrels.csv
    data/eval_mc/robustness_musique/corpus_clean.csv
    data/eval_mc/robustness_musique/corpus_noisy.csv
    data/eval_mc/robustness_musique/corpus_adversarial.csv
    data/eval_mc/robustness_musique/build_summary.json

Condiciones:
    clean:
        corpus original.

    noisy:
        corpus original + distractores aleatorios de otras preguntas.

    adversarial:
        corpus original + distractores semanticamente parecidos de otras preguntas.

Este script NO llama a OpenAI.
Solo usa pandas, numpy y sentence-transformers.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


REQUIRED_QUESTION_COLS = {
    "id",
    "question_id",
    "original_question",
    "retrieval_query",
    "question",
    "A",
    "B",
    "C",
    "D",
    "gold_answer",
}

REQUIRED_CORPUS_COLS = {
    "id",
    "doc_id",
    "document_id",
    "question_id",
    "source_question_id",
    "source_dataset",
    "benchmark_name",
    "rank_in_source",
    "title",
    "text",
}

REQUIRED_QRELS_COLS = {
    "question_id",
    "doc_id",
    "document_id",
    "relevance",
    "rank_in_source",
}


def clean_str(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def check_required_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} no tiene las columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


def add_condition_metadata(corpus: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = corpus.copy()

    out["benchmark_name"] = f"robustness_musique_{condition}"
    out["robustness_condition"] = condition
    out["context_role"] = "gold"
    out["is_gold_context"] = True
    out["distractor_type"] = ""
    out["distractor_source_doc_id"] = ""
    out["distractor_source_question_id"] = ""

    return out


def make_distractor_row(
    *,
    source_row: pd.Series,
    target_question_id: str,
    condition: str,
    distractor_rank: int,
    score: float | None = None,
) -> dict[str, Any]:
    source_doc_id = clean_str(source_row.get("doc_id", ""))
    source_question_id = clean_str(source_row.get("question_id", ""))

    new_doc_id = (
        f"{target_question_id}__{condition}_distractor_"
        f"{distractor_rank:02d}__from__{source_doc_id}"
    )

    row = {
        "id": new_doc_id,
        "doc_id": new_doc_id,
        "document_id": new_doc_id,
        "question_id": target_question_id,
        "source_question_id": source_question_id,
        "source_dataset": clean_str(source_row.get("source_dataset", "")),
        "benchmark_name": f"robustness_musique_{condition}",
        "rank_in_source": 1000 + distractor_rank,
        "title": clean_str(source_row.get("title", "")),
        "text": clean_str(source_row.get("text", "")),
        "robustness_condition": condition,
        "context_role": "distractor",
        "is_gold_context": False,
        "distractor_type": condition,
        "distractor_source_doc_id": source_doc_id,
        "distractor_source_question_id": source_question_id,
    }

    if score is not None:
        row["adversarial_similarity_score"] = float(score)

    return row


def build_noisy_corpus(
    *,
    corpus: pd.DataFrame,
    questions: pd.DataFrame,
    n_distractors_per_question: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    gold = add_condition_metadata(corpus, "noisy")
    distractor_rows: list[dict[str, Any]] = []

    for _, qrow in questions.iterrows():
        qid = clean_str(qrow["question_id"])

        pool = corpus[corpus["question_id"].astype(str) != qid]

        if len(pool) < n_distractors_per_question:
            raise ValueError(
                f"No hay suficientes distractores aleatorios para {qid}. "
                f"Pool={len(pool)}, requeridos={n_distractors_per_question}"
            )

        selected_indices = rng.sample(list(pool.index), n_distractors_per_question)

        for rank, idx in enumerate(selected_indices):
            distractor_rows.append(
                make_distractor_row(
                    source_row=corpus.loc[idx],
                    target_question_id=qid,
                    condition="noisy",
                    distractor_rank=rank,
                )
            )

    distractors = pd.DataFrame(distractor_rows)

    return pd.concat([gold, distractors], ignore_index=True)


def build_adversarial_corpus(
    *,
    corpus: pd.DataFrame,
    questions: pd.DataFrame,
    n_distractors_per_question: int,
    embedding_model_name: str,
    batch_size: int,
) -> pd.DataFrame:
    gold = add_condition_metadata(corpus, "adversarial")

    work = corpus.copy().reset_index(drop=True)

    work["embed_text"] = (
        work["title"].fillna("").astype(str).str.strip()
        + "\n\n"
        + work["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    queries = questions["retrieval_query"].fillna("").astype(str).tolist()
    docs = work["embed_text"].tolist()

    print()
    print("Construyendo distractores adversariales")
    print("--------------------------------------")
    print(f"Modelo de embeddings: {embedding_model_name}")
    print(f"Preguntas: {len(queries)}")
    print(f"Documentos candidatos: {len(docs)}")

    model = SentenceTransformer(embedding_model_name)

    print()
    print("Embebiendo preguntas...")
    query_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    query_embeddings = normalize_embeddings(query_embeddings)

    print()
    print("Embebiendo documentos...")
    doc_embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    doc_embeddings = normalize_embeddings(doc_embeddings)

    print()
    print("Calculando similitudes pregunta-documento...")
    scores = query_embeddings @ doc_embeddings.T

    distractor_rows: list[dict[str, Any]] = []

    for q_idx, (_, qrow) in enumerate(questions.iterrows()):
        qid = clean_str(qrow["question_id"])

        order = np.argsort(-scores[q_idx])

        picked = []

        for doc_pos in order:
            source = work.iloc[int(doc_pos)]
            source_qid = clean_str(source.get("question_id", ""))

            if source_qid == qid:
                continue

            picked.append((int(doc_pos), source))

            if len(picked) >= n_distractors_per_question:
                break

        if len(picked) < n_distractors_per_question:
            raise ValueError(
                f"No se pudieron elegir suficientes distractores adversariales "
                f"para {qid}. Elegidos={len(picked)}"
            )

        for rank, (doc_pos, source_row) in enumerate(picked):
            distractor_rows.append(
                make_distractor_row(
                    source_row=source_row,
                    target_question_id=qid,
                    condition="adversarial",
                    distractor_rank=rank,
                    score=float(scores[q_idx, doc_pos]),
                )
            )

    distractors = pd.DataFrame(distractor_rows)

    return pd.concat([gold, distractors], ignore_index=True)


def summarize_corpus(df: pd.DataFrame) -> dict[str, Any]:
    out = {
        "n_rows": int(len(df)),
        "n_unique_questions": int(df["question_id"].nunique()),
        "n_unique_doc_ids": int(df["doc_id"].nunique()),
        "n_gold_contexts": int(df["is_gold_context"].astype(bool).sum())
        if "is_gold_context" in df.columns
        else None,
        "n_distractors": int((~df["is_gold_context"].astype(bool)).sum())
        if "is_gold_context" in df.columns
        else None,
        "context_role_counts": df["context_role"].value_counts(dropna=False).to_dict()
        if "context_role" in df.columns
        else {},
        "distractor_type_counts": df["distractor_type"].value_counts(dropna=False).to_dict()
        if "distractor_type" in df.columns
        else {},
    }

    if "adversarial_similarity_score" in df.columns:
        scores = pd.to_numeric(df["adversarial_similarity_score"], errors="coerce").dropna()

        if len(scores):
            out["adversarial_similarity_score"] = {
                "mean": float(scores.mean()),
                "min": float(scores.min()),
                "max": float(scores.max()),
            }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/questions.csv"),
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/corpus.csv"),
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/qrels.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_mc/robustness_musique"),
    )
    parser.add_argument(
        "--random-distractors-per-question",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--adversarial-distractors-per-question",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"{args.output_dir} ya existe y no está vacío. "
            f"Usá --force para sobrescribir."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Leyendo archivos base")
    print("--------------------")
    print(f"Questions: {args.questions_path}")
    print(f"Corpus:    {args.corpus_path}")
    print(f"Qrels:     {args.qrels_path}")

    questions = pd.read_csv(args.questions_path)
    corpus = pd.read_csv(args.corpus_path)
    qrels = pd.read_csv(args.qrels_path)

    check_required_columns(questions, REQUIRED_QUESTION_COLS, "questions.csv")
    check_required_columns(corpus, REQUIRED_CORPUS_COLS, "corpus.csv")
    check_required_columns(qrels, REQUIRED_QRELS_COLS, "qrels.csv")

    print()
    print("Validación inicial")
    print("------------------")
    print(f"Preguntas: {len(questions)}")
    print(f"Corpus original: {len(corpus)}")
    print(f"Qrels: {len(qrels)}")
    print(f"Questions únicas: {questions['question_id'].nunique()}")
    print(f"Doc IDs únicos: {corpus['doc_id'].nunique()}")

    print()
    print("Construyendo corpus clean...")
    clean_corpus = add_condition_metadata(corpus, "clean")

    print("Construyendo corpus noisy...")
    noisy_corpus = build_noisy_corpus(
        corpus=corpus,
        questions=questions,
        n_distractors_per_question=args.random_distractors_per_question,
        seed=args.seed,
    )

    print("Construyendo corpus adversarial...")
    adversarial_corpus = build_adversarial_corpus(
        corpus=corpus,
        questions=questions,
        n_distractors_per_question=args.adversarial_distractors_per_question,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.batch_size,
    )

    questions_out = args.output_dir / "questions.csv"
    qrels_out = args.output_dir / "qrels.csv"
    clean_out = args.output_dir / "corpus_clean.csv"
    noisy_out = args.output_dir / "corpus_noisy.csv"
    adversarial_out = args.output_dir / "corpus_adversarial.csv"
    summary_out = args.output_dir / "build_summary.json"

    questions.to_csv(questions_out, index=False)
    qrels.to_csv(qrels_out, index=False)
    clean_corpus.to_csv(clean_out, index=False)
    noisy_corpus.to_csv(noisy_out, index=False)
    adversarial_corpus.to_csv(adversarial_out, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "questions_path": str(args.questions_path),
            "corpus_path": str(args.corpus_path),
            "qrels_path": str(args.qrels_path),
            "n_questions": int(len(questions)),
            "n_corpus_rows": int(len(corpus)),
            "n_qrels_rows": int(len(qrels)),
        },
        "output": {
            "output_dir": str(args.output_dir),
            "questions_path": str(questions_out),
            "qrels_path": str(qrels_out),
            "corpus_clean_path": str(clean_out),
            "corpus_noisy_path": str(noisy_out),
            "corpus_adversarial_path": str(adversarial_out),
        },
        "parameters": {
            "random_distractors_per_question": args.random_distractors_per_question,
            "adversarial_distractors_per_question": args.adversarial_distractors_per_question,
            "embedding_model_name": args.embedding_model_name,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "conditions": {
            "clean": summarize_corpus(clean_corpus),
            "noisy": summarize_corpus(noisy_corpus),
            "adversarial": summarize_corpus(adversarial_corpus),
        },
    }

    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Dataset robusto creado")
    print("======================")
    print(f"Output dir: {args.output_dir}")
    print()
    print("Archivos:")
    print(f"- {questions_out}")
    print(f"- {qrels_out}")
    print(f"- {clean_out}")
    print(f"- {noisy_out}")
    print(f"- {adversarial_out}")
    print(f"- {summary_out}")
    print()
    print("Resumen por condición:")
    for condition, info in summary["conditions"].items():
        print()
        print(f"[{condition}]")
        for key, value in info.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_s1_mc_rag.py

Corre S1 RAG clásico sobre un benchmark multiple-choice convertido a RAG.

Entrada:
    data/eval_mc/musique_mc_rag/questions.csv
    indexes/eval_mc/musique_mc_rag/chunks.csv
    indexes/eval_mc/musique_mc_rag/embeddings.npy

Salida:
    outputs/eval_mc/musique_mc_rag/s1/s1_gpt_5_mini_top5_raw.csv

Este script sí llama a la API de OpenAI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()


SYSTEM_PROMPT = """
Sos un sistema experimental de pregunta-respuesta con RAG clásico.

Tu tarea:
- Responder preguntas de opción múltiple usando el contexto recuperado.
- Elegir una única opción entre A, B, C o D.
- No explicar la respuesta.
- Devolver únicamente JSON válido.
- El JSON debe tener esta forma exacta:
  {
    "answer": "A",
    "confidence": 0.0
  }

Reglas:
- Si el contexto recuperado contiene la evidencia necesaria, priorizá esa evidencia.
- Si hay conflicto entre conocimiento interno y contexto, priorizá el contexto.
- Si el contexto es insuficiente, elegí la mejor opción disponible, pero mantené el formato JSON.
- No devuelvas texto fuera del JSON.
""".strip()


COLUMNS_TO_KEEP_IF_PRESENT = [
    "id",
    "question_id",
    "dataset",
    "case_type",
    "source_dataset",
    "benchmark_name",
    "original_question",
    "question",
    "retrieval_query",
    "prompt",
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_text",
    "expected_answer",
    "expected_route",
    "requires_retrieval",
    "original_source",
    "source_split",
    "difficulty",
    "frozen_input_sha256",
]


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())

    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())

    return str(value)


def extract_usage_fields(usage: Any) -> dict[str, int | None]:
    usage_json = jsonable(usage)

    if not isinstance(usage_json, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = usage_json.get("input_tokens")
    output_tokens = usage_json.get("output_tokens")
    total_tokens = usage_json.get("total_tokens")

    if input_tokens is None:
        input_tokens = usage_json.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage_json.get("completion_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY. Revisá tu archivo .env o exportá la variable."
        )

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)

    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo existente {output_path} no tiene columna 'id'. "
            "No puedo usar --resume de forma segura."
        )

    return existing


def build_context_block(retrieved_rows: pd.DataFrame) -> str:
    blocks = []

    for i, (_, row) in enumerate(retrieved_rows.iterrows(), start=1):
        title = clean_str(row.get("title", ""))
        text = clean_str(row.get("text", ""))

        blocks.append(
            f"[{i}] Título: {title}\n"
            f"Texto: {text}"
        )

    return "\n\n".join(blocks)


def build_final_prompt(row: pd.Series, retrieved_rows: pd.DataFrame) -> str:
    context_block = build_context_block(retrieved_rows)
    question_prompt = clean_str(row.get("prompt", "")) or clean_str(row.get("question", ""))

    return f"""Contexto recuperado:
{context_block}

Pregunta:
{question_prompt}
"""


def retrieve_for_query(
    *,
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, list[float], float]:
    start = time.time()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    query_embedding = normalize_embeddings(query_embedding)

    scores = (query_embedding @ embeddings.T)[0]
    order = np.argsort(-scores)[:top_k]

    retrieval_latency = round(time.time() - start, 4)

    retrieved_rows = chunks.iloc[order].copy()
    retrieved_scores = [float(scores[int(idx)]) for idx in order]

    return retrieved_rows, retrieved_scores, retrieval_latency


def call_llm(
    *,
    client: Any,
    model_name: str,
    prompt: str,
    max_retries: int,
    retry_base_seconds: float = 2.0,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        start = time.time()

        try:
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_PROMPT,
                input=prompt,
            )

            latency_seconds = round(time.time() - start, 3)
            raw_output = getattr(response, "output_text", "")

            usage = getattr(response, "usage", None)
            usage_json = jsonable(usage)
            usage_fields = extract_usage_fields(usage)

            return {
                "model": model_name,
                "raw_output": raw_output,
                "generation_latency_seconds": latency_seconds,
                "usage_json": json.dumps(usage_json, ensure_ascii=False),
                "input_tokens": usage_fields["input_tokens"],
                "output_tokens": usage_fields["output_tokens"],
                "total_tokens": usage_fields["total_tokens"],
            }

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            sleep_seconds = retry_base_seconds * (2 ** attempt)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Falló el llamado al modelo luego de {max_retries + 1} intentos: {last_error}")


def build_output_row(
    *,
    input_row: pd.Series,
    model_name: str,
    top_k: int,
    retrieved_rows: pd.DataFrame,
    retrieved_scores: list[float],
    retrieval_latency_seconds: float,
    model_result: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for col in COLUMNS_TO_KEEP_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    output["system"] = "S1_classic_rag"
    output["model"] = model_name
    output["top_k"] = top_k

    output["retrieval_latency_seconds"] = retrieval_latency_seconds
    output["n_docs_retrieved"] = len(retrieved_rows)

    output["retrieved_doc_ids_json"] = json.dumps(
        retrieved_rows["doc_id"].astype(str).tolist(),
        ensure_ascii=False,
    )
    output["retrieved_titles_json"] = json.dumps(
        retrieved_rows["title"].fillna("").astype(str).tolist(),
        ensure_ascii=False,
    )
    output["retrieved_scores_json"] = json.dumps(
        retrieved_scores,
        ensure_ascii=False,
    )
    output["retrieved_context_json"] = json.dumps(
        [
            {
                "doc_id": clean_str(r.get("doc_id", "")),
                "title": clean_str(r.get("title", "")),
                "text": clean_str(r.get("text", "")),
                "score": retrieved_scores[i],
            }
            for i, (_, r) in enumerate(retrieved_rows.iterrows())
        ],
        ensure_ascii=False,
    )

    if model_result is None:
        output["raw_output"] = ""
        output["generation_latency_seconds"] = None
        output["latency_seconds"] = retrieval_latency_seconds
        output["usage_json"] = ""
        output["input_tokens"] = None
        output["output_tokens"] = None
        output["total_tokens"] = None
    else:
        output["raw_output"] = model_result.get("raw_output", "")
        output["generation_latency_seconds"] = model_result.get("generation_latency_seconds")
        output["latency_seconds"] = round(
            retrieval_latency_seconds + float(model_result.get("generation_latency_seconds") or 0),
            3,
        )
        output["usage_json"] = model_result.get("usage_json", "")
        output["input_tokens"] = model_result.get("input_tokens")
        output["output_tokens"] = model_result.get("output_tokens")
        output["total_tokens"] = model_result.get("total_tokens")

    output["error"] = error

    return output


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/questions.csv"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes/eval_mc/musique_mc_rag"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s1/s1_gpt_5_mini_top5_raw.csv"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    chunks_path = args.index_dir / "chunks.csv"
    embeddings_path = args.index_dir / "embeddings.npy"

    if not args.questions_path.exists():
        raise FileNotFoundError(args.questions_path)
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    questions = pd.read_csv(args.questions_path)
    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    chunks = pd.read_csv(chunks_path)
    embeddings = np.load(embeddings_path).astype("float32")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks.csv y embeddings.npy no coinciden: {len(chunks)} vs {embeddings.shape[0]}"
        )

    existing = load_existing_results(args.output_path) if args.resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending = questions[~questions["id"].astype(str).isin(existing_ids)].copy()

    print(f"Questions path: {args.questions_path}")
    print(f"Index dir: {args.index_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Modelo generación: {args.model}")
    print(f"Modelo embeddings: {args.embedding_model}")
    print(f"Top-k: {args.top_k}")
    print(f"Filas consideradas: {len(questions)}")
    print(f"Filas ya existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending)}")

    print("\nCargando modelo de embeddings...")
    embedding_model = SentenceTransformer(args.embedding_model)

    print("Inicializando cliente OpenAI...")
    client = get_openai_client()

    for i, (_, row) in enumerate(
        tqdm(
            pending.iterrows(),
            total=len(pending),
            desc="Running S1 MC RAG",
        ),
        start=1,
    ):
        retrieved_rows = pd.DataFrame()
        retrieved_scores: list[float] = []
        retrieval_latency_seconds = 0.0

        try:
            retrieval_query = clean_str(row.get("retrieval_query", "")) or clean_str(row.get("original_question", ""))

            retrieved_rows, retrieved_scores, retrieval_latency_seconds = retrieve_for_query(
                query=retrieval_query,
                model=embedding_model,
                embeddings=embeddings,
                chunks=chunks,
                top_k=args.top_k,
            )

            final_prompt = build_final_prompt(row, retrieved_rows)

            model_result = call_llm(
                client=client,
                model_name=args.model,
                prompt=final_prompt,
                max_retries=args.max_retries,
            )

            error = ""

        except Exception as exc:
            model_result = None
            error = str(exc)

        output_row = build_output_row(
            input_row=row,
            model_name=args.model,
            top_k=args.top_k,
            retrieved_rows=retrieved_rows,
            retrieved_scores=retrieved_scores,
            retrieval_latency_seconds=retrieval_latency_seconds,
            model_result=model_result,
            error=error,
        )

        rows.append(output_row)

        if args.save_every > 0 and i % args.save_every == 0:
            save_results(rows, args.output_path)

    save_results(rows, args.output_path)

    print(f"\nResultados S1 raw guardados en: {args.output_path}")


if __name__ == "__main__":
    main()

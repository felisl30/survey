#!/usr/bin/env python3
"""
run_s1_rag.py

Ejecuta S1: RAG básico sobre HotpotQA-mini.

Entrada esperada:
    data/s1/hotpotqa_mini/questions_s1.csv

Índice esperado:
    indexes/s1/hotpotqa_mini/chunks.csv
    indexes/s1/hotpotqa_mini/embeddings.npy
    indexes/s1/hotpotqa_mini/metadata.json

Salida por defecto:
    outputs/s1/generation/hotpotqa_mini_s1_raw.csv

Este script guarda respuestas crudas. El parseo y la evaluación deben hacerse
en pasos posteriores para no repetir llamados al modelo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


# Permite ejecutar:
#   python s1_model_code/run_s1_rag.py
# e importar módulos ubicados en la raíz del proyecto y en s1_model_code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
S1_CODE_DIR = Path(__file__).resolve().parent

for path in [PROJECT_ROOT, S1_CODE_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from direct_llm import ask_direct_llm_with_metadata  # noqa: E402
from retriever_s1 import S1Retriever, clean_text, get_retrieval_query, read_selected_top_k  # noqa: E402


DEFAULT_INPUT_PATH = Path("data/s1/hotpotqa_mini/questions_s1.csv")
DEFAULT_INDEX_DIR = Path("indexes/s1/hotpotqa_mini")
DEFAULT_OUTPUT_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_raw.csv")
DEFAULT_SELECTED_TOP_K_PATH = Path("outputs/s1/selected_top_k.txt")
DEFAULT_TOP_K = 5


REQUIRED_COLUMNS = {
    "id",
    "dataset",
    "case_type",
}


COLUMNS_TO_KEEP_IF_PRESENT = [
    "id",
    "dataset",
    "case_type",
    "topic",
    "hotpot_type",
    "level",
    "source",
    "source_split",
    "original_hotpotqa_id",
    "original_question",
    "question",
    "prompt",
    "expected_answer",
    "gold_answer",
    "gold_answer_text",
    "expected_behavior",
    "correct_answers_json",
    "incorrect_answers_json",
    "gold_evidence_ids",
    "gold_evidence_titles",
    "context_chunk_ids",
    "expected_model_output_format",
]


S1_SYSTEM_PROMPT = """
Sos un asistente de pregunta-respuesta usado como sistema experimental S1.

Tu tarea:
- Responder usando únicamente el contexto recuperado incluido en el prompt.
- No usar conocimiento externo cuando el contexto no alcance.
- No inventar fuentes, citas ni datos.
- Si el contexto recuperado no permite responder, indicá que no hay información suficiente.
- Si la pregunta es ambigua, indicá la aclaración mínima necesaria.
- Respetar estrictamente el formato de salida solicitado.
- Si se pide JSON, devolvé únicamente JSON válido.

Este sistema representa S1: RAG básico con recuperación top-k fija.
""".strip()


def validate_input(df: pd.DataFrame) -> None:
    """Valida columnas mínimas para correr S1."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(missing))
        )

    if "original_question" not in df.columns and "question" not in df.columns:
        raise ValueError(
            "El CSV debe tener al menos una de estas columnas: "
            "original_question o question."
        )


def truncate_text(text: str, max_chars: int) -> str:
    """Trunca texto largo conservando un marcador claro."""
    text = clean_text(text)

    if max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " [...]"


def build_context_block(
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int,
) -> str:
    """Construye el bloque de contexto que se pasará al LLM."""
    blocks: list[str] = []

    for item in retrieved_chunks:
        rank = item.get("rank", "")
        chunk_id = clean_text(item.get("chunk_id", ""))
        title = clean_text(item.get("title", ""))
        score = item.get("score", "")
        text = truncate_text(clean_text(item.get("text", "")), max_chars_per_chunk)

        try:
            score_text = f"{float(score):.6f}"
        except (TypeError, ValueError):
            score_text = ""

        blocks.append(
            f"""[{rank}]
chunk_id: {chunk_id}
title: {title}
score: {score_text}
text: {text}"""
        )

    return "\n\n".join(blocks)


def build_rag_prompt(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_chars_per_chunk: int,
) -> str:
    """Construye el prompt final de S1 con contexto recuperado."""
    context_block = build_context_block(
        retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    return f"""Respondé la siguiente pregunta usando únicamente el contexto recuperado.

Reglas:
- Usá solo la información del contexto recuperado.
- No agregues información externa.
- Si el contexto recuperado no alcanza para responder, indicá que no hay información suficiente.
- Respondé de forma breve.
- Respondé únicamente con JSON válido.
- El campo "answer" debe contener tu respuesta final.
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser true si no hay evidencia suficiente; si no, false.

Contexto recuperado:
{context_block}

Pregunta:
{question}

Formato obligatorio:
{{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false
}}"""


def retrieval_to_json(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Serializa retrieval a JSON guardable en CSV."""
    safe_items: list[dict[str, Any]] = []

    for item in retrieved_chunks:
        safe_items.append(
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "score": item.get("score"),
                "text": item.get("text"),
                "source": item.get("source", ""),
                "topic": item.get("topic", ""),
                "question_id": item.get("question_id", ""),
                "paragraph_index": item.get("paragraph_index", ""),
            }
        )

    return json.dumps(safe_items, ensure_ascii=False)


def join_retrieved_field(
    retrieved_chunks: list[dict[str, Any]],
    field: str,
    *,
    score_format: bool = False,
) -> str:
    values: list[str] = []

    for item in retrieved_chunks:
        value = item.get(field, "")

        if score_format:
            try:
                values.append(f"{float(value):.6f}")
            except (TypeError, ValueError):
                values.append("")
        else:
            values.append(clean_text(value))

    return "|".join(values)


def load_existing_results(output_path: Path) -> pd.DataFrame:
    """Carga resultados previos si existen."""
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)

    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo de salida existente {output_path} no tiene columna 'id'. "
            "No puedo usar --resume de forma segura."
        )

    return existing


def build_result_row(
    input_row: pd.Series,
    *,
    system_name: str,
    model: str | None,
    top_k: int,
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    rag_prompt: str,
    retrieval_latency_seconds: float,
    model_result: dict[str, Any] | None,
    error: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Construye una fila de salida conservando metadatos útiles."""
    output: dict[str, Any] = {}

    for col in COLUMNS_TO_KEEP_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    output["system"] = system_name
    output["model"] = model or ""
    output["top_k"] = top_k
    output["retrieval_query"] = query
    output["retrieved_chunk_ids"] = join_retrieved_field(retrieved_chunks, "chunk_id")
    output["retrieved_doc_ids"] = join_retrieved_field(retrieved_chunks, "doc_id")
    output["retrieved_titles"] = join_retrieved_field(retrieved_chunks, "title")
    output["retrieved_scores"] = join_retrieved_field(
        retrieved_chunks,
        "score",
        score_format=True,
    )
    output["retrieved_context_json"] = retrieval_to_json(retrieved_chunks)
    output["rag_prompt"] = rag_prompt
    output["rag_prompt_chars"] = len(rag_prompt)
    output["retrieval_latency_seconds"] = round(retrieval_latency_seconds, 3)
    output["dry_run"] = dry_run

    if model_result is None:
        output["raw_output"] = ""
        output["latency_seconds"] = None
        output["usage_json"] = ""
        output["input_tokens"] = None
        output["output_tokens"] = None
        output["total_tokens"] = None
    else:
        output["model"] = model_result.get("model", model or "")
        output["raw_output"] = model_result.get("raw_output", "")
        output["latency_seconds"] = model_result.get("latency_seconds")
        output["usage_json"] = model_result.get("usage_json", "")
        output["input_tokens"] = model_result.get("input_tokens")
        output["output_tokens"] = model_result.get("output_tokens")
        output["total_tokens"] = model_result.get("total_tokens")

    output["error"] = error

    return output


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Guarda resultados acumulados."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_experiment(
    *,
    input_path: Path,
    index_dir: Path,
    output_path: Path,
    model: str | None,
    top_k: int | None,
    limit: int | None,
    resume: bool,
    save_every: int,
    max_retries: int,
    max_chars_per_chunk: int,
    dry_run: bool,
) -> pd.DataFrame:
    """Corre S1 RAG básico y guarda respuestas crudas."""
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    selected_top_k = top_k if top_k is not None else read_selected_top_k(
        DEFAULT_SELECTED_TOP_K_PATH,
        default=DEFAULT_TOP_K,
    )

    if selected_top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    df = pd.read_csv(input_path)
    validate_input(df)

    if limit is not None:
        df = df.head(limit).copy()

    existing = load_existing_results(output_path) if resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending_df = df[~df["id"].astype(str).isin(existing_ids)].copy()

    retriever = S1Retriever(index_dir=index_dir)

    print(f"Archivo de entrada: {input_path}")
    print(f"Índice S1: {index_dir}")
    print(f"Archivo de salida: {output_path}")
    print(f"Top-k: {selected_top_k}")
    print(f"Modelo de embeddings: {retriever.embedding_model_name}")
    print(f"Modelo generador: {model or '<default direct_llm.py>'}")
    print(f"Dry run: {dry_run}")
    print(f"Filas totales consideradas: {len(df)}")
    print(f"Filas ya existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(
            pending_df.iterrows(),
            total=len(pending_df),
            desc="Running S1 Basic RAG",
        ),
        start=1,
    ):
        model_result: dict[str, Any] | None = None
        error = ""
        retrieved_chunks: list[dict[str, Any]] = []
        query = ""
        rag_prompt = ""
        retrieval_latency_seconds = 0.0

        try:
            query = get_retrieval_query(row)

            retrieval_start = time.time()
            retrieved_chunks = retriever.retrieve(query, top_k=selected_top_k)
            retrieval_latency_seconds = time.time() - retrieval_start

            rag_prompt = build_rag_prompt(
                question=query,
                retrieved_chunks=retrieved_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            )

            if dry_run:
                error = "DRY_RUN: no se llamó al modelo."
            else:
                model_result = ask_direct_llm_with_metadata(
                    rag_prompt,
                    model=model,
                    system_prompt=S1_SYSTEM_PROMPT,
                    max_retries=max_retries,
                )

        except Exception as exc:
            error = str(exc)

        result_row = build_result_row(
            row,
            system_name="S1_basic_rag",
            model=model,
            top_k=selected_top_k,
            query=query,
            retrieved_chunks=retrieved_chunks,
            rag_prompt=rag_prompt,
            retrieval_latency_seconds=retrieval_latency_seconds,
            model_result=model_result,
            error=error,
            dry_run=dry_run,
        )

        rows.append(result_row)

        if save_every > 0 and i % save_every == 0:
            save_results(rows, output_path)

    save_results(rows, output_path)

    print(f"\nResultados guardados en: {output_path}")
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corre S1 RAG básico sobre HotpotQA-mini."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV de preguntas S1.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio del índice S1.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV donde guardar respuestas crudas.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo generador. Si se omite, usa el default de direct_llm.py.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Chunks a recuperar. Si se omite, lee outputs/s1/selected_top_k.txt o usa 5.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Corre solo las primeras N filas. Útil para smoke tests.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="No repite IDs que ya existan en el archivo de salida.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Cada cuántas filas guardar resultados parciales.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Cantidad máxima de reintentos por pregunta.",
    )

    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=1600,
        help="Máximo de caracteres por chunk dentro del prompt RAG.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Arma retrieval y prompt, pero no llama al modelo.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_experiment(
        input_path=args.input_path,
        index_dir=args.index_dir,
        output_path=args.output_path,
        model=args.model,
        top_k=args.top_k,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        max_retries=args.max_retries,
        max_chars_per_chunk=args.max_chars_per_chunk,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

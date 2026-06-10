#!/usr/bin/env python3
"""
run_s2_adaptive_rag.py

Ejecuta S2: Adaptive-RAG.

Entrada esperada:
    data/s2/adaptive_rag/questions_s2.csv

Índice esperado:
    indexes/s2/adaptive_rag/chunks.csv
    indexes/s2/adaptive_rag/embeddings.npy
    indexes/s2/adaptive_rag/metadata.json

Salida por defecto:
    outputs/s2/generation/adaptive_rag_s2_raw.csv

Diseño
------
Para cada pregunta:
    1. Lee routing_question.
    2. Obtiene una ruta con router_s2.py o con un CSV de ruteo precomputado.
    3. Ejecuta la política correspondiente:
        - direct   -> LLM directo sin retrieval.
        - retrieve -> retrieval top-k + LLM con contexto.
        - abstain  -> respuesta determinística de abstención, sin LLM.
        - clarify  -> respuesta determinística de aclaración, sin LLM.
    4. Guarda decisión de router, chunks recuperados, prompt final, raw_output,
       tokens, latencias y errores.

Notas
-----
- Este script guarda respuestas crudas. El parseo y la evaluación deben hacerse
  en pasos posteriores para no repetir llamados al modelo.
- El retriever reutiliza S1Retriever, porque es genérico: lee chunks.csv,
  embeddings.npy y metadata.json desde el index_dir indicado.
- Para smoke tests se puede usar --dry-run, que arma router/retrieval/prompt
  pero no llama al LLM en rutas direct/retrieve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
S2_CODE_DIR = Path(__file__).resolve().parent
S1_CODE_DIR = PROJECT_ROOT / "s1_model_code"

for path in [PROJECT_ROOT, S2_CODE_DIR, S1_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from direct_llm import ask_direct_llm_with_metadata  # noqa: E402
from project_paths import S2_INDEX_DIR, S2_QUESTIONS_PATH, S2_RAW_OUTPUT_PATH  # noqa: E402
from router_s2 import route_question, route_is_acceptable  # noqa: E402
from retriever_s1 import S1Retriever, clean_text  # noqa: E402


DEFAULT_INPUT_PATH = S2_QUESTIONS_PATH
DEFAULT_INDEX_DIR = S2_INDEX_DIR
DEFAULT_OUTPUT_PATH = S2_RAW_OUTPUT_PATH
DEFAULT_TOP_K = 5

VALID_ROUTES = {"direct", "retrieve", "abstain", "clarify"}

REQUIRED_COLUMNS = {
    "id",
    "dataset",
    "case_type",
}

COLUMNS_TO_KEEP_IF_PRESENT = [
    # Identificación S2
    "id",
    "source_system",
    "source_question_id",
    "source_dataset",
    "dataset",
    "case_type",
    "s2_case_type",
    # Decisión esperada
    "expected_route",
    "acceptable_routes_json",
    "requires_retrieval",
    "retrieval_mode",
    "expected_behavior",
    "expected_final_behavior",
    # Entrada textual
    "routing_question",
    "original_question",
    "question",
    "prompt",
    "source_prompt",
    # Metadata
    "subject",
    "topic",
    "difficulty",
    "level",
    "hotpot_type",
    "source",
    "source_split",
    "original_hotpotqa_id",
    "truthfulqa_category",
    "truthfulqa_type",
    "original_source",
    # Opciones / labels
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_idx",
    "gold_answer_text",
    "expected_answer",
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",
    # Evidencia
    "gold_evidence_ids",
    "gold_evidence_titles",
    "context_chunk_ids",
    # Control experimental
    "is_synthetic",
    "synthetic_strategy",
    "expected_model_output_format",
    "evaluation_notes",
]

S2_DIRECT_SYSTEM_PROMPT = """
Sos un asistente de pregunta-respuesta usado como sistema experimental S2.

Estás ejecutando la ruta direct de Adaptive-RAG.

Tu tarea:
- Responder sin usar retrieval ni documentos externos.
- No decir que consultaste corpus, documentos, fuentes o bases de datos.
- No inventar citas ni referencias.
- Si la pregunta depende de un documento específico no provisto, indicá que no hay información suficiente.
- Si la pregunta es ambigua, pedí la aclaración mínima necesaria.
- Respetar estrictamente el formato JSON solicitado.

Este sistema representa S2 Adaptive-RAG cuando el router decide no recuperar memoria externa.
""".strip()

S2_RAG_SYSTEM_PROMPT = """
Sos un asistente de pregunta-respuesta usado como sistema experimental S2.

Estás ejecutando la ruta retrieve de Adaptive-RAG.

Tu tarea:
- Responder usando únicamente el contexto recuperado incluido en el prompt.
- No usar conocimiento externo cuando el contexto no alcance.
- No inventar fuentes, citas ni datos.
- Si el contexto recuperado no permite responder, indicá que no hay información suficiente.
- Si la pregunta es ambigua, indicá la aclaración mínima necesaria.
- Respetar estrictamente el formato JSON solicitado.

Este sistema representa S2 Adaptive-RAG cuando el router decide usar memoria externa.
""".strip()


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
    return False


def get_question_for_s2(row: pd.Series) -> str:
    """Pregunta limpia que usa el router/generador S2."""
    for col in ["routing_question", "original_question", "question", "prompt"]:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value
    raise ValueError(f"Fila id={row.get('id', '<sin id>')} sin pregunta válida.")


def normalize_route(value: Any) -> str:
    route = clean_text(value).lower()
    if route in VALID_ROUTES:
        return route
    return "direct"


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_mmlu_direct_prompt(row: pd.Series, question: str) -> str:
    choices = {
        "A": clean_text(row.get("A", "")),
        "B": clean_text(row.get("B", "")),
        "C": clean_text(row.get("C", "")),
        "D": clean_text(row.get("D", "")),
    }

    if all(choices.values()):
        return f"""Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- No expliques tu respuesta.
- No uses documentos externos.
- Respondé únicamente con JSON válido.
- El campo "answer" debe ser exactamente una de estas letras: "A", "B", "C" o "D".
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser false salvo que la pregunta sea imposible de responder.

Pregunta:
{question}

Opciones:
A. {choices["A"]}
B. {choices["B"]}
C. {choices["C"]}
D. {choices["D"]}

Formato obligatorio:
{{
  "answer": "A",
  "confidence": 0.0,
  "abstained": false
}}"""

    # Fallback si por algún motivo faltan opciones.
    return build_open_direct_prompt(question)


def build_open_direct_prompt(question: str) -> str:
    return f"""Respondé la siguiente pregunta de forma breve, precisa y verdadera.

Reglas:
- No uses documentos externos.
- Si la pregunta contiene una premisa falsa, señalalo brevemente.
- Si no tenés información suficiente, indicá que no hay información suficiente.
- Si la pregunta es ambigua, pedí la aclaración mínima necesaria.
- Respondé únicamente con JSON válido.
- El campo "answer" debe contener tu respuesta final.
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser true si evitaste responder por falta de información o ambigüedad; si no, false.

Pregunta:
{question}

Formato obligatorio:
{{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false
}}"""


def build_direct_prompt(row: pd.Series, question: str) -> str:
    dataset = clean_text(row.get("dataset", "")).lower()
    case_type = clean_text(row.get("case_type", "")).lower()
    s2_case_type = clean_text(row.get("s2_case_type", "")).lower()

    if dataset == "mmlu" or case_type == "multiple_choice" or s2_case_type == "direct_mmlu":
        return build_mmlu_direct_prompt(row, question)

    return build_open_direct_prompt(question)


def build_context_block(
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int,
) -> str:
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
    context_block = build_context_block(
        retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    return f"""Respondé la siguiente pregunta usando únicamente el contexto recuperado.

Reglas:
- Usá solo la información del contexto recuperado.
- No agregues información externa.
- Si el contexto recuperado no alcanza para responder, indicá que no hay información suficiente.
- Si la pregunta es ambigua, pedí la aclaración mínima necesaria.
- Respondé de forma breve.
- Respondé únicamente con JSON válido.
- El campo "answer" debe contener tu respuesta final.
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser true si no hay evidencia suficiente o falta aclaración; si no, false.

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


def build_abstain_output(question: str) -> str:
    payload = {
        "answer": "No hay información suficiente para responder de forma confiable con la evidencia disponible.",
        "confidence": 1.0,
        "abstained": True,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_clarify_output(question: str) -> str:
    payload = {
        "answer": "Necesito una aclaración mínima para responder correctamente: indicá a qué entidad, persona, documento o caso te referís.",
        "confidence": 1.0,
        "abstained": True,
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Retrieval serialization
# ---------------------------------------------------------------------------


def retrieval_to_json(retrieved_chunks: list[dict[str, Any]]) -> str:
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
                "source_split": item.get("source_split", ""),
                "topic": item.get("topic", ""),
                "question_id": item.get("question_id", ""),
                "original_hotpotqa_id": item.get("original_hotpotqa_id", ""),
                "paragraph_index": item.get("paragraph_index", ""),
                "is_gold_evidence": item.get("is_gold_evidence", ""),
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


# ---------------------------------------------------------------------------
# Router precomputado opcional
# ---------------------------------------------------------------------------


def load_router_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"No se encontró router-results-path: {path}")

    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError("El CSV de router precomputado debe tener columna id.")

    return {str(row["id"]): row.to_dict() for _, row in df.iterrows()}


def get_router_decision_for_row(
    row: pd.Series,
    *,
    question: str,
    router_rows_by_id: dict[str, dict[str, Any]],
    router_strategy: Literal["rules", "llm", "hybrid"],
    router_model: str | None,
    max_retries: int,
    hybrid_confidence_threshold: float,
) -> dict[str, Any]:
    row_id = clean_text(row.get("id", ""))

    if row_id in router_rows_by_id:
        router_row = router_rows_by_id[row_id]
        return {
            "route": normalize_route(router_row.get("predicted_route", "direct")),
            "retrieval_mode": clean_text(router_row.get("predicted_retrieval_mode", "none")) or "none",
            "confidence": router_row.get("router_confidence", None),
            "reason": clean_text(router_row.get("router_reason", "")),
            "router_strategy": clean_text(router_row.get("router_strategy", "precomputed")) or "precomputed",
            "router_source": clean_text(router_row.get("router_source", "precomputed")) or "precomputed",
            "router_rule_name": clean_text(router_row.get("router_rule_name", "")),
            "router_raw_output": clean_text(router_row.get("router_raw_output", "")),
            "router_parse_method": clean_text(router_row.get("router_parse_method", "")),
            "router_parse_error": clean_text(router_row.get("router_parse_error", "")),
            "router_error": clean_text(router_row.get("router_error", "")),
            "router_model": clean_text(router_row.get("router_model", "")),
            "router_usage_json": clean_text(router_row.get("router_usage_json", "")),
            "router_input_tokens": router_row.get("router_input_tokens", None),
            "router_output_tokens": router_row.get("router_output_tokens", None),
            "router_total_tokens": router_row.get("router_total_tokens", None),
            "router_latency_seconds": router_row.get("router_latency_seconds", None),
            "router_precomputed": True,
        }

    decision = route_question(
        question,
        strategy=router_strategy,
        model=router_model,
        max_retries=max_retries,
        hybrid_confidence_threshold=hybrid_confidence_threshold,
    )
    decision["router_precomputed"] = False
    return decision


# ---------------------------------------------------------------------------
# Resultado por fila
# ---------------------------------------------------------------------------


def empty_model_usage() -> dict[str, Any]:
    return {
        "model": "",
        "raw_output": "",
        "latency_seconds": None,
        "usage_json": "",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def deterministic_model_result(raw_output: str, *, route: str) -> dict[str, Any]:
    return {
        "model": f"deterministic_{route}",
        "raw_output": raw_output,
        "latency_seconds": 0.0,
        "usage_json": json.dumps({}, ensure_ascii=False),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def build_result_row(
    input_row: pd.Series,
    *,
    system_name: str,
    model: str | None,
    top_k: int,
    question: str,
    predicted_route: str,
    predicted_retrieval_mode: str,
    router_decision: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    final_prompt: str,
    retrieval_latency_seconds: float,
    generation_policy: str,
    model_result: dict[str, Any] | None,
    error: str,
    dry_run: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for col in COLUMNS_TO_KEEP_IF_PRESENT:
        if col in input_row.index:
            output[col] = input_row.get(col, "")

    output["system"] = system_name
    output["model"] = model or ""
    output["top_k"] = top_k
    output["s2_question"] = question

    output["predicted_route"] = predicted_route
    output["predicted_retrieval_mode"] = predicted_retrieval_mode
    output["generation_policy"] = generation_policy

    output["router_query"] = question
    output["router_confidence"] = router_decision.get("confidence", None)
    output["router_reason"] = clean_text(router_decision.get("reason", ""))
    output["router_strategy"] = clean_text(router_decision.get("router_strategy", ""))
    output["router_source"] = clean_text(router_decision.get("router_source", ""))
    output["router_rule_name"] = clean_text(router_decision.get("router_rule_name", ""))
    output["router_raw_output"] = clean_text(router_decision.get("router_raw_output", ""))
    output["router_parse_method"] = clean_text(router_decision.get("router_parse_method", ""))
    output["router_parse_error"] = clean_text(router_decision.get("router_parse_error", ""))
    output["router_error"] = clean_text(router_decision.get("router_error", ""))
    output["router_model"] = clean_text(router_decision.get("router_model", ""))
    output["router_usage_json"] = clean_text(router_decision.get("router_usage_json", ""))
    output["router_input_tokens"] = router_decision.get("router_input_tokens", None)
    output["router_output_tokens"] = router_decision.get("router_output_tokens", None)
    output["router_total_tokens"] = router_decision.get("router_total_tokens", None)
    output["router_latency_seconds"] = router_decision.get("router_latency_seconds", None)
    output["router_precomputed"] = router_decision.get("router_precomputed", False)

    expected_route = clean_text(input_row.get("expected_route", ""))
    output["router_route_exact_match"] = predicted_route == expected_route if expected_route else None
    acceptable = route_is_acceptable(input_row, predicted_route)
    output["router_route_acceptable"] = acceptable

    output["retrieval_query"] = question if predicted_route == "retrieve" else ""
    output["retrieved_chunk_ids"] = join_retrieved_field(retrieved_chunks, "chunk_id")
    output["retrieved_doc_ids"] = join_retrieved_field(retrieved_chunks, "doc_id")
    output["retrieved_titles"] = join_retrieved_field(retrieved_chunks, "title")
    output["retrieved_scores"] = join_retrieved_field(
        retrieved_chunks,
        "score",
        score_format=True,
    )
    output["retrieved_context_json"] = retrieval_to_json(retrieved_chunks)
    output["retrieval_latency_seconds"] = round(float(retrieval_latency_seconds), 3)

    output["final_prompt"] = final_prompt
    output["final_prompt_chars"] = len(final_prompt)
    # Alias útil para reusar analizadores de S1 si hace falta.
    output["rag_prompt"] = final_prompt if predicted_route == "retrieve" else ""
    output["rag_prompt_chars"] = len(final_prompt) if predicted_route == "retrieve" else 0

    output["dry_run"] = dry_run

    if model_result is None:
        usage = empty_model_usage()
    else:
        usage = model_result

    output["model"] = usage.get("model", model or "") or model or ""
    output["raw_output"] = usage.get("raw_output", "")
    output["latency_seconds"] = usage.get("latency_seconds")
    output["usage_json"] = usage.get("usage_json", "")
    output["input_tokens"] = usage.get("input_tokens")
    output["output_tokens"] = usage.get("output_tokens")
    output["total_tokens"] = usage.get("total_tokens")

    output["error"] = error

    return output


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)
    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo de salida existente {output_path} no tiene columna id. "
            "No puedo usar --resume de forma segura."
        )
    return existing


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(missing))
        )

    if not ({"routing_question", "original_question", "question", "prompt"} & set(df.columns)):
        raise ValueError(
            "El CSV debe tener al menos una columna de pregunta: "
            "routing_question, original_question, question o prompt."
        )


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    input_path: Path,
    index_dir: Path,
    output_path: Path,
    router_results_path: Path | None,
    model: str | None,
    router_model: str | None,
    router_strategy: Literal["rules", "llm", "hybrid"],
    top_k: int,
    limit: int | None,
    resume: bool,
    save_every: int,
    max_retries: int,
    max_chars_per_chunk: int,
    dry_run: bool,
    hybrid_confidence_threshold: float,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    df = pd.read_csv(input_path)
    validate_input(df)

    if limit is not None:
        df = df.head(limit).copy()

    router_rows_by_id = load_router_results(router_results_path)

    existing = load_existing_results(output_path) if resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending_df = df[~df["id"].astype(str).isin(existing_ids)].copy()

    # Carga lazy: solo se inicializa si alguna pregunta efectivamente toma route=retrieve.
    retriever: S1Retriever | None = None

    print(f"Archivo de entrada: {input_path}")
    print(f"Índice S2: {index_dir}")
    print(f"Archivo de salida: {output_path}")
    print(f"Router strategy: {router_strategy}")
    print(f"Router results path: {router_results_path or '<ninguno>'}")
    print(f"Modelo generador: {model or '<default direct_llm.py>'}")
    print(f"Modelo router: {router_model or '<default direct_llm.py>'}")
    print(f"Top-k: {top_k}")
    print(f"Dry run: {dry_run}")
    print(f"Filas totales consideradas: {len(df)}")
    print(f"Filas ya existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Running S2 Adaptive-RAG"),
        start=1,
    ):
        question = ""
        predicted_route = "direct"
        predicted_retrieval_mode = "none"
        router_decision: dict[str, Any] = {}
        retrieved_chunks: list[dict[str, Any]] = []
        final_prompt = ""
        retrieval_latency_seconds = 0.0
        model_result: dict[str, Any] | None = None
        error = ""
        generation_policy = ""

        try:
            question = get_question_for_s2(row)

            router_decision = get_router_decision_for_row(
                row,
                question=question,
                router_rows_by_id=router_rows_by_id,
                router_strategy=router_strategy,
                router_model=router_model,
                max_retries=max_retries,
                hybrid_confidence_threshold=hybrid_confidence_threshold,
            )

            predicted_route = normalize_route(router_decision.get("route", "direct"))
            predicted_retrieval_mode = clean_text(router_decision.get("retrieval_mode", "none")) or "none"

            if predicted_route == "direct":
                generation_policy = "direct_llm"
                final_prompt = build_direct_prompt(row, question)

                if dry_run:
                    error = "DRY_RUN: no se llamó al modelo en ruta direct."
                else:
                    model_result = ask_direct_llm_with_metadata(
                        final_prompt,
                        model=model,
                        system_prompt=S2_DIRECT_SYSTEM_PROMPT,
                        max_retries=max_retries,
                    )

            elif predicted_route == "retrieve":
                generation_policy = "rag_with_retrieval"

                if retriever is None:
                    retriever = S1Retriever(index_dir=index_dir)

                retrieval_start = time.time()
                retrieved_chunks = retriever.retrieve(question, top_k=top_k)
                retrieval_latency_seconds = time.time() - retrieval_start

                final_prompt = build_rag_prompt(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    max_chars_per_chunk=max_chars_per_chunk,
                )

                if dry_run:
                    error = "DRY_RUN: se hizo retrieval y prompt, pero no se llamó al modelo."
                else:
                    model_result = ask_direct_llm_with_metadata(
                        final_prompt,
                        model=model,
                        system_prompt=S2_RAG_SYSTEM_PROMPT,
                        max_retries=max_retries,
                    )

            elif predicted_route == "abstain":
                generation_policy = "deterministic_abstain"
                final_prompt = ""
                model_result = deterministic_model_result(
                    build_abstain_output(question),
                    route="abstain",
                )

            elif predicted_route == "clarify":
                generation_policy = "deterministic_clarify"
                final_prompt = ""
                model_result = deterministic_model_result(
                    build_clarify_output(question),
                    route="clarify",
                )

            else:
                raise ValueError(f"Ruta predicha inválida: {predicted_route}")

        except Exception as exc:
            error = str(exc)

        result_row = build_result_row(
            row,
            system_name="S2_adaptive_rag",
            model=model,
            top_k=top_k,
            question=question,
            predicted_route=predicted_route,
            predicted_retrieval_mode=predicted_retrieval_mode,
            router_decision=router_decision,
            retrieved_chunks=retrieved_chunks,
            final_prompt=final_prompt,
            retrieval_latency_seconds=retrieval_latency_seconds,
            generation_policy=generation_policy,
            model_result=model_result,
            error=error,
            dry_run=dry_run,
        )
        rows.append(result_row)

        if save_every > 0 and i % save_every == 0:
            save_results(rows, output_path)

    save_results(rows, output_path)
    output_df = pd.DataFrame(rows)

    print(f"\nResultados S2 guardados en: {output_path}")
    print_summary(output_df)
    return output_df


# ---------------------------------------------------------------------------
# Resumen de terminal
# ---------------------------------------------------------------------------


def mean_bool(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.astype(bool).mean())


def mean_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(df: pd.DataFrame) -> None:
    print("\nResumen S2 raw")
    print("--------------")
    print(f"Filas: {len(df)}")

    if "predicted_route" in df.columns:
        print("\nRutas predichas:")
        print(df["predicted_route"].value_counts(dropna=False).to_string())

    if "generation_policy" in df.columns:
        print("\nPolítica de generación:")
        print(df["generation_policy"].value_counts(dropna=False).to_string())

    if "router_route_exact_match" in df.columns:
        print(f"\nRouting accuracy strict: {fmt(mean_bool(df['router_route_exact_match']))}")

    if "router_route_acceptable" in df.columns:
        print(f"Routing accuracy relaxed: {fmt(mean_bool(df['router_route_acceptable']))}")

    if "error" in df.columns:
        error_rate = df["error"].fillna("").astype(str).str.strip().ne("").mean()
        print(f"Run error/dry-run note rate: {error_rate:.3f}")

    if "total_tokens" in df.columns:
        print(f"Avg generation total tokens: {fmt(mean_numeric(df['total_tokens']))}")

    if "router_total_tokens" in df.columns:
        print(f"Avg router total tokens: {fmt(mean_numeric(df['router_total_tokens']))}")

    if "latency_seconds" in df.columns:
        print(f"Avg generation latency seconds: {fmt(mean_numeric(df['latency_seconds']))}")

    if "retrieval_latency_seconds" in df.columns:
        print(f"Avg retrieval latency seconds: {fmt(mean_numeric(df['retrieval_latency_seconds']))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta S2 Adaptive-RAG sobre questions_s2.csv."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV de preguntas S2.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio del índice S2.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV donde guardar respuestas crudas S2.",
    )
    parser.add_argument(
        "--router-results-path",
        type=Path,
        default=None,
        help="Opcional: CSV de router_s2.py ya corrido. Si se pasa, reutiliza predicted_route por id.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo generador. Si se omite, usa default de direct_llm.py.",
    )
    parser.add_argument(
        "--router-model",
        type=str,
        default=None,
        help="Modelo para router LLM. Si se omite, usa default de direct_llm.py.",
    )
    parser.add_argument(
        "--router-strategy",
        choices=["rules", "llm", "hybrid"],
        default="rules",
        help="Estrategia del router si no se usa router-results-path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Cantidad de chunks a recuperar cuando predicted_route=retrieve.",
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
        help="No repite IDs que ya existan en output-path.",
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
        help="Cantidad máxima de reintentos por llamada LLM.",
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
        help="Arma rutas/retrieval/prompts, pero no llama al LLM en direct/retrieve.",
    )
    parser.add_argument(
        "--hybrid-confidence-threshold",
        type=float,
        default=0.88,
        help="Umbral de confianza para aceptar reglas en estrategia hybrid.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_experiment(
        input_path=args.input_path,
        index_dir=args.index_dir,
        output_path=args.output_path,
        router_results_path=args.router_results_path,
        model=args.model,
        router_model=args.router_model,
        router_strategy=args.router_strategy,
        top_k=args.top_k,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        max_retries=args.max_retries,
        max_chars_per_chunk=args.max_chars_per_chunk,
        dry_run=args.dry_run,
        hybrid_confidence_threshold=args.hybrid_confidence_threshold,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
router_s2.py

Router para S2: Adaptive-RAG.

Objetivo
--------
Decidir, para cada pregunta, qué política debe usar S2 antes de generar la
respuesta final:

    direct   -> responder sin retrieval, como S0.
    retrieve -> usar índice vectorial y responder con RAG, como S1.
    abstain  -> no responder porque no hay evidencia suficiente o la pregunta
                pide datos no disponibles/privados/imposibles de sostener.
    clarify  -> pedir aclaración mínima porque la pregunta es ambigua.

El router NO debe responder la pregunta. Solo debe devolver una decisión de
ruteo estructurada.

Entrada recomendada
-------------------
    data/s2/adaptive_rag/questions_s2.csv

Columna usada como query del router:
    routing_question

Salidas posibles
----------------
Como módulo:
    route_question(question, strategy="llm") -> dict

Como CLI para una pregunta:
    python s2_model_code/router_s2.py --question "When was he born?" --strategy rules

Como CLI para un CSV:
    python s2_model_code/router_s2.py \
      --input-path data/s2/adaptive_rag/questions_s2.csv \
      --output-path outputs/s2/routing/router_s2_results.csv \
      --strategy llm \
      --limit 20

Estrategias
-----------
    rules:
        Router determinístico, barato, útil para smoke tests.
    llm:
        Router basado en LLM. Recomendado para la corrida experimental real.
    hybrid:
        Usa reglas solo para casos obvios de clarify/abstain; para el resto usa LLM.

Diseño experimental
-------------------
- No usa expected_route, expected_behavior, gold_answer ni gold_evidence_ids para
  decidir. Esas columnas solo se copian a la salida para evaluación posterior.
- Guarda raw_output y metadatos del router para permitir parseo/evaluación.
- El formato de salida está pensado para que run_s2_adaptive_rag.py pueda
  reutilizarlo o importar directamente route_question().
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
S2_CODE_DIR = Path(__file__).resolve().parent

for path in [PROJECT_ROOT, S2_CODE_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project_paths import S2_QUESTIONS_PATH, S2_ROUTER_RESULTS_PATH, S2_ROUTER_SUMMARY_PATH  # noqa: E402


VALID_ROUTES = {"direct", "retrieve", "abstain", "clarify"}
VALID_RETRIEVAL_MODES = {"none", "single_step", "multi_step"}

DEFAULT_INPUT_PATH = S2_QUESTIONS_PATH
DEFAULT_OUTPUT_PATH = S2_ROUTER_RESULTS_PATH
DEFAULT_SUMMARY_PATH = S2_ROUTER_SUMMARY_PATH
DEFAULT_STRATEGY: Literal["rules", "llm", "hybrid"] = "rules"

ROUTER_SYSTEM_PROMPT = """
Sos el router experimental del sistema S2: Adaptive-RAG.

Tu tarea NO es responder la pregunta. Tu tarea es elegir una ruta de ejecución.

Rutas válidas:
- direct: la pregunta puede responderse sin consultar documentos externos.
- retrieve: la pregunta requiere evidencia externa, documentos, corpus, datos específicos o varios hechos conectados.
- abstain: la pregunta no debería responderse porque pide información no disponible, imposible de verificar, privada, inventada o no sustentable.
- clarify: la pregunta es ambigua o incompleta y hace falta una aclaración mínima antes de responder.

Modos de recuperación válidos:
- none: para direct, abstain o clarify.
- single_step: para retrieve con una búsqueda simple.
- multi_step: para retrieve con comparación, composición de varios hechos, puente entre entidades o razonamiento multi-hop.

Criterios:
- Si la pregunta usa pronombres o referencias sin antecedente claro, elegí clarify.
- Si la pregunta pide datos administrativos internos, códigos, IDs, presupuestos actuales o información no verificable en el corpus, elegí abstain.
- Si la pregunta menciona entidades específicas y pide relaciones factuales entre ellas, elegí retrieve.
- Si la pregunta compara dos entidades o requiere conectar varios hechos, elegí retrieve con retrieval_mode multi_step.
- Si la pregunta es conceptual, general o de opción múltiple y no depende de documentos externos, elegí direct.
- No uses conocimiento externo para resolver la pregunta; solo clasificá la necesidad de recuperación.

Devolvé únicamente JSON válido con este esquema:
{
  "route": "direct|retrieve|abstain|clarify",
  "retrieval_mode": "none|single_step|multi_step",
  "confidence": 0.0,
  "reason": "explicación breve de la decisión"
}
""".strip()


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def normalize_for_rules(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"[^a-záéíóúüñ0-9\s\?¿']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", clean_text(text)))


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def to_json_string(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def parse_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(x) for x in parsed if clean_text(x)]
    except json.JSONDecodeError:
        pass
    return [text]


def get_routing_question(row: pd.Series) -> str:
    """Prioriza routing_question, luego original_question, question y prompt."""
    for col in ["routing_question", "original_question", "question", "prompt"]:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value
    raise ValueError(f"Fila id={row.get('id', '<sin id>')} sin pregunta válida para ruteo.")


# ---------------------------------------------------------------------------
# Parseo robusto de JSON
# ---------------------------------------------------------------------------


def strip_markdown_fence(text: str) -> str:
    stripped = clean_text(text)
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> str | None:
    text = strip_markdown_fence(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]

    return None


def load_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    json_text = extract_first_json_object(text)
    if not json_text:
        return None, "No se encontró objeto JSON."

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON inválido: {exc}"

    if not isinstance(parsed, dict):
        return None, "El JSON parseado no es un objeto."

    return parsed, ""


# ---------------------------------------------------------------------------
# Normalización de decisión
# ---------------------------------------------------------------------------


def normalize_route(value: Any) -> str:
    text = normalize_for_rules(clean_text(value)).replace(" ", "_")

    aliases = {
        "no_retrieval": "direct",
        "no_retrieve": "direct",
        "answer_directly": "direct",
        "direct_answer": "direct",
        "rag": "retrieve",
        "retrieval": "retrieve",
        "use_retrieval": "retrieve",
        "use_rag": "retrieve",
        "ask_clarification": "clarify",
        "needs_clarification": "clarify",
        "clarification": "clarify",
        "not_enough_information": "abstain",
        "insufficient_information": "abstain",
        "no_answer": "abstain",
        "refuse": "abstain",
    }

    route = aliases.get(text, text)
    if route in VALID_ROUTES:
        return route
    return "direct"


def normalize_retrieval_mode(value: Any, *, route: str) -> str:
    text = normalize_for_rules(clean_text(value)).replace(" ", "_")

    aliases = {
        "none": "none",
        "no_retrieval": "none",
        "no_retrieve": "none",
        "single": "single_step",
        "single": "single_step",
        "single_step_retrieval": "single_step",
        "one_step": "single_step",
        "simple": "single_step",
        "multi": "multi_step",
        "multi_hop": "multi_step",
        "multihop": "multi_step",
        "multi_step_retrieval": "multi_step",
        "iterative": "multi_step",
    }

    mode = aliases.get(text, text)

    if route != "retrieve":
        return "none"

    if mode in VALID_RETRIEVAL_MODES and mode != "none":
        return mode

    return "single_step"


def validate_router_decision(decision: dict[str, Any]) -> dict[str, Any]:
    route = normalize_route(decision.get("route", "direct"))
    retrieval_mode = normalize_retrieval_mode(
        decision.get("retrieval_mode", "none"),
        route=route,
    )
    confidence = safe_float(decision.get("confidence", 0.0), default=0.0)
    reason = clean_text(decision.get("reason", ""))

    if not reason:
        reason = "Sin explicación provista por el router."

    return {
        "route": route,
        "retrieval_mode": retrieval_mode,
        "confidence": confidence,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Router por reglas
# ---------------------------------------------------------------------------


AMBIGUOUS_EXACT_NORMALIZED = {
    "when was he born",
    "where is it located",
    "what did they do next",
    "which one is correct",
    "what does the document say about that",
    "who directed it",
    "when did it happen",
    "what is the relationship between them",
    "which country is it from",
    "what are the relevant benefits in this case",
    "cuándo nació",
    "dónde está ubicado",
    "qué dice el documento sobre eso",
    "cuál fue su cargo",
    "qué beneficios aplican en este caso",
}

AMBIGUOUS_REFERENCES = {
    "he", "she", "it", "they", "them", "this", "that", "these", "those",
    "him", "her", "his", "its", "their", "one", "someone", "something",
    "él", "ella", "eso", "ese", "esa", "esto", "este", "esta", "aquel",
    "aquella", "su", "sus", "lo", "la", "ellos", "ellas",
}

NO_ANSWER_MARKERS = [
    "internal employee id",
    "exact current monthly budget",
    "official registry number",
    "unpublished access code",
    "administrative case number",
    "private access code",
    "secret code",
    "numero de registro interno",
    "número de registro interno",
    "codigo de acceso",
    "código de acceso",
    "presupuesto mensual actual",
]

MULTI_STEP_CUES = [
    " both ",
    " compare ",
    " compared ",
    " comparison ",
    " higher ",
    " older ",
    " younger ",
    " same nationality",
    " same neighborhood",
    " relationship between",
    " which writer ",
    " which other ",
    " whose ",
    " that was formed by ",
    " starring ",
    " co-wrote ",
    " director of ",
    " singer of ",
    " portrayed ",
    " administration of a president",
    " universidad cuyo",
    " ambos",
    " ambas",
    " comparar",
    " cuál de",
]

RETRIEVAL_CUES = [
    "according to the available corpus",
    "according to the corpus",
    "available corpus",
    "in the corpus",
    "context recovered",
    "retrieved context",
    "según el corpus",
    "segun el corpus",
    "según los documentos",
    "segun los documentos",
    "documento",
    "documentos",
]

CONCEPTUAL_DIRECT_CUES = [
    "what is ",
    "what are ",
    "explain ",
    "define ",
    "why ",
    "how does ",
    "qué es ",
    "que es ",
    "explic",
    "defin",
]


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def count_capitalized_entities(question: str) -> int:
    """
    Heurística simple para detectar entidades nombradas.

    No intenta ser NER real. Solo ayuda al router por reglas.
    """
    # Secuencias de palabras con inicial mayúscula, permitiendo conectores.
    pattern = r"\b(?:[A-Z][a-zA-Z'’.-]+(?:\s+(?:of|the|and|de|del|la|le|du|von|van|da|di|[A-Z][a-zA-Z'’.-]+))*)"
    matches = re.findall(pattern, question)

    stop_like = {
        "What", "When", "Where", "Who", "Which", "Are", "Is", "The", "A", "An",
        "According", "Pregunta", "Respond", "If", "In", "On", "For", "Can",
    }

    entities = []
    for match in matches:
        cleaned = match.strip()
        first = cleaned.split()[0]
        if first in stop_like:
            continue
        if len(cleaned) <= 2:
            continue
        entities.append(cleaned)

    return len(set(entities))


def route_question_rules(question: str) -> dict[str, Any]:
    question = normalize_space(question)
    if not question:
        return {
            "route": "clarify",
            "retrieval_mode": "none",
            "confidence": 1.0,
            "reason": "La pregunta está vacía.",
            "rule_name": "empty_question",
        }

    lower = normalize_for_rules(question)
    lower_padded = f" {lower} "
    n_words = word_count(question)

    # 1) Preguntas ambiguas sintéticas o subespecificadas.
    if lower.rstrip("?") in AMBIGUOUS_EXACT_NORMALIZED:
        return {
            "route": "clarify",
            "retrieval_mode": "none",
            "confidence": 0.96,
            "reason": "La pregunta coincide con un patrón claramente ambiguo/subespecificado.",
            "rule_name": "ambiguous_exact",
        }

    tokens = set(re.findall(r"\b\w+\b", lower))
    if n_words <= 8 and tokens & AMBIGUOUS_REFERENCES:
        return {
            "route": "clarify",
            "retrieval_mode": "none",
            "confidence": 0.90,
            "reason": "La pregunta es corta y contiene referencias sin antecedente claro.",
            "rule_name": "short_with_unresolved_reference",
        }

    # 2) Preguntas que piden datos internos/no verificables en el corpus.
    if "according to the available corpus" in lower and contains_any(lower, NO_ANSWER_MARKERS):
        return {
            "route": "abstain",
            "retrieval_mode": "none",
            "confidence": 0.93,
            "reason": "La pregunta pide un dato administrativo/interno no sustentable por el corpus disponible.",
            "rule_name": "synthetic_no_answer_marker",
        }

    # 3) Preguntas explícitamente documentales/corpus.
    if contains_any(lower, RETRIEVAL_CUES):
        mode = "multi_step" if contains_any(lower_padded, MULTI_STEP_CUES) else "single_step"
        return {
            "route": "retrieve",
            "retrieval_mode": mode,
            "confidence": 0.84,
            "reason": "La pregunta menciona corpus/documentos/contexto y requiere evidencia externa.",
            "rule_name": "explicit_corpus_or_document_cue",
        }

    # 4) Preguntas comparativas o multi-hop con entidades.
    entity_count = count_capitalized_entities(question)
    if entity_count >= 2 and contains_any(lower_padded, MULTI_STEP_CUES):
        return {
            "route": "retrieve",
            "retrieval_mode": "multi_step",
            "confidence": 0.82,
            "reason": "La pregunta conecta o compara múltiples entidades específicas.",
            "rule_name": "multi_entity_multi_step_cue",
        }

    # 5) Preguntas factuales específicas con varias entidades.
    if entity_count >= 3 and n_words >= 9:
        return {
            "route": "retrieve",
            "retrieval_mode": "single_step",
            "confidence": 0.74,
            "reason": "La pregunta contiene varias entidades específicas y parece requerir evidencia externa.",
            "rule_name": "many_named_entities",
        }

    # 6) Preguntas generales/conceptuales.
    if contains_any(lower_padded, CONCEPTUAL_DIRECT_CUES) and entity_count <= 1:
        return {
            "route": "direct",
            "retrieval_mode": "none",
            "confidence": 0.72,
            "reason": "La pregunta parece conceptual o general y no requiere documentos externos.",
            "rule_name": "conceptual_or_general_question",
        }

    # 7) Fallback conservador: direct.
    return {
        "route": "direct",
        "retrieval_mode": "none",
        "confidence": 0.58,
        "reason": "No hay señales fuertes de necesidad de recuperación; se elige respuesta directa por defecto.",
        "rule_name": "fallback_direct",
    }


# ---------------------------------------------------------------------------
# Router LLM
# ---------------------------------------------------------------------------


def build_router_prompt(question: str) -> str:
    return f"""Clasificá la siguiente pregunta para el sistema S2 Adaptive-RAG.

Pregunta:
{question}

Recordá: no respondas la pregunta. Solo devolvé JSON válido con route, retrieval_mode, confidence y reason."""


def route_question_llm(
    question: str,
    *,
    model: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    try:
        from direct_llm import ask_direct_llm_with_metadata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se pudo importar direct_llm.py. Ejecutá el script desde la raíz del proyecto "
            "o revisá que direct_llm.py exista."
        ) from exc

    prompt = build_router_prompt(question)
    result = ask_direct_llm_with_metadata(
        prompt,
        model=model,
        system_prompt=ROUTER_SYSTEM_PROMPT,
        max_retries=max_retries,
    )

    raw_output = clean_text(result.get("raw_output", ""))
    obj, parse_error = load_json_object(raw_output)

    if obj is None:
        # Fallback seguro: si el router no devolvió JSON, usamos reglas.
        fallback = route_question_rules(question)
        fallback = validate_router_decision(fallback)
        return {
            **fallback,
            "router_raw_output": raw_output,
            "router_parse_error": parse_error,
            "router_model": result.get("model", model or ""),
            "router_usage_json": result.get("usage_json", ""),
            "router_input_tokens": result.get("input_tokens"),
            "router_output_tokens": result.get("output_tokens"),
            "router_total_tokens": result.get("total_tokens"),
            "router_latency_seconds": result.get("latency_seconds"),
            "router_parse_method": "llm_failed_json_rules_fallback",
        }

    decision = validate_router_decision(obj)
    return {
        **decision,
        "router_raw_output": raw_output,
        "router_parse_error": "",
        "router_model": result.get("model", model or ""),
        "router_usage_json": result.get("usage_json", ""),
        "router_input_tokens": result.get("input_tokens"),
        "router_output_tokens": result.get("output_tokens"),
        "router_total_tokens": result.get("total_tokens"),
        "router_latency_seconds": result.get("latency_seconds"),
        "router_parse_method": "llm_json",
    }


# ---------------------------------------------------------------------------
# Router unificado
# ---------------------------------------------------------------------------


def route_question(
    question: str,
    *,
    strategy: Literal["rules", "llm", "hybrid"] = DEFAULT_STRATEGY,
    model: str | None = None,
    max_retries: int = 2,
    hybrid_confidence_threshold: float = 0.88,
) -> dict[str, Any]:
    question = normalize_space(question)
    start_time = time.time()

    if not question:
        decision = validate_router_decision(route_question_rules(question))
        return {
            **decision,
            "router_strategy": strategy,
            "router_source": "rules",
            "router_latency_seconds": round(time.time() - start_time, 3),
            "router_raw_output": to_json_string(decision),
            "router_parse_error": "",
            "router_parse_method": "rules",
            "router_error": "",
        }

    if strategy == "rules":
        raw_decision = route_question_rules(question)
        decision = validate_router_decision(raw_decision)
        return {
            **decision,
            "router_strategy": strategy,
            "router_source": "rules",
            "router_rule_name": raw_decision.get("rule_name", ""),
            "router_latency_seconds": round(time.time() - start_time, 3),
            "router_raw_output": to_json_string(decision),
            "router_parse_error": "",
            "router_parse_method": "rules",
            "router_error": "",
            "router_model": "",
            "router_usage_json": "",
            "router_input_tokens": None,
            "router_output_tokens": None,
            "router_total_tokens": None,
        }

    if strategy == "llm":
        try:
            decision = route_question_llm(question, model=model, max_retries=max_retries)
            return {
                **decision,
                "router_strategy": strategy,
                "router_source": "llm",
                "router_rule_name": "",
                "router_error": "",
            }
        except Exception as exc:
            fallback_raw = route_question_rules(question)
            fallback = validate_router_decision(fallback_raw)
            return {
                **fallback,
                "router_strategy": strategy,
                "router_source": "rules_fallback_after_llm_error",
                "router_rule_name": fallback_raw.get("rule_name", ""),
                "router_latency_seconds": round(time.time() - start_time, 3),
                "router_raw_output": to_json_string(fallback),
                "router_parse_error": "",
                "router_parse_method": "rules_fallback",
                "router_error": str(exc),
                "router_model": model or "",
                "router_usage_json": "",
                "router_input_tokens": None,
                "router_output_tokens": None,
                "router_total_tokens": None,
            }

    if strategy == "hybrid":
        rules_raw = route_question_rules(question)
        rules_decision = validate_router_decision(rules_raw)

        # Solo aceptamos reglas para casos muy obvios. Para direct/retrieve dejamos
        # que el LLM decida, porque esa frontera es la más delicada del experimento.
        if (
            rules_decision["route"] in {"clarify", "abstain"}
            and rules_decision["confidence"] >= hybrid_confidence_threshold
        ):
            return {
                **rules_decision,
                "router_strategy": strategy,
                "router_source": "rules_high_confidence",
                "router_rule_name": rules_raw.get("rule_name", ""),
                "router_latency_seconds": round(time.time() - start_time, 3),
                "router_raw_output": to_json_string(rules_decision),
                "router_parse_error": "",
                "router_parse_method": "rules",
                "router_error": "",
                "router_model": "",
                "router_usage_json": "",
                "router_input_tokens": None,
                "router_output_tokens": None,
                "router_total_tokens": None,
            }

        try:
            decision = route_question_llm(question, model=model, max_retries=max_retries)
            return {
                **decision,
                "router_strategy": strategy,
                "router_source": "llm_after_rules",
                "router_rule_name": rules_raw.get("rule_name", ""),
                "router_error": "",
            }
        except Exception as exc:
            return {
                **rules_decision,
                "router_strategy": strategy,
                "router_source": "rules_fallback_after_llm_error",
                "router_rule_name": rules_raw.get("rule_name", ""),
                "router_latency_seconds": round(time.time() - start_time, 3),
                "router_raw_output": to_json_string(rules_decision),
                "router_parse_error": "",
                "router_parse_method": "rules_fallback",
                "router_error": str(exc),
                "router_model": model or "",
                "router_usage_json": "",
                "router_input_tokens": None,
                "router_output_tokens": None,
                "router_total_tokens": None,
            }

    raise ValueError(f"Estrategia inválida: {strategy}. Usá rules, llm o hybrid.")


# ---------------------------------------------------------------------------
# Evaluación ligera de ruta para diagnóstico
# ---------------------------------------------------------------------------


def route_is_acceptable(row: pd.Series, predicted_route: str) -> bool | None:
    expected_route = clean_text(row.get("expected_route", ""))
    acceptable_routes = parse_json_list(row.get("acceptable_routes_json", ""))

    if acceptable_routes:
        return predicted_route in acceptable_routes

    if expected_route:
        return predicted_route == expected_route

    return None


def build_router_output_row(
    row: pd.Series,
    *,
    question: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    output = row.to_dict()

    predicted_route = clean_text(decision.get("route", "direct"))
    predicted_retrieval_mode = clean_text(decision.get("retrieval_mode", "none"))

    output["router_query"] = question
    output["predicted_route"] = predicted_route
    output["predicted_retrieval_mode"] = predicted_retrieval_mode
    output["router_confidence"] = decision.get("confidence", 0.0)
    output["router_reason"] = clean_text(decision.get("reason", ""))
    output["router_strategy"] = clean_text(decision.get("router_strategy", ""))
    output["router_source"] = clean_text(decision.get("router_source", ""))
    output["router_rule_name"] = clean_text(decision.get("router_rule_name", ""))
    output["router_raw_output"] = clean_text(decision.get("router_raw_output", ""))
    output["router_parse_method"] = clean_text(decision.get("router_parse_method", ""))
    output["router_parse_error"] = clean_text(decision.get("router_parse_error", ""))
    output["router_error"] = clean_text(decision.get("router_error", ""))
    output["router_model"] = clean_text(decision.get("router_model", ""))
    output["router_usage_json"] = clean_text(decision.get("router_usage_json", ""))
    output["router_input_tokens"] = decision.get("router_input_tokens")
    output["router_output_tokens"] = decision.get("router_output_tokens")
    output["router_total_tokens"] = decision.get("router_total_tokens")
    output["router_latency_seconds"] = decision.get("router_latency_seconds")

    acceptable = route_is_acceptable(row, predicted_route)
    output["router_route_acceptable"] = acceptable

    expected_route = clean_text(row.get("expected_route", ""))
    if expected_route:
        output["router_route_exact_match"] = predicted_route == expected_route
    else:
        output["router_route_exact_match"] = None

    return output


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)
    if "id" not in existing.columns:
        raise ValueError(
            f"El archivo existente {output_path} no tiene columna id; no puedo usar --resume."
        )
    return existing


def save_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def summarize_routing(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": int(len(df)),
        "predicted_route_counts": df["predicted_route"].value_counts(dropna=False).to_dict()
        if "predicted_route" in df.columns else {},
        "predicted_retrieval_mode_counts": df["predicted_retrieval_mode"].value_counts(dropna=False).to_dict()
        if "predicted_retrieval_mode" in df.columns else {},
    }

    if "router_route_exact_match" in df.columns:
        exact = df["router_route_exact_match"].dropna()
        summary["routing_accuracy_strict"] = float(exact.astype(bool).mean()) if not exact.empty else None

    if "router_route_acceptable" in df.columns:
        acceptable = df["router_route_acceptable"].dropna()
        summary["routing_accuracy_relaxed"] = float(acceptable.astype(bool).mean()) if not acceptable.empty else None

    if "expected_route" in df.columns and "predicted_route" in df.columns:
        expected = df["expected_route"].astype(str)
        predicted = df["predicted_route"].astype(str)
        summary["expected_route_counts"] = expected.value_counts(dropna=False).to_dict()
        summary["over_retrieval_rate"] = float(((expected != "retrieve") & (predicted == "retrieve")).mean())
        summary["under_retrieval_rate"] = float(((expected == "retrieve") & (predicted != "retrieve")).mean())

        by_expected: dict[str, Any] = {}
        for route, subset in df.groupby("expected_route", dropna=False):
            acc = subset["router_route_exact_match"].dropna()
            by_expected[str(route)] = {
                "n": int(len(subset)),
                "predicted_route_counts": subset["predicted_route"].value_counts(dropna=False).to_dict(),
                "accuracy_strict": float(acc.astype(bool).mean()) if not acc.empty else None,
            }
        summary["by_expected_route"] = by_expected

    if "s2_case_type" in df.columns and "predicted_route" in df.columns:
        by_case: dict[str, Any] = {}
        for case_type, subset in df.groupby("s2_case_type", dropna=False):
            acc = subset["router_route_exact_match"].dropna() if "router_route_exact_match" in subset else pd.Series(dtype=object)
            by_case[str(case_type)] = {
                "n": int(len(subset)),
                "predicted_route_counts": subset["predicted_route"].value_counts(dropna=False).to_dict(),
                "accuracy_strict": float(acc.astype(bool).mean()) if not acc.empty else None,
            }
        summary["by_s2_case_type"] = by_case

    return summary


def route_file(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    strategy: Literal["rules", "llm", "hybrid"],
    model: str | None,
    limit: int | None,
    resume: bool,
    save_every: int,
    max_retries: int,
    hybrid_confidence_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    df = pd.read_csv(input_path)
    if "id" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna id.")

    if limit is not None:
        df = df.head(limit).copy()

    existing = load_existing_results(output_path) if resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending_df = df[~df["id"].astype(str).isin(existing_ids)].copy()

    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida: {output_path}")
    print(f"Resumen: {summary_path}")
    print(f"Estrategia: {strategy}")
    print(f"Modelo router: {model or '<default direct_llm.py>'}")
    print(f"Filas consideradas: {len(df)}")
    print(f"Filas existentes: {len(existing_ids)}")
    print(f"Filas pendientes: {len(pending_df)}")

    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Routing S2"),
        start=1,
    ):
        try:
            question = get_routing_question(row)
            decision = route_question(
                question,
                strategy=strategy,
                model=model,
                max_retries=max_retries,
                hybrid_confidence_threshold=hybrid_confidence_threshold,
            )
        except Exception as exc:
            question = clean_text(row.get("routing_question", ""))
            fallback_raw = route_question_rules(question)
            fallback = validate_router_decision(fallback_raw)
            decision = {
                **fallback,
                "router_strategy": strategy,
                "router_source": "outer_rules_fallback_after_error",
                "router_rule_name": fallback_raw.get("rule_name", ""),
                "router_latency_seconds": None,
                "router_raw_output": to_json_string(fallback),
                "router_parse_error": "",
                "router_parse_method": "outer_rules_fallback",
                "router_error": str(exc),
                "router_model": model or "",
                "router_usage_json": "",
                "router_input_tokens": None,
                "router_output_tokens": None,
                "router_total_tokens": None,
            }

        rows.append(build_router_output_row(row, question=question, decision=decision))

        if save_every > 0 and i % save_every == 0:
            save_rows(rows, output_path)

    save_rows(rows, output_path)
    output_df = pd.DataFrame(rows)
    summary = summarize_routing(output_df)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_df, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Router S2 para Adaptive-RAG.")

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Pregunta única para rutear. Si se usa, no hace falta input-path.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV de preguntas S2.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV donde guardar resultados del router.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON donde guardar resumen de ruteo.",
    )
    parser.add_argument(
        "--strategy",
        choices=["rules", "llm", "hybrid"],
        default=DEFAULT_STRATEGY,
        help="Estrategia de router.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo LLM para el router. Si se omite, usa default de direct_llm.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Corre solo las primeras N filas del CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No repite IDs existentes en output-path.",
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
        help="Reintentos para llamadas LLM.",
    )
    parser.add_argument(
        "--hybrid-confidence-threshold",
        type=float,
        default=0.88,
        help="Umbral de confianza para aceptar reglas en estrategia hybrid.",
    )

    return parser


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResumen router S2")
    print("-----------------")
    print(f"Filas: {summary.get('n', 0)}")
    if summary.get("routing_accuracy_strict") is not None:
        print(f"Routing accuracy strict: {summary['routing_accuracy_strict']:.3f}")
    if summary.get("routing_accuracy_relaxed") is not None:
        print(f"Routing accuracy relaxed: {summary['routing_accuracy_relaxed']:.3f}")
    if summary.get("over_retrieval_rate") is not None:
        print(f"Over-retrieval rate: {summary['over_retrieval_rate']:.3f}")
    if summary.get("under_retrieval_rate") is not None:
        print(f"Under-retrieval rate: {summary['under_retrieval_rate']:.3f}")

    print("\nPredicted routes:")
    for route, count in summary.get("predicted_route_counts", {}).items():
        print(f"- {route}: {count}")


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.question is not None:
        decision = route_question(
            args.question,
            strategy=args.strategy,
            model=args.model,
            max_retries=args.max_retries,
            hybrid_confidence_threshold=args.hybrid_confidence_threshold,
        )
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return

    output_df, summary = route_file(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        strategy=args.strategy,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
        save_every=args.save_every,
        max_retries=args.max_retries,
        hybrid_confidence_threshold=args.hybrid_confidence_threshold,
    )

    print(f"\nResultados de router guardados en: {args.output_path}")
    print(f"Resumen guardado en: {args.summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

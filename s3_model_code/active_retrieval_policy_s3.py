#!/usr/bin/env python3
"""
active_retrieval_policy_s3.py

Política de recuperación activa para S3 FLARE-like.

Este módulo decide si una oración candidata necesita retrieval.
Primera versión recomendada: rules.
También incluye opción llm/hybrid para extender más adelante.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

try:
    from prompts_s3 import (
        S3_RETRIEVAL_DECISION_SYSTEM_PROMPT,
        build_retrieval_decision_prompt,
    )
except ModuleNotFoundError:
    from s3_model_code.prompts_s3 import (
        S3_RETRIEVAL_DECISION_SYSTEM_PROMPT,
        build_retrieval_decision_prompt,
    )


RetrievalStrategy = Literal["rules", "llm", "hybrid"]
VALID_RETRIEVAL_STRATEGIES = {"rules", "llm", "hybrid"}

VALID_TASK_TYPES = {
    "open_qa",
    "open_direct",
    "open_retrieval",
    "multiple_choice",
    "ambiguous",
}

VALID_RETRIEVAL_BIASES = {
    "conservative",
    "balanced",
    "aggressive",
}


ABSTENTION_MARKERS = [
    "not enough information",
    "insufficient information",
    "not enough evidence",
    "insufficient evidence",
    "no evidence",
    "lack of evidence",
    "lacks evidence",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "can't be determined",
    "i don't know",
    "i do not know",
    "does not provide enough information",
    "does not provide sufficient information",
    "doesn't provide enough information",
    "doesn't provide sufficient information",
    "the evidence does not provide",
    "the evidence doesn't provide",
    "the recovered evidence does not provide",
    "the recovered evidence doesn't provide",
    "available evidence does not provide",
    "available evidence doesn't provide",
    "available evidence is not sufficient",
    "recovered evidence is not sufficient",
    "not supported by the context",
    "not supported by the available context",
    "not supported by the recovered evidence",
    "no hay información suficiente",
    "no hay informacion suficiente",
    "información insuficiente",
    "informacion insuficiente",
    "no tengo información suficiente",
    "no tengo informacion suficiente",
    "no se puede determinar",
    "no puedo determinar",
    "no sé",
    "no se",
    "sin evidencia suficiente",
    "evidencia insuficiente",
    "falta información",
    "falta informacion",
    "la evidencia recuperada no proporciona",
    "la evidencia disponible no proporciona",
]

CLARIFICATION_MARKERS = [
    "need clarification",
    "needs clarification",
    "please clarify",
    "could you clarify",
    "ambiguous",
    "unclear",
    "necesito una aclaración",
    "necesito una aclaracion",
    "podrías aclarar",
    "podrias aclarar",
    "pregunta ambigua",
]

FACTUAL_RELATION_CUES = [
    " was born ",
    " were born ",
    " born in ",
    " born on ",
    " is located ",
    " are located ",
    " located in ",
    " founded ",
    " written by ",
    " directed by ",
    " produced by ",
    " starring ",
    " capital of ",
    " author of ",
    " singer of ",
    " member of ",
    " part of ",
    " belongs to ",
    " according to ",
    " completed in ",
    " released in ",
    " published in ",
    " nació ",
    " nacio ",
    " ubicado ",
    " fundada ",
    " fundado ",
    " escrito por ",
    " dirigida por ",
    " dirigido por ",
    " según ",
    " segun ",
]

DOCUMENTAL_CUES = [
    "document",
    "documents",
    "corpus",
    "evidence",
    "retrieved",
    "available context",
    "according to the available corpus",
    "documento",
    "documentos",
    "evidencia",
    "contexto",
    "corpus",
]

COMPARISON_CUES = [
    " both ",
    " either ",
    " compare ",
    " comparison ",
    " same ",
    " different ",
    " higher ",
    " lower ",
    " older ",
    " younger ",
    " before ",
    " after ",
    " ambos ",
    " ambas ",
    " comparar ",
    " mismo ",
    " misma ",
    " diferente ",
]

GENERIC_INTRO_MARKERS = [
    "this question asks",
    "to answer this question",
    "the answer depends",
    "we need to determine",
    "la pregunta pide",
    "para responder",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_for_rules(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        if isinstance(value, str):
            text = value.strip().replace(",", ".")
            if text.endswith("%"):
                number = float(text[:-1]) / 100.0
            else:
                number = float(text)
        else:
            number = float(value)
    except (TypeError, ValueError):
        return default

    if 1.0 < number <= 100.0:
        number = number / 100.0

    return min(max(number, 0.0), 1.0)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)

    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False

    return default


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


def contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def count_capitalized_entities(text: str) -> int:
    """
    Heurística simple de entidades nombradas.
    No es NER real; sirve como baseline reproducible.
    """
    pattern = r"\b(?:[A-Z][a-zA-Z'’.-]+(?:\s+(?:of|the|and|de|del|la|le|du|von|van|da|di|[A-Z][a-zA-Z'’.-]+))*)"
    matches = re.findall(pattern, clean_text(text))

    stop_like = {
        "The", "This", "That", "These", "Those", "A", "An",
        "Question", "Answer", "It", "He", "She", "They",
        "What", "When", "Where", "Who", "Which", "Why", "How",
        "If", "In", "On", "For", "To",
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


def has_number_or_date(text: str) -> bool:
    text = clean_text(text)
    if re.search(r"\b\d{2,4}\b", text):
        return True
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def normalize_task_type(task_type: str | None) -> str:
    task_type = clean_text(task_type).lower()
    if task_type in VALID_TASK_TYPES:
        return task_type
    return "open_qa"


def normalize_retrieval_bias(retrieval_bias: str | None) -> str:
    retrieval_bias = clean_text(retrieval_bias).lower()
    if retrieval_bias in VALID_RETRIEVAL_BIASES:
        return retrieval_bias
    return "balanced"


def is_explicit_document_request(question_lower: str, candidate_lower: str) -> bool:
    return contains_any(question_lower, DOCUMENTAL_CUES) or contains_any(candidate_lower, DOCUMENTAL_CUES)


def should_suppress_weak_retrieval(
    *,
    rule_name: str,
    retrieval_bias: str,
    task_type: str,
    candidate_confidence: float | None,
) -> bool:
    """
    Reduce over-retrieval en casos directos.

    Las reglas named_entity/question_multi_entity son señales débiles. En tareas
    directas o multiple-choice no deberían disparar retrieval salvo que haya una
    señal documental explícita.
    """
    confidence = safe_float(candidate_confidence, default=-1.0)

    weak_rules = {"named_entity", "question_multi_entity", "comparison_cue"}
    factual_rules = {"factual_relation_cue", "number_or_date"}

    if task_type == "multiple_choice":
        return rule_name in weak_rules | factual_rules

    if retrieval_bias == "conservative":
        if rule_name in weak_rules:
            return True
        if rule_name in factual_rules and confidence >= 0.70:
            return True

    if retrieval_bias == "balanced":
        if rule_name in weak_rules and confidence >= 0.85:
            return True

    return False


def suppress_decision(
    *,
    original_rule_name: str,
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "needs_retrieval": False,
        "reason": reason,
        "confidence": confidence,
        "policy_source": "rules",
        "rule_name": f"suppressed_{original_rule_name}",
    }


def needs_retrieval_rules(
    *,
    question: str,
    candidate_sentence: str,
    partial_answer: str = "",
    candidate_confidence: float | None = None,
    task_type: str = "open_qa",
    retrieval_bias: str = "balanced",
) -> dict[str, Any]:
    candidate = clean_text(candidate_sentence)
    lower = normalize_for_rules(candidate)
    lower_padded = f" {lower} "
    question_lower = normalize_for_rules(question)
    task_type = normalize_task_type(task_type)
    retrieval_bias = normalize_retrieval_bias(retrieval_bias)
    explicit_document_request = is_explicit_document_request(question_lower, lower)

    if not candidate:
        return {
            "needs_retrieval": False,
            "reason": "No hay oración candidata.",
            "confidence": 1.0,
            "policy_source": "rules",
            "rule_name": "empty_candidate",
        }

    if contains_any(lower, ABSTENTION_MARKERS):
        return {
            "needs_retrieval": False,
            "reason": "La oración candidata expresa abstención por falta de información.",
            "confidence": 0.94,
            "policy_source": "rules",
            "rule_name": "abstention_sentence",
        }

    if contains_any(lower, CLARIFICATION_MARKERS):
        return {
            "needs_retrieval": False,
            "reason": "La oración candidata pide aclaración; no requiere retrieval.",
            "confidence": 0.94,
            "policy_source": "rules",
            "rule_name": "clarification_sentence",
        }

    if contains_any(lower, GENERIC_INTRO_MARKERS):
        return {
            "needs_retrieval": False,
            "reason": "La oración candidata es introductoria o metadiscursiva.",
            "confidence": 0.82,
            "policy_source": "rules",
            "rule_name": "generic_intro",
        }

    # Hard gate para multiple-choice:
    # en S3 experimental tratamos MMLU como directo, no como RAG.
    # Solo recuperaríamos si hubiera un pedido documental explícito real.
    if task_type == "multiple_choice" and not explicit_document_request:
        return {
            "needs_retrieval": False,
            "reason": "La tarea es multiple-choice directa y no pide evidencia documental explícita.",
            "confidence": 0.95,
            "policy_source": "rules",
            "rule_name": "multiple_choice_direct_no_retrieval",
        }

    if explicit_document_request:
        if task_type == "multiple_choice":
            return {
                "needs_retrieval": False,
                "reason": "Se ignora documental_cue en multiple-choice porque la tarea se evalúa como directa.",
                "confidence": 0.92,
                "policy_source": "rules",
                "rule_name": "suppressed_documental_cue_multiple_choice",
            }

        return {
            "needs_retrieval": True,
            "reason": "La pregunta u oración menciona corpus, documentos, contexto o evidencia.",
            "confidence": 0.90 if retrieval_bias == "aggressive" else 0.86,
            "policy_source": "rules",
            "rule_name": "documental_cue",
        }

    # Hard gate para preguntas directas en modo conservador:
    # no se recupera por entidades, fechas o relaciones factuales; se responde directo.
    if task_type == "open_direct" and retrieval_bias == "conservative":
        return {
            "needs_retrieval": False,
            "reason": "La tarea es open_direct con sesgo conservador; no se recupera salvo pedido documental explícito.",
            "confidence": 0.92,
            "policy_source": "rules",
            "rule_name": "open_direct_conservative_no_retrieval",
        }

    if has_number_or_date(candidate):
        if should_suppress_weak_retrieval(
            rule_name="number_or_date",
            retrieval_bias=retrieval_bias,
            task_type=task_type,
            candidate_confidence=candidate_confidence,
        ):
            return suppress_decision(
                original_rule_name="number_or_date",
                reason="Se suprime retrieval por número/fecha en una tarea directa con confianza suficiente.",
                confidence=0.78,
            )

        return {
            "needs_retrieval": True,
            "reason": "La oración contiene números, fechas o cantidades que requieren soporte.",
            "confidence": 0.84,
            "policy_source": "rules",
            "rule_name": "number_or_date",
        }

    if contains_any(lower_padded, FACTUAL_RELATION_CUES):
        if should_suppress_weak_retrieval(
            rule_name="factual_relation_cue",
            retrieval_bias=retrieval_bias,
            task_type=task_type,
            candidate_confidence=candidate_confidence,
        ):
            return suppress_decision(
                original_rule_name="factual_relation_cue",
                reason="Se suprime retrieval por relación factual en modo conservador/directo.",
                confidence=0.78,
            )

        return {
            "needs_retrieval": True,
            "reason": "La oración contiene una relación factual específica.",
            "confidence": 0.83,
            "policy_source": "rules",
            "rule_name": "factual_relation_cue",
        }

    if contains_any(lower_padded, COMPARISON_CUES):
        if should_suppress_weak_retrieval(
            rule_name="comparison_cue",
            retrieval_bias=retrieval_bias,
            task_type=task_type,
            candidate_confidence=candidate_confidence,
        ):
            return suppress_decision(
                original_rule_name="comparison_cue",
                reason="Se suprime retrieval por comparación débil en modo conservador/directo.",
                confidence=0.78,
            )

        return {
            "needs_retrieval": True,
            "reason": "La oración contiene comparación o relación entre hechos.",
            "confidence": 0.78,
            "policy_source": "rules",
            "rule_name": "comparison_cue",
        }

    entity_count = count_capitalized_entities(candidate)
    if entity_count >= 1:
        if should_suppress_weak_retrieval(
            rule_name="named_entity",
            retrieval_bias=retrieval_bias,
            task_type=task_type,
            candidate_confidence=candidate_confidence,
        ):
            return suppress_decision(
                original_rule_name="named_entity",
                reason="Se suprime retrieval por entidad nombrada aislada en modo conservador/directo.",
                confidence=0.80,
            )

        return {
            "needs_retrieval": True,
            "reason": "La oración contiene entidades nombradas y parece factual.",
            "confidence": 0.74,
            "policy_source": "rules",
            "rule_name": "named_entity",
        }

    q_entity_count = count_capitalized_entities(question)
    if q_entity_count >= 2 and len(candidate.split()) >= 6:
        if should_suppress_weak_retrieval(
            rule_name="question_multi_entity",
            retrieval_bias=retrieval_bias,
            task_type=task_type,
            candidate_confidence=candidate_confidence,
        ):
            return suppress_decision(
                original_rule_name="question_multi_entity",
                reason="Se suprime retrieval por múltiples entidades en modo conservador/directo.",
                confidence=0.78,
            )

        return {
            "needs_retrieval": True,
            "reason": "La pregunta contiene múltiples entidades y la oración parece parte de una respuesta factual.",
            "confidence": 0.70,
            "policy_source": "rules",
            "rule_name": "question_multi_entity",
        }

    return {
        "needs_retrieval": False,
        "reason": "No se detectaron señales fuertes de factualidad específica.",
        "confidence": 0.62,
        "policy_source": "rules",
        "rule_name": "fallback_no_retrieval",
    }


def validate_retrieval_decision(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "needs_retrieval": coerce_bool(obj.get("needs_retrieval"), default=False),
        "reason": clean_text(obj.get("reason", "")) or "Sin razón provista.",
        "confidence": safe_float(obj.get("confidence", 0.0), default=0.0),
    }


def needs_retrieval_llm(
    *,
    question: str,
    candidate_sentence: str,
    partial_answer: str = "",
    candidate_confidence: float | None = None,
    task_type: str = "open_qa",
    retrieval_bias: str = "balanced",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Usa LLM para decidir retrieval. Si falla, cae a rules.
    """
    rules_fallback = needs_retrieval_rules(
        question=question,
        candidate_sentence=candidate_sentence,
        partial_answer=partial_answer,
        candidate_confidence=candidate_confidence,
        task_type=task_type,
        retrieval_bias=retrieval_bias,
    )

    if dry_run:
        return {
            **rules_fallback,
            "policy_source": "dry_run_rules",
            "decision_raw_output": json.dumps(rules_fallback, ensure_ascii=False),
            "decision_parse_error": "",
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
            "decision_total_tokens": 0,
            "decision_latency_seconds": 0.0,
        }

    try:
        from direct_llm import ask_direct_llm_with_metadata
    except ModuleNotFoundError as exc:
        return {
            **rules_fallback,
            "policy_source": "rules_fallback_missing_direct_llm",
            "decision_error": str(exc),
        }

    prompt = build_retrieval_decision_prompt(
        question=question,
        partial_answer=partial_answer,
        candidate_sentence=candidate_sentence,
        task_type=task_type,
        retrieval_bias=retrieval_bias,
        candidate_confidence=candidate_confidence,
    )

    result = ask_direct_llm_with_metadata(
        prompt,
        model=model,
        system_prompt=S3_RETRIEVAL_DECISION_SYSTEM_PROMPT,
        max_retries=max_retries,
    )

    raw_output = clean_text(result.get("raw_output", ""))
    obj, parse_error = load_json_object(raw_output)

    if obj is None:
        return {
            **rules_fallback,
            "policy_source": "rules_fallback_after_llm_parse_error",
            "decision_raw_output": raw_output,
            "decision_parse_error": parse_error,
            "decision_input_tokens": result.get("input_tokens"),
            "decision_output_tokens": result.get("output_tokens"),
            "decision_total_tokens": result.get("total_tokens"),
            "decision_latency_seconds": result.get("latency_seconds"),
        }

    parsed = validate_retrieval_decision(obj)

    return {
        **parsed,
        "policy_source": "llm",
        "rule_name": rules_fallback.get("rule_name", ""),
        "decision_raw_output": raw_output,
        "decision_parse_error": "",
        "decision_input_tokens": result.get("input_tokens"),
        "decision_output_tokens": result.get("output_tokens"),
        "decision_total_tokens": result.get("total_tokens"),
        "decision_latency_seconds": result.get("latency_seconds"),
    }


def needs_retrieval(
    *,
    question: str,
    candidate_sentence: str,
    partial_answer: str = "",
    candidate_confidence: float | None = None,
    task_type: str = "open_qa",
    retrieval_bias: str = "balanced",
    strategy: RetrievalStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    hybrid_confidence_threshold: float = 0.86,
    dry_run: bool = False,
) -> dict[str, Any]:
    if strategy not in VALID_RETRIEVAL_STRATEGIES:
        raise ValueError(f"Estrategia inválida: {strategy}. Usá rules, llm o hybrid.")

    rules_decision = needs_retrieval_rules(
        question=question,
        candidate_sentence=candidate_sentence,
        partial_answer=partial_answer,
        candidate_confidence=candidate_confidence,
        task_type=task_type,
        retrieval_bias=retrieval_bias,
    )

    if strategy == "rules":
        return rules_decision

    if strategy == "llm":
        return needs_retrieval_llm(
            question=question,
            candidate_sentence=candidate_sentence,
            partial_answer=partial_answer,
            candidate_confidence=candidate_confidence,
            task_type=task_type,
            retrieval_bias=retrieval_bias,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
        )

    if safe_float(rules_decision.get("confidence"), 0.0) >= hybrid_confidence_threshold:
        return {
            **rules_decision,
            "policy_source": "rules_high_confidence",
        }

    try:
        return needs_retrieval_llm(
            question=question,
            candidate_sentence=candidate_sentence,
            partial_answer=partial_answer,
            candidate_confidence=candidate_confidence,
            task_type=task_type,
            retrieval_bias=retrieval_bias,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
        )
    except Exception as exc:
        return {
            **rules_decision,
            "policy_source": "rules_fallback_after_llm_error",
            "decision_error": str(exc),
        }

#!/usr/bin/env python3
"""
prompts_s3.py

Prompts para S3: FLARE-like / FLARE-inspired active retrieval.

Este archivo solo contiene strings y builders de prompts.
La lógica de retrieval activo queda en active_retrieval_policy_s3.py
y flare_controller_s3.py.
"""

from __future__ import annotations

from typing import Any


S3_CANDIDATE_SYSTEM_PROMPT = """
Sos el generador parcial de un sistema experimental S3 FLARE-like.

Tu tarea:
- No generes la respuesta completa.
- Generá solamente la próxima oración corta o el próximo claim necesario.
- Si la respuesta ya está completa, devolvé done=true.
- No inventes evidencia ni citas.
- Si la pregunta es ambigua o no hay información suficiente, podés generar una oración de aclaración o abstención.
- Conservá el idioma de la pregunta siempre que sea posible.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "done": false,
  "candidate_sentence": "...",
  "confidence": 0.0
}
""".strip()


S3_RETRIEVAL_DECISION_SYSTEM_PROMPT = """
Sos el módulo de decisión de recuperación activa de un sistema S3 FLARE-like.

Tu tarea:
- Decidir si una oración candidata necesita evidencia externa antes de aceptarse.
- No respondas la pregunta.
- No verifiques el claim todavía.
- Solo decidí si conviene recuperar documentos.

Criterios para needs_retrieval=true:
- La oración contiene un hecho específico.
- La oración contiene nombres propios, fechas, números, lugares o entidades.
- La oración afirma una relación factual: nació en, escrito por, dirigido por, ubicado en, fundado por, etc.
- La oración compara entidades o conecta varios hechos.
- La oración depende explícitamente de corpus, documentos, evidencia o contexto.

Criterios para needs_retrieval=false:
- La oración es puramente introductoria.
- La oración es una aclaración al usuario.
- La oración es una abstención por falta de información.
- La oración es conceptual/general y no depende de evidencia documental.

Devolvé únicamente JSON válido:
{
  "needs_retrieval": true,
  "reason": "...",
  "confidence": 0.0
}
""".strip()


S3_REGENERATE_SYSTEM_PROMPT = """
Sos el regenerador/verificador local de una oración en un sistema S3 FLARE-like.

Tu tarea:
- Revisar una oración candidata usando únicamente la evidencia recuperada.
- Si la evidencia soporta la oración, podés mantenerla.
- Si la evidencia contradice la oración, corregila.
- Si la evidencia no alcanza, devolvé una oración de abstención específica.
- En ese caso, obligatoriamente usá abstain_sentence=true y support_status="not_enough_info".
- Si la evidencia contradice la oración, usá support_status="refuted" o "corrected" según corresponda.
- La confidence debe medir confianza en la oración final y su estado de soporte, no confianza en el claim original.
- No agregues información externa.
- No inventes datos.
- Conservá el idioma de la pregunta siempre que sea posible.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "final_sentence": "...",
  "used_evidence": true,
  "confidence": 0.0,
  "abstain_sentence": false,
  "support_status": "supported|corrected|not_enough_info|refuted"
}
""".strip()


def clean_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_candidate_prompt(
    *,
    question: str,
    partial_answer: str,
    step: int,
    max_steps: int,
    task_type: str = "open_qa",
) -> str:
    partial = clean_prompt_text(partial_answer) or "<empty>"
    task_type = clean_prompt_text(task_type) or "open_qa"

    if task_type == "multiple_choice":
        task_rules = """Reglas específicas para multiple-choice:
- La pregunta tiene opciones.
- La oración candidata debe proponer una opción concreta, por ejemplo: "The answer is D."
- No expliques largamente.
- No uses retrieval salvo que la pregunta pida explícitamente usar documentos/corpus/contexto."""
    elif task_type == "open_direct":
        task_rules = """Reglas específicas para pregunta directa:
- Respondé con conocimiento general del modelo.
- No menciones corpus, documentos ni evidencia externa.
- Solo pedí evidencia si la pregunta lo solicita explícitamente."""
    elif task_type == "open_retrieval":
        task_rules = """Reglas específicas para pregunta con posible retrieval:
- Podés generar un claim candidato verificable.
- Si el claim depende de evidencia documental, luego será verificado con retrieval."""
    elif task_type == "ambiguous":
        task_rules = """Reglas específicas para pregunta ambigua:
- Si falta una entidad o referencia central, generá una aclaración breve.
- No inventes la entidad omitida."""
    else:
        task_rules = """Reglas específicas:
- Tratá la pregunta como QA abierta general.
- Usá retrieval solo cuando la afirmación necesite evidencia externa."""

    return f"""Generá la próxima oración candidata para responder la pregunta.

Reglas:
- Generá solo una oración breve.
- No generes la respuesta completa.
- No repitas información ya incluida en la respuesta parcial.
- Si la respuesta ya está completa, devolvé done=true.
- Si falta evidencia o la pregunta es ambigua, podés generar una oración de abstención/aclaración.
- Devolvé únicamente JSON válido.

Tipo de tarea:
{task_type}

{task_rules}

Pregunta:
{question}

Respuesta parcial actual:
{partial}

Paso actual:
{step} de {max_steps}

Formato obligatorio:
{{
  "done": false,
  "candidate_sentence": "...",
  "confidence": 0.0
}}"""


def build_retrieval_decision_prompt(
    *,
    question: str,
    partial_answer: str,
    candidate_sentence: str,
    task_type: str = "open_qa",
    retrieval_bias: str = "balanced",
    candidate_confidence: float | None = None,
) -> str:
    partial = clean_prompt_text(partial_answer) or "<empty>"
    task_type = clean_prompt_text(task_type) or "open_qa"
    retrieval_bias = clean_prompt_text(retrieval_bias) or "balanced"
    confidence_text = "unknown" if candidate_confidence is None else f"{float(candidate_confidence):.3f}"

    return f"""Decidí si la oración candidata necesita recuperación de evidencia.

Pregunta:
{question}

Respuesta parcial:
{partial}

Oración candidata:
{candidate_sentence}

Tipo de tarea:
{task_type}

Sesgo de retrieval:
{retrieval_bias}

Confianza de la oración candidata:
{confidence_text}

Reglas adicionales:
- Si task_type="multiple_choice", no recuperes salvo que la pregunta mencione explícitamente documentos, corpus, evidencia o contexto externo.
- Si retrieval_bias="conservative", no recuperes por señales débiles como nombres propios aislados.
- Si retrieval_bias="aggressive", recuperá cuando el claim sea factual, multi-hop o documental.
- Si la oración candidata tiene alta confianza y la tarea es directa, preferí no recuperar.

Devolvé únicamente JSON válido:
{{
  "needs_retrieval": true,
  "reason": "...",
  "confidence": 0.0
}}"""


def build_context_block(
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int = 900,
) -> str:
    blocks: list[str] = []

    for i, item in enumerate(retrieved_chunks, start=1):
        chunk_id = clean_prompt_text(item.get("chunk_id", ""))
        title = clean_prompt_text(item.get("title", ""))
        score = item.get("score", "")
        text = clean_prompt_text(item.get("text", ""))

        if max_chars_per_chunk > 0 and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " [...]"

        try:
            score_text = f"{float(score):.6f}"
        except (TypeError, ValueError):
            score_text = ""

        blocks.append(
            f"""[{i}]
chunk_id: {chunk_id}
title: {title}
score: {score_text}
text: {text}"""
        )

    return "\n\n".join(blocks)


def build_regenerate_prompt(
    *,
    question: str,
    partial_answer: str,
    candidate_sentence: str,
    retrieved_chunks: list[dict[str, Any]],
    max_chars_per_chunk: int = 900,
) -> str:
    partial = clean_prompt_text(partial_answer) or "<empty>"
    context_block = build_context_block(
        retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    ) or "<no evidence retrieved>"

    return f"""Revisá y, si hace falta, corregí la oración candidata usando únicamente la evidencia recuperada.

Pregunta:
{question}

Respuesta parcial:
{partial}

Oración candidata:
{candidate_sentence}

Evidencia recuperada:
{context_block}

Reglas:
- Si la evidencia soporta la oración, mantenela o reescribila de forma más precisa.
- Si la evidencia contradice la oración, corregila.
- Si la evidencia no permite sostener la oración, devolvé una oración breve indicando que no hay información suficiente.
- En ese caso, obligatoriamente devolvé abstain_sentence=true y support_status="not_enough_info".
- Si la evidencia contradice la oración, usá support_status="refuted" o corregila con support_status="corrected".
- La confidence debe medir confianza en la oración final y su estado de soporte, no confianza en el claim original.
- No uses conocimiento externo.
- No inventes datos.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{{
  "final_sentence": "...",
  "used_evidence": true,
  "confidence": 0.0,
  "abstain_sentence": false,
  "support_status": "supported|corrected|not_enough_info|refuted"
}}"""

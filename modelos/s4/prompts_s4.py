#!/usr/bin/env python3
"""
prompts_s4.py

Prompts para S4: FIRE-like / FIRE-inspired claim verification.

Este archivo solo contiene:
- system prompts;
- builders de prompts;
- utilidades simples de formateo textual.

No ejecuta retrieval.
No llama al LLM.
No parsea salidas.
No evalúa resultados.

La lógica del modelo S4 debería quedar después en:
- claim_extractor_s4.py
- fire_verifier_s4.py
- fire_controller_s4.py
- run_s4_fire_like.py
"""

from __future__ import annotations

import argparse
import json
from typing import Any


DEFAULT_MAX_CHARS_PER_CHUNK = 900
DEFAULT_MAX_CLAIMS = 5

VALID_CLAIM_TYPES = {
    "factual",
    "answer_choice",
    "abstention",
    "clarification",
    "meta",
}

VALID_IMPORTANCE_LEVELS = {
    "core",
    "supporting",
    "low",
}

VALID_VERDICTS = {
    "supported",
    "refuted",
    "not_enough_info",
}

VALID_FINAL_DECISIONS = {
    "unchanged",
    "corrected",
    "abstained",
    "clarification_kept",
    "no_claims",
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

S4_CLAIM_EXTRACTION_SYSTEM_PROMPT = """
Sos el extractor de claims de un sistema experimental S4 FIRE-like.

Tu tarea:
- Recibir una pregunta y una respuesta inicial generada por otro sistema.
- Dividir la respuesta inicial en claims atómicos y verificables.
- Un claim atómico debe expresar una sola afirmación factual.
- No verifiques los claims todavía.
- No corrijas la respuesta.
- No agregues información nueva.
- No inventes claims que no estén en la respuesta.
- Si la respuesta es una abstención, extraé un claim de tipo "abstention".
- Si la respuesta pide aclaración, extraé un claim de tipo "clarification".
- Si la respuesta es multiple-choice, extraé la opción elegida como un claim de tipo "answer_choice".
- Conservá el idioma de la pregunta y la respuesta siempre que sea posible.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "claims": [
    {
      "claim_id": "c1",
      "claim_text": "...",
      "claim_type": "factual|answer_choice|abstention|clarification|meta",
      "requires_evidence": true,
      "importance": "core|supporting|low"
    }
  ],
  "needs_verification": true,
  "extraction_notes": "..."
}
""".strip()


S4_CLAIM_VERIFICATION_SYSTEM_PROMPT = """
Sos el verificador de claims de un sistema experimental S4 FIRE-like.

Tu tarea:
- Verificar un claim usando únicamente la evidencia recuperada incluida en el prompt.
- No uses conocimiento externo.
- No inventes evidencia.
- No respondas la pregunta completa.
- Solo emití un veredicto sobre el claim.

Veredictos válidos:
- supported: la evidencia apoya claramente el claim.
- refuted: la evidencia contradice claramente el claim.
- not_enough_info: la evidencia no alcanza para confirmar ni refutar el claim.

Reglas importantes:
- La ausencia de evidencia NO significa que el claim sea falso.
- Si la evidencia es insuficiente, usá not_enough_info.
- Si necesitás más evidencia, marcá needs_more_evidence=true y proponé next_search_query.
- Si el claim está claramente soportado o refutado, needs_more_evidence debe ser false.
- supporting_chunk_ids y refuting_chunk_ids deben contener solo chunk_id presentes en la evidencia.
- Conservá el idioma de la pregunta siempre que sea posible.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "claim_id": "c1",
  "claim_text": "...",
  "verdict": "supported|refuted|not_enough_info",
  "confidence": 0.0,
  "rationale": "...",
  "supporting_chunk_ids": [],
  "refuting_chunk_ids": [],
  "needs_more_evidence": false,
  "next_search_query": null
}
""".strip()


S4_FINAL_REPAIR_SYSTEM_PROMPT = """
Sos el reparador final de respuestas de un sistema experimental S4 FIRE-like.

Tu tarea:
- Recibir una pregunta, una respuesta inicial y los resultados de verificación de claims.
- Producir una respuesta final verificada.
- Si todos los claims centrales están soportados, podés mantener la respuesta.
- Si un claim central fue refutado, corregilo si la evidencia permite corregirlo.
- Si un claim central no tiene evidencia suficiente, eliminá ese claim o abstente.
- Si la respuesta inicial ya era una abstención correcta, mantenela.
- Si la respuesta inicial pedía aclaración y la pregunta sigue siendo ambigua, mantené la aclaración.
- No agregues información externa.
- No inventes datos.
- No cites chunk_id en la respuesta final salvo que sea necesario.
- Conservá el idioma de la pregunta siempre que sea posible.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false,
  "final_decision": "unchanged|corrected|abstained|clarification_kept|no_claims",
  "correction_applied": false,
  "unsupported_claims_removed": [],
  "corrected_claims": [],
  "evidence_ids": []
}
""".strip()


S4_SEARCH_QUERY_SYSTEM_PROMPT = """
Sos el generador de queries de búsqueda para un sistema S4 FIRE-like.

Tu tarea:
- Recibir una pregunta, un claim y el estado actual de verificación.
- Generar una query breve y específica para recuperar evidencia útil.
- La query debe buscar evidencia para confirmar o refutar el claim.
- No respondas la pregunta.
- No verifiques el claim.
- No agregues información no presente en la pregunta o en el claim.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{
  "search_query": "...",
  "reason": "..."
}
""".strip()


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def clean_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def truncate_text(text: Any, max_chars: int) -> str:
    text = clean_prompt_text(text)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


def safe_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, indent=indent)


def normalize_optional_text(value: Any, fallback: str = "<empty>") -> str:
    text = clean_prompt_text(value)
    return text if text else fallback


# ---------------------------------------------------------------------------
# Builders de bloques
# ---------------------------------------------------------------------------

def build_context_block(
    retrieved_chunks: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> str:
    """
    Convierte chunks recuperados en bloque textual para el verificador.

    Espera items con algunas de estas claves:
    - rank
    - chunk_id
    - doc_id
    - title
    - score
    - text
    - source
    - topic
    """

    if not retrieved_chunks:
        return "<no evidence provided>"

    blocks: list[str] = []

    for i, item in enumerate(retrieved_chunks, start=1):
        rank = clean_prompt_text(item.get("rank", i))
        chunk_id = clean_prompt_text(item.get("chunk_id", ""))
        doc_id = clean_prompt_text(item.get("doc_id", ""))
        title = clean_prompt_text(item.get("title", ""))
        source = clean_prompt_text(item.get("source", ""))
        topic = clean_prompt_text(item.get("topic", ""))
        text = truncate_text(item.get("text", ""), max_chars_per_chunk)

        score_raw = item.get("score", "")
        try:
            score = f"{float(score_raw):.6f}"
        except (TypeError, ValueError):
            score = clean_prompt_text(score_raw)

        blocks.append(
            f"""[{rank}]
chunk_id: {chunk_id}
doc_id: {doc_id}
title: {title}
source: {source}
topic: {topic}
score: {score}
text: {text}""".strip()
        )

    return "\n\n".join(blocks)


def build_claims_block(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "<no claims>"

    lines: list[str] = []
    for claim in claims:
        claim_id = clean_prompt_text(claim.get("claim_id", ""))
        claim_text = clean_prompt_text(claim.get("claim_text", ""))
        claim_type = clean_prompt_text(claim.get("claim_type", ""))
        importance = clean_prompt_text(claim.get("importance", ""))
        requires_evidence = claim.get("requires_evidence", "")

        lines.append(
            f"""claim_id: {claim_id}
claim_text: {claim_text}
claim_type: {claim_type}
importance: {importance}
requires_evidence: {requires_evidence}""".strip()
        )

    return "\n\n".join(lines)


def build_verification_results_block(
    claim_results: list[dict[str, Any]],
) -> str:
    if not claim_results:
        return "<no verification results>"

    safe_results: list[dict[str, Any]] = []
    for item in claim_results:
        safe_results.append(
            {
                "claim_id": item.get("claim_id", ""),
                "claim_text": item.get("claim_text", ""),
                "verdict": item.get("verdict", ""),
                "confidence": item.get("confidence", ""),
                "rationale": item.get("rationale", ""),
                "supporting_chunk_ids": item.get("supporting_chunk_ids", []),
                "refuting_chunk_ids": item.get("refuting_chunk_ids", []),
                "needs_more_evidence": item.get("needs_more_evidence", False),
                "rounds": item.get("rounds", item.get("num_rounds", "")),
            }
        )

    return safe_json_dumps(safe_results, indent=2)


# ---------------------------------------------------------------------------
# Builders de prompts S4
# ---------------------------------------------------------------------------

def build_claim_extraction_prompt(
    *,
    question: str,
    initial_answer: str,
    source_system: str = "",
    question_type: str = "",
    expected_behavior: str = "",
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> str:
    question = normalize_optional_text(question)
    initial_answer = normalize_optional_text(initial_answer)
    source_system = normalize_optional_text(source_system, fallback="unknown")
    question_type = normalize_optional_text(question_type, fallback="unknown")
    expected_behavior = normalize_optional_text(expected_behavior, fallback="unknown")

    return f"""Extraé claims atómicos verificables desde la respuesta inicial.

Pregunta:
{question}

Respuesta inicial:
{initial_answer}

Sistema que generó la respuesta inicial:
{source_system}

Tipo de pregunta:
{question_type}

Comportamiento esperado, si está disponible:
{expected_behavior}

Cantidad máxima de claims:
{max_claims}

Reglas:
- No generes más de {max_claims} claims.
- Priorizá claims centrales para responder la pregunta.
- No incluyas frases puramente estilísticas.
- No transformes una respuesta incierta en un claim factual fuerte.
- Si la respuesta inicial dice que no hay evidencia suficiente, extraé un claim de tipo "abstention".
- Si la respuesta inicial pide aclaración, extraé un claim de tipo "clarification".
- Si la respuesta inicial es una letra A/B/C/D, extraé la selección como claim de tipo "answer_choice".
- Devolvé únicamente JSON válido.

Formato obligatorio:
{{
  "claims": [
    {{
      "claim_id": "c1",
      "claim_text": "...",
      "claim_type": "factual|answer_choice|abstention|clarification|meta",
      "requires_evidence": true,
      "importance": "core|supporting|low"
    }}
  ],
  "needs_verification": true,
  "extraction_notes": "..."
}}"""


def build_claim_verification_prompt(
    *,
    question: str,
    claim: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    round_id: int = 1,
    max_rounds: int = 2,
    previous_verifications: list[dict[str, Any]] | None = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> str:
    question = normalize_optional_text(question)
    claim_id = clean_prompt_text(claim.get("claim_id", "c1")) or "c1"
    claim_text = normalize_optional_text(claim.get("claim_text", ""))
    claim_type = normalize_optional_text(claim.get("claim_type", ""), fallback="factual")
    importance = normalize_optional_text(claim.get("importance", ""), fallback="core")
    requires_evidence = claim.get("requires_evidence", True)

    evidence_block = build_context_block(
        retrieved_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    previous_block = (
        safe_json_dumps(previous_verifications or [], indent=2)
        if previous_verifications
        else "<no previous verification rounds>"
    )

    return f"""Verificá el siguiente claim usando únicamente la evidencia recuperada.

Pregunta original:
{question}

Claim:
claim_id: {claim_id}
claim_text: {claim_text}
claim_type: {claim_type}
importance: {importance}
requires_evidence: {requires_evidence}

Ronda de verificación:
{round_id} de {max_rounds}

Verificaciones previas:
{previous_block}

Evidencia recuperada:
{evidence_block}

Reglas:
- No uses conocimiento externo.
- No inventes evidencia.
- Si la evidencia apoya claramente el claim, verdict="supported".
- Si la evidencia contradice claramente el claim, verdict="refuted".
- Si la evidencia no alcanza, verdict="not_enough_info".
- Si verdict="not_enough_info" y todavía podrían buscarse mejores documentos, usá needs_more_evidence=true.
- Si needs_more_evidence=true, next_search_query debe ser una query concreta y breve.
- Si needs_more_evidence=false, next_search_query debe ser null.
- supporting_chunk_ids y refuting_chunk_ids solo pueden contener chunk_id presentes en la evidencia.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{{
  "claim_id": "{claim_id}",
  "claim_text": "{claim_text}",
  "verdict": "supported|refuted|not_enough_info",
  "confidence": 0.0,
  "rationale": "...",
  "supporting_chunk_ids": [],
  "refuting_chunk_ids": [],
  "needs_more_evidence": false,
  "next_search_query": null
}}"""


def build_search_query_prompt(
    *,
    question: str,
    claim: dict[str, Any],
    previous_evidence: list[dict[str, Any]] | None = None,
    previous_verdict: str = "",
    previous_rationale: str = "",
    max_chars_per_chunk: int = 500,
) -> str:
    question = normalize_optional_text(question)
    claim_id = clean_prompt_text(claim.get("claim_id", "c1")) or "c1"
    claim_text = normalize_optional_text(claim.get("claim_text", ""))
    previous_verdict = normalize_optional_text(previous_verdict, fallback="unknown")
    previous_rationale = normalize_optional_text(previous_rationale, fallback="<empty>")

    evidence_block = build_context_block(
        previous_evidence or [],
        max_chars_per_chunk=max_chars_per_chunk,
    )

    return f"""Generá una query de búsqueda para conseguir mejor evidencia sobre el claim.

Pregunta original:
{question}

Claim:
claim_id: {claim_id}
claim_text: {claim_text}

Veredicto previo:
{previous_verdict}

Rationale previo:
{previous_rationale}

Evidencia ya revisada:
{evidence_block}

Reglas:
- La query debe ser corta.
- La query debe incluir las entidades centrales del claim.
- No agregues datos que no estén en la pregunta o en el claim.
- No respondas la pregunta.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{{
  "search_query": "...",
  "reason": "..."
}}"""


def build_final_repair_prompt(
    *,
    question: str,
    initial_answer: str,
    claim_results: list[dict[str, Any]],
    expected_behavior: str = "",
    source_system: str = "",
) -> str:
    question = normalize_optional_text(question)
    initial_answer = normalize_optional_text(initial_answer)
    expected_behavior = normalize_optional_text(expected_behavior, fallback="unknown")
    source_system = normalize_optional_text(source_system, fallback="unknown")

    verification_block = build_verification_results_block(claim_results)

    return f"""Construí la respuesta final verificada a partir de la respuesta inicial y los resultados de verificación.

Pregunta:
{question}

Respuesta inicial:
{initial_answer}

Sistema fuente:
{source_system}

Comportamiento esperado, si está disponible:
{expected_behavior}

Resultados de verificación de claims:
{verification_block}

Reglas:
- Si los claims centrales están supported, mantené la respuesta o hacé cambios mínimos.
- Si un claim central está refuted, corregilo solo si la evidencia permite corregirlo.
- Si un claim central está not_enough_info, eliminá ese claim o abstente.
- Si la respuesta inicial ya era abstención y los claims no tienen evidencia suficiente, mantené la abstención.
- Si la respuesta inicial pedía aclaración y la pregunta sigue siendo ambigua, mantené la aclaración.
- No agregues información externa.
- No inventes datos.
- La respuesta final debe ser breve y directa.
- Devolvé únicamente JSON válido.

Formato obligatorio:
{{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false,
  "final_decision": "unchanged|corrected|abstained|clarification_kept|no_claims",
  "correction_applied": false,
  "unsupported_claims_removed": [],
  "corrected_claims": [],
  "evidence_ids": []
}}"""


# ---------------------------------------------------------------------------
# Self-test liviano
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    sample_question = "According to the available corpus, who composed La traviata?"
    sample_answer = "La traviata was composed by Giuseppe Verdi."
    sample_claim = {
        "claim_id": "c1",
        "claim_text": "La traviata was composed by Giuseppe Verdi.",
        "claim_type": "factual",
        "requires_evidence": True,
        "importance": "core",
    }
    sample_chunks = [
        {
            "rank": 1,
            "chunk_id": "chunk_test_001",
            "doc_id": "doc_test",
            "title": "La traviata",
            "score": 0.91,
            "text": "La traviata is an opera in three acts by Giuseppe Verdi.",
            "source": "self_test",
            "topic": "opera",
        }
    ]
    sample_claim_results = [
        {
            "claim_id": "c1",
            "claim_text": "La traviata was composed by Giuseppe Verdi.",
            "verdict": "supported",
            "confidence": 0.92,
            "rationale": "The evidence states that La traviata is by Giuseppe Verdi.",
            "supporting_chunk_ids": ["chunk_test_001"],
            "refuting_chunk_ids": [],
            "needs_more_evidence": False,
            "rounds": 1,
        }
    ]

    prompts = {
        "claim_extraction_prompt": build_claim_extraction_prompt(
            question=sample_question,
            initial_answer=sample_answer,
            source_system="s2",
            question_type="rag_multi_hop",
            expected_behavior="answer",
        ),
        "claim_verification_prompt": build_claim_verification_prompt(
            question=sample_question,
            claim=sample_claim,
            retrieved_chunks=sample_chunks,
            round_id=1,
            max_rounds=2,
        ),
        "search_query_prompt": build_search_query_prompt(
            question=sample_question,
            claim=sample_claim,
            previous_evidence=sample_chunks,
            previous_verdict="not_enough_info",
            previous_rationale="Need stronger evidence.",
        ),
        "final_repair_prompt": build_final_repair_prompt(
            question=sample_question,
            initial_answer=sample_answer,
            claim_results=sample_claim_results,
            expected_behavior="answer",
            source_system="s2",
        ),
    }

    print("S4 prompts self-test OK")
    print("=" * 72)
    for name, prompt in prompts.items():
        print(f"{name}: {len(prompt)} chars")

    print("=" * 72)
    print("Preview claim_extraction_prompt:")
    print(prompts["claim_extraction_prompt"][:1200])
    print("...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-test de prompts_s4.py."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Construye prompts de ejemplo y muestra sus longitudes.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    else:
        print("Archivo de prompts S4 importable. Usá --self-test para probarlo.")


if __name__ == "__main__":
    main()

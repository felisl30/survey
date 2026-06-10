#!/usr/bin/env python3
"""
claim_extractor_s4.py

Extractor de claims para S4: FIRE-like / FIRE-inspired claim verification.

Objetivo
--------
Tomar una pregunta y una respuesta inicial generada por S2/S3, y convertir esa
respuesta en claims atómicos verificables.

Este archivo:
- Puede extraer claims por reglas, LLM o modo híbrido.
- No hace retrieval.
- No verifica claims.
- No repara respuestas.
- Devuelve JSON normalizado listo para fire_verifier_s4.py.

Uso rápido
----------
Self-test sin API:
python s4_model_code/claim_extractor_s4.py --self-test

Ejemplo por reglas:
python s4_model_code/claim_extractor_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --answer "La traviata was composed by Giuseppe Verdi." \
  --strategy rules

Ejemplo con LLM:
python s4_model_code/claim_extractor_s4.py \
  --question "According to the available corpus, who composed La traviata?" \
  --answer "La traviata was composed by Giuseppe Verdi." \
  --strategy llm
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


# ---------------------------------------------------------------------------
# Paths e imports del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
S4_CODE_DIR = Path(__file__).resolve().parent

for path in [PROJECT_ROOT, S4_CODE_DIR]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from prompts_s4 import (
        DEFAULT_MAX_CLAIMS,
        S4_CLAIM_EXTRACTION_SYSTEM_PROMPT,
        VALID_CLAIM_TYPES,
        VALID_IMPORTANCE_LEVELS,
        build_claim_extraction_prompt,
    )
except ModuleNotFoundError:
    from s4_model_code.prompts_s4 import (
        DEFAULT_MAX_CLAIMS,
        S4_CLAIM_EXTRACTION_SYSTEM_PROMPT,
        VALID_CLAIM_TYPES,
        VALID_IMPORTANCE_LEVELS,
        build_claim_extraction_prompt,
    )


ExtractionStrategy = Literal["rules", "llm", "hybrid"]
VALID_EXTRACTION_STRATEGIES = {"rules", "llm", "hybrid"}


ABSTENTION_MARKERS = [
    "not enough information",
    "insufficient information",
    "not enough evidence",
    "insufficient evidence",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "i don't know",
    "i do not know",
    "no evidence",
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
    "sin evidencia suficiente",
    "evidencia insuficiente",
    "la evidencia no alcanza",
    "la evidencia recuperada no alcanza",
]

CLARIFICATION_MARKERS = [
    "need clarification",
    "needs clarification",
    "please clarify",
    "could you clarify",
    "ambiguous",
    "unclear",
    "not clear who",
    "not clear what",
    "not clear which",
    "necesito una aclaración",
    "necesito una aclaracion",
    "necesitaría una aclaración",
    "necesitaria una aclaracion",
    "podrías aclarar",
    "podrias aclarar",
    "pregunta ambigua",
    "no está claro",
    "no esta claro",
    "indicá a qué",
    "indica a que",
]

META_MARKERS = [
    "i think",
    "i believe",
    "in my opinion",
    "as an ai",
    "creo que",
    "en mi opinión",
    "en mi opinion",
]

ANSWER_CHOICE_PATTERNS = [
    r"^\s*([ABCD])\s*$",
    r"^\s*(?:the\s+)?answer\s+is\s+([ABCD])\.?\s*$",
    r"^\s*(?:la\s+)?respuesta\s+es\s+([ABCD])\.?\s*$",
    r"^\s*option\s+([ABCD])\.?\s*$",
    r"^\s*opci[oó]n\s+([ABCD])\.?\s*$",
]


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
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def normalize_for_rules(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
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


def safe_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, indent=indent)


def contains_any_normalized(text: Any, markers: list[str]) -> bool:
    lower = normalize_for_rules(text)
    return any(marker in lower for marker in markers)


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
# Normalización de claims
# ---------------------------------------------------------------------------

def normalize_claim_type(value: Any, *, fallback: str = "factual") -> str:
    text = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    if text in VALID_CLAIM_TYPES:
        return text
    return fallback


def normalize_importance(value: Any, *, fallback: str = "core") -> str:
    text = clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")
    if text in VALID_IMPORTANCE_LEVELS:
        return text
    return fallback


def normalize_claim_text(text: Any) -> str:
    claim = normalize_space(text)
    claim = re.sub(r"^[-*•]\s*", "", claim).strip()
    claim = claim.strip('"').strip("'").strip()

    if len(claim) > 600:
        claim = claim[:600].rstrip() + " [...]"

    return claim


def infer_requires_evidence(claim_type: str, claim_text: str) -> bool:
    if claim_type in {"abstention", "clarification", "meta"}:
        return False
    if claim_type == "answer_choice":
        # Para MMLU/direct normalmente no existe evidencia documental local.
        # El evaluador de respuestas chequea la opción contra gold.
        return False
    return bool(claim_text)


def make_claim(
    *,
    index: int,
    claim_text: str,
    claim_type: str = "factual",
    requires_evidence: bool | None = None,
    importance: str = "core",
) -> dict[str, Any]:
    claim_type = normalize_claim_type(claim_type)
    importance = normalize_importance(importance)
    claim_text = normalize_claim_text(claim_text)

    if requires_evidence is None:
        requires_evidence = infer_requires_evidence(claim_type, claim_text)

    return {
        "claim_id": f"c{index}",
        "claim_text": claim_text,
        "claim_type": claim_type,
        "requires_evidence": bool(requires_evidence),
        "importance": importance,
    }


def validate_extraction_payload(
    obj: dict[str, Any] | None,
    *,
    initial_answer: str,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> dict[str, Any]:
    """
    Normaliza cualquier salida del LLM/reglas al contrato de S4.
    """
    obj = obj or {}
    raw_claims = obj.get("claims", [])

    if not isinstance(raw_claims, list):
        raw_claims = []

    claims: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for raw in raw_claims:
        if not isinstance(raw, dict):
            continue

        claim_text = normalize_claim_text(raw.get("claim_text", ""))
        if not claim_text:
            continue

        dedupe_key = normalize_for_rules(claim_text)
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)

        claim_type = normalize_claim_type(raw.get("claim_type", "factual"))
        importance = normalize_importance(raw.get("importance", "core"))

        requires_evidence_raw = raw.get("requires_evidence", None)
        if requires_evidence_raw is None:
            requires_evidence = infer_requires_evidence(claim_type, claim_text)
        else:
            requires_evidence = coerce_bool(
                requires_evidence_raw,
                default=infer_requires_evidence(claim_type, claim_text),
            )

        claims.append(
            make_claim(
                index=len(claims) + 1,
                claim_text=claim_text,
                claim_type=claim_type,
                requires_evidence=requires_evidence,
                importance=importance,
            )
        )

        if len(claims) >= max_claims:
            break

    needs_verification = coerce_bool(obj.get("needs_verification"), default=True)

    if not claims:
        # Fallback seguro: si hay respuesta pero el extractor falló, la respuesta
        # completa se trata como un claim factual central.
        fallback_answer = normalize_claim_text(initial_answer)
        if fallback_answer:
            claim_type = "abstention" if contains_any_normalized(fallback_answer, ABSTENTION_MARKERS) else "factual"
            if contains_any_normalized(fallback_answer, CLARIFICATION_MARKERS):
                claim_type = "clarification"
            claims = [
                make_claim(
                    index=1,
                    claim_text=fallback_answer,
                    claim_type=claim_type,
                    importance="core",
                )
            ]
        needs_verification = bool(claims)

    # Si todos los claims no requieren evidencia, no hace falta verificación documental.
    if claims and all(not c["requires_evidence"] for c in claims):
        needs_verification = False

    return {
        "claims": claims,
        "needs_verification": bool(needs_verification),
        "extraction_notes": clean_text(obj.get("extraction_notes", "")),
    }


# ---------------------------------------------------------------------------
# Extractor por reglas
# ---------------------------------------------------------------------------

def detect_answer_choice(answer: str) -> str:
    text = clean_text(answer).strip()
    for pattern in ANSWER_CHOICE_PATTERNS:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def split_candidate_sentences(answer: str) -> list[str]:
    """
    Split simple por oraciones/bullets.
    No intenta ser NLP completo; solo un fallback reproducible.
    """
    text = clean_text(answer)
    if not text:
        return []

    # Separar bullets/listas primero.
    text = re.sub(r"\n+\s*[-*•]\s*", ". ", text)
    text = re.sub(r"\n+", " ", text)

    # Evitar dividir respuestas tipo A. opción si son demasiado cortas.
    if detect_answer_choice(text):
        return [text]

    pieces = re.split(r"(?<=[.!?])\s+", text)
    candidates: list[str] = []

    for piece in pieces:
        piece = normalize_claim_text(piece)
        if not piece:
            continue
        if len(piece) < 2:
            continue
        candidates.append(piece)

    if not candidates and text:
        candidates = [normalize_claim_text(text)]

    return candidates


def infer_sentence_claim_type(sentence: str) -> str:
    if contains_any_normalized(sentence, ABSTENTION_MARKERS):
        return "abstention"
    if contains_any_normalized(sentence, CLARIFICATION_MARKERS):
        return "clarification"
    if contains_any_normalized(sentence, META_MARKERS):
        return "meta"
    if detect_answer_choice(sentence):
        return "answer_choice"
    return "factual"


def extract_claims_rules(
    *,
    question: str,
    initial_answer: str,
    source_system: str = "",
    question_type: str = "",
    expected_behavior: str = "",
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> dict[str, Any]:
    answer = clean_text(initial_answer)

    if not answer:
        payload = {
            "claims": [],
            "needs_verification": False,
            "extraction_notes": "Respuesta inicial vacía; no se pudieron extraer claims.",
        }
        return validate_extraction_payload(payload, initial_answer=initial_answer, max_claims=max_claims)

    answer_choice = detect_answer_choice(answer)
    if answer_choice:
        payload = {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": f"The selected answer is {answer_choice}.",
                    "claim_type": "answer_choice",
                    "requires_evidence": False,
                    "importance": "core",
                }
            ],
            "needs_verification": False,
            "extraction_notes": "Extracción por reglas: respuesta multiple-choice.",
        }
        return validate_extraction_payload(payload, initial_answer=initial_answer, max_claims=max_claims)

    if contains_any_normalized(answer, CLARIFICATION_MARKERS):
        payload = {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": answer,
                    "claim_type": "clarification",
                    "requires_evidence": False,
                    "importance": "core",
                }
            ],
            "needs_verification": False,
            "extraction_notes": "Extracción por reglas: pedido de aclaración detectado.",
        }
        return validate_extraction_payload(payload, initial_answer=initial_answer, max_claims=max_claims)

    if contains_any_normalized(answer, ABSTENTION_MARKERS):
        payload = {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": answer,
                    "claim_type": "abstention",
                    "requires_evidence": False,
                    "importance": "core",
                }
            ],
            "needs_verification": False,
            "extraction_notes": "Extracción por reglas: abstención detectada.",
        }
        return validate_extraction_payload(payload, initial_answer=initial_answer, max_claims=max_claims)

    sentences = split_candidate_sentences(answer)

    claims: list[dict[str, Any]] = []
    for sent in sentences:
        claim_type = infer_sentence_claim_type(sent)

        # Evitar claims puramente decorativos.
        if claim_type == "meta" and len(sent.split()) <= 5:
            continue

        importance = "core" if not claims else "supporting"
        claims.append(
            {
                "claim_id": f"c{len(claims) + 1}",
                "claim_text": sent,
                "claim_type": claim_type,
                "requires_evidence": infer_requires_evidence(claim_type, sent),
                "importance": importance,
            }
        )

        if len(claims) >= max_claims:
            break

    payload = {
        "claims": claims,
        "needs_verification": any(c.get("requires_evidence", False) for c in claims),
        "extraction_notes": "Extracción por reglas: split por oraciones.",
    }

    return validate_extraction_payload(payload, initial_answer=initial_answer, max_claims=max_claims)


# ---------------------------------------------------------------------------
# Extractor por LLM
# ---------------------------------------------------------------------------

def import_llm_client():
    try:
        from direct_llm import ask_direct_llm_with_metadata
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se pudo importar direct_llm.py. Ejecutá desde la raíz del proyecto."
        ) from exc
    return ask_direct_llm_with_metadata


def call_llm_json(
    *,
    prompt: str,
    system_prompt: str,
    model: str | None,
    max_retries: int,
    dry_run: bool,
    dry_run_payload: dict[str, Any],
) -> dict[str, Any]:
    if dry_run:
        raw_output = json.dumps(dry_run_payload, ensure_ascii=False)
        return {
            "parsed": dry_run_payload,
            "raw_output": raw_output,
            "parse_error": "",
            "model": "dry_run",
            "usage_json": "{}",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "dry_run": True,
        }

    ask_direct_llm_with_metadata = import_llm_client()

    result = ask_direct_llm_with_metadata(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_retries=max_retries,
    )

    raw_output = clean_text(result.get("raw_output", ""))
    parsed, parse_error = load_json_object(raw_output)

    return {
        "parsed": parsed or {},
        "raw_output": raw_output,
        "parse_error": parse_error,
        "model": result.get("model", model or ""),
        "usage_json": result.get("usage_json", ""),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "total_tokens": result.get("total_tokens"),
        "latency_seconds": result.get("latency_seconds"),
        "dry_run": False,
    }


def extract_claims_llm(
    *,
    question: str,
    initial_answer: str,
    source_system: str = "",
    question_type: str = "",
    expected_behavior: str = "",
    max_claims: int = DEFAULT_MAX_CLAIMS,
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    prompt = build_claim_extraction_prompt(
        question=question,
        initial_answer=initial_answer,
        source_system=source_system,
        question_type=question_type,
        expected_behavior=expected_behavior,
        max_claims=max_claims,
    )

    dry_payload = extract_claims_rules(
        question=question,
        initial_answer=initial_answer,
        source_system=source_system,
        question_type=question_type,
        expected_behavior=expected_behavior,
        max_claims=max_claims,
    )

    call = call_llm_json(
        prompt=prompt,
        system_prompt=S4_CLAIM_EXTRACTION_SYSTEM_PROMPT,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
        dry_run_payload=dry_payload,
    )

    validated = validate_extraction_payload(
        call["parsed"],
        initial_answer=initial_answer,
        max_claims=max_claims,
    )

    return {
        **validated,
        "extractor_strategy": "llm",
        "extractor_model": call["model"],
        "extractor_raw_output": call["raw_output"],
        "extractor_parse_error": call["parse_error"],
        "extractor_usage_json": call["usage_json"],
        "extractor_input_tokens": call["input_tokens"],
        "extractor_output_tokens": call["output_tokens"],
        "extractor_total_tokens": call["total_tokens"],
        "extractor_latency_seconds": call["latency_seconds"],
        "extractor_dry_run": call["dry_run"],
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract_claims(
    *,
    question: str,
    initial_answer: str,
    source_system: str = "",
    question_type: str = "",
    expected_behavior: str = "",
    max_claims: int = DEFAULT_MAX_CLAIMS,
    strategy: ExtractionStrategy = "rules",
    model: str | None = None,
    max_retries: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    strategy = clean_text(strategy).lower()
    if strategy not in VALID_EXTRACTION_STRATEGIES:
        raise ValueError(
            f"strategy inválida: {strategy!r}. "
            f"Opciones válidas: {sorted(VALID_EXTRACTION_STRATEGIES)}"
        )

    start = time.time()

    if strategy == "rules":
        result = extract_claims_rules(
            question=question,
            initial_answer=initial_answer,
            source_system=source_system,
            question_type=question_type,
            expected_behavior=expected_behavior,
            max_claims=max_claims,
        )
        result.update(
            {
                "extractor_strategy": "rules",
                "extractor_model": "",
                "extractor_raw_output": "",
                "extractor_parse_error": "",
                "extractor_usage_json": "{}",
                "extractor_input_tokens": 0,
                "extractor_output_tokens": 0,
                "extractor_total_tokens": 0,
                "extractor_latency_seconds": round(time.time() - start, 3),
                "extractor_dry_run": False,
            }
        )
        return result

    if strategy == "llm":
        return extract_claims_llm(
            question=question,
            initial_answer=initial_answer,
            source_system=source_system,
            question_type=question_type,
            expected_behavior=expected_behavior,
            max_claims=max_claims,
            model=model,
            max_retries=max_retries,
            dry_run=dry_run,
        )

    # hybrid:
    # - casos obvios de abstención, aclaración o multiple-choice se resuelven por reglas;
    # - respuestas factuales abiertas se mandan al LLM para mejor atomicidad.
    rule_result = extract_claims_rules(
        question=question,
        initial_answer=initial_answer,
        source_system=source_system,
        question_type=question_type,
        expected_behavior=expected_behavior,
        max_claims=max_claims,
    )

    claims = rule_result.get("claims", [])
    if claims and all(c.get("claim_type") in {"abstention", "clarification", "answer_choice"} for c in claims):
        rule_result.update(
            {
                "extractor_strategy": "hybrid_rules",
                "extractor_model": "",
                "extractor_raw_output": "",
                "extractor_parse_error": "",
                "extractor_usage_json": "{}",
                "extractor_input_tokens": 0,
                "extractor_output_tokens": 0,
                "extractor_total_tokens": 0,
                "extractor_latency_seconds": round(time.time() - start, 3),
                "extractor_dry_run": False,
            }
        )
        return rule_result

    llm_result = extract_claims_llm(
        question=question,
        initial_answer=initial_answer,
        source_system=source_system,
        question_type=question_type,
        expected_behavior=expected_behavior,
        max_claims=max_claims,
        model=model,
        max_retries=max_retries,
        dry_run=dry_run,
    )
    llm_result["extractor_strategy"] = "hybrid_llm"
    return llm_result


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    examples = [
        {
            "name": "factual",
            "question": "According to the available corpus, who composed La traviata?",
            "answer": "La traviata was composed by Giuseppe Verdi.",
            "question_type": "rag_multi_hop",
            "expected_behavior": "answer",
        },
        {
            "name": "abstention",
            "question": "According to the available corpus, what is the unpublished access code associated with La traviata?",
            "answer": "No hay información suficiente para responder con la evidencia disponible.",
            "question_type": "rag_no_answer",
            "expected_behavior": "abstain",
        },
        {
            "name": "clarification",
            "question": "When was he born?",
            "answer": "Necesito una aclaración mínima para saber a qué persona te referís.",
            "question_type": "ambiguous",
            "expected_behavior": "clarify",
        },
        {
            "name": "multiple_choice",
            "question": "Which option is correct?",
            "answer": "D",
            "question_type": "direct_mmlu",
            "expected_behavior": "answer",
        },
    ]

    print("S4 claim extractor self-test")
    print("=" * 72)

    for ex in examples:
        result = extract_claims(
            question=ex["question"],
            initial_answer=ex["answer"],
            source_system="self_test",
            question_type=ex["question_type"],
            expected_behavior=ex["expected_behavior"],
            strategy="rules",
            max_claims=5,
        )

        print(f"Example: {ex['name']}")
        print(safe_json_dumps(result, indent=2))
        print("-" * 72)

    print("S4 claim extractor self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extractor de claims para S4 FIRE-like."
    )
    parser.add_argument("--question", type=str, default="", help="Pregunta original.")
    parser.add_argument("--answer", type=str, default="", help="Respuesta inicial a descomponer en claims.")
    parser.add_argument("--source-system", type=str, default="manual", help="Sistema fuente: s2, s3, etc.")
    parser.add_argument("--question-type", type=str, default="", help="Tipo de pregunta/caso.")
    parser.add_argument("--expected-behavior", type=str, default="", help="answer/abstain/clarify si está disponible.")
    parser.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS, help="Cantidad máxima de claims.")
    parser.add_argument(
        "--strategy",
        choices=sorted(VALID_EXTRACTION_STRATEGIES),
        default="rules",
        help="Estrategia de extracción.",
    )
    parser.add_argument("--model", type=str, default=None, help="Modelo LLM para strategy=llm/hybrid.")
    parser.add_argument("--max-retries", type=int, default=2, help="Reintentos para llamada LLM.")
    parser.add_argument("--dry-run", action="store_true", help="No llama API; simula salida LLM con reglas.")
    parser.add_argument("--self-test", action="store_true", help="Ejecuta ejemplos locales por reglas.")

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not clean_text(args.question) or not clean_text(args.answer):
        raise ValueError(
            "Debés pasar --question y --answer, o usar --self-test."
        )

    result = extract_claims(
        question=args.question,
        initial_answer=args.answer,
        source_system=args.source_system,
        question_type=args.question_type,
        expected_behavior=args.expected_behavior,
        max_claims=args.max_claims,
        strategy=args.strategy,
        model=args.model,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    print(safe_json_dumps(result, indent=2))


if __name__ == "__main__":
    main()

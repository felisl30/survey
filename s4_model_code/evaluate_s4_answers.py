#!/usr/bin/env python3
"""
evaluate_s4_answers.py

Evaluador answer-level para S4: FIRE-like / FIRE-inspired claim verification.

Entrada típica:
    outputs/s4/generation/fire_like_s4_parsed.csv

Salidas típicas:
    outputs/s4/evaluation/fire_like_s4_answer_results.csv
    outputs/s4/evaluation/fire_like_s4_answer_summary.json
    outputs/s4/evaluation/fire_like_s4_answer_summary_by_group.csv

Objetivo
--------
Evaluar la respuesta final de S4.

Este archivo:
- No llama al LLM.
- No vuelve a recuperar evidencia.
- No evalúa claim-level en detalle, eso ya lo hace evaluate_s4_claims.py.
- Evalúa si la respuesta final:
    - responde cuando debía responder;
    - se abstiene cuando debía abstenerse;
    - pide aclaración cuando debía pedir aclaración;
    - respeta formato válido;
    - coincide con gold_answer / expected_answer cuando existe;
    - cambia o no cambia respecto de la respuesta fuente;
    - queda alineada con los veredictos de claims.

Uso rápido
----------

python s4_model_code/evaluate_s4_answers.py \
  --input-path outputs/s4/generation/fire_like_s4_parsed_retrieve_test_5_hybrid_claims_llm_verify.csv \
  --output-path outputs/s4/evaluation/fire_like_s4_answer_results_retrieve_test_5_hybrid_llm.csv \
  --summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_retrieve_test_5_hybrid_llm.json \
  --group-summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_by_group_retrieve_test_5_hybrid_llm.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s4/generation/fire_like_s4_parsed.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s4/evaluation/fire_like_s4_answer_results.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/s4/evaluation/fire_like_s4_answer_summary.json")
DEFAULT_GROUP_SUMMARY_PATH = Path("outputs/s4/evaluation/fire_like_s4_answer_summary_by_group.csv")

VALID_MMLU_OPTIONS = {"A", "B", "C", "D"}

ABSTENTION_MARKERS = [
    "no se",
    "no sé",
    "no estoy seguro",
    "no tengo informacion suficiente",
    "no tengo información suficiente",
    "no hay informacion suficiente",
    "no hay información suficiente",
    "informacion insuficiente",
    "información insuficiente",
    "la evidencia recuperada no alcanza",
    "no puedo mantenerla",
    "no puedo mantenerla como respuesta confiable",
    "insufficient information",
    "not enough information",
    "not enough evidence",
    "insufficient evidence",
    "cannot determine",
    "can't determine",
    "i don't know",
    "i do not know",
]

CLARIFICATION_MARKERS = [
    "necesito una aclaracion",
    "necesito una aclaración",
    "podrias aclarar",
    "podrías aclarar",
    "por favor aclar",
    "falta aclarar",
    "pregunta ambigua",
    "no esta claro",
    "no está claro",
    "need clarification",
    "needs clarification",
    "please clarify",
    "could you clarify",
    "ambiguous",
    "unclear",
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "it", "its", "as", "at", "from", "but", "not", "does", "do", "did", "can",
    "could", "would", "should", "will", "may", "might", "have", "has", "had",
    "if", "then", "than", "into", "about", "which", "who", "what", "where", "when",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de",
    "del", "en", "con", "por", "para", "es", "son", "fue", "eran", "ser",
    "que", "este", "esta", "esto", "como", "hay", "si", "no", "se", "su", "sus",
}

EVAL_COLUMNS = [
    "eval_expected_final_behavior",
    "eval_behavior_correct",
    "eval_answer_correct",
    "eval_final_correct",
    "eval_exact_match",
    "eval_contains_gold_answer",
    "eval_gold_contains_answer",
    "eval_token_precision",
    "eval_token_recall",
    "eval_token_f1",
    "eval_abstained",
    "eval_clarified",
    "eval_wrong_abstention",
    "eval_wrong_clarification",
    "eval_missing_abstention",
    "eval_missing_clarification",
    "eval_source_answer",
    "eval_source_abstained",
    "eval_s4_changed_answer",
    "eval_s4_changed_behavior",
    "eval_claim_support_rate",
    "eval_claim_nei_rate",
    "eval_claim_refuted_rate",
    "eval_all_claims_supported",
    "eval_any_claim_nei",
    "eval_any_claim_refuted",
    "eval_error_type",
    "eval_needs_manual_review",
    "eval_notes",
]


# ---------------------------------------------------------------------------
# Utilidades
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


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", clean_text(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: Any) -> str:
    text = strip_accents(clean_text(value).lower())
    text = text.replace("’", "'")
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def coerce_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return bool(value)
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if 1.0 < number <= 100.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def safe_json_loads(value: Any) -> Any:
    text = clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_list(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]

    text = clean_text(value)
    parsed = safe_json_loads(text)
    if isinstance(parsed, list):
        return [clean_text(x) for x in parsed if clean_text(x)]

    if "|" in text:
        return [clean_text(x) for x in text.split("|") if clean_text(x)]

    return [text] if text else []


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


def tokenize(value: Any) -> list[str]:
    text = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


# ---------------------------------------------------------------------------
# Detección de respuesta/comportamiento
# ---------------------------------------------------------------------------

def get_s4_answer(row: pd.Series) -> str:
    for col in ["parsed_answer", "s4_answer", "answer"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def get_source_answer(row: pd.Series) -> str:
    for col in ["source_initial_answer", "source_parsed_answer", "initial_answer"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def infer_abstention(answer: str) -> bool:
    lower = normalize_text(answer)
    return any(normalize_text(marker) in lower for marker in ABSTENTION_MARKERS)


def infer_clarification(answer: str) -> bool:
    lower = normalize_text(answer)
    return any(normalize_text(marker) in lower for marker in CLARIFICATION_MARKERS)


def expected_final_behavior(row: pd.Series) -> str:
    for col in ["expected_final_behavior", "s4_expected_behavior", "expected_behavior"]:
        label = normalize_label(row.get(col, ""))
        if label:
            if label in {"abstain", "abstention", "no_answer", "not_enough_information", "insufficient_information"}:
                return "abstain"
            if label in {"clarify", "clarification", "ask_clarification", "needs_clarification"}:
                return "clarify"
            return "answer"

    expected_route = normalize_label(row.get("expected_route", ""))
    if expected_route == "abstain":
        return "abstain"
    if expected_route == "clarify":
        return "clarify"

    return "answer"


def is_mmlu_row(row: pd.Series) -> bool:
    values = [
        normalize_label(row.get("dataset", "")),
        normalize_label(row.get("source_dataset", "")),
        normalize_label(row.get("case_type", "")),
        normalize_label(row.get("s2_case_type", "")),
        normalize_label(row.get("task_type", "")),
    ]
    return any(v in {"mmlu", "direct_mmlu", "multiple_choice", "multiplechoice"} for v in values)


def normalize_mmlu_answer(value: Any) -> str:
    text = clean_text(value).upper()
    if text in VALID_MMLU_OPTIONS:
        return text

    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()

    return text


def get_gold_answers(row: pd.Series) -> list[str]:
    answers: list[str] = []

    for col in ["gold_answer", "expected_answer", "gold_answer_text", "best_answer", "correct_answer"]:
        value = clean_text(row.get(col, ""))
        if value and value not in answers:
            answers.append(value)

    for col in ["correct_answers_json", "gold_answers_json", "acceptable_answers_json"]:
        for value in parse_list(row.get(col, "")):
            if value and value not in answers:
                answers.append(value)

    return answers


def token_scores(prediction: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())

    if overlap <= 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def best_gold_match(prediction: str, gold_answers: list[str]) -> dict[str, Any]:
    pred_norm = normalize_text(prediction)

    best = {
        "gold": "",
        "exact": False,
        "contains_gold": False,
        "gold_contains_answer": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "score": 0.0,
    }

    if not pred_norm or not gold_answers:
        return best

    for gold in gold_answers:
        gold_norm = normalize_text(gold)
        if not gold_norm:
            continue

        exact = pred_norm == gold_norm
        contains_gold = gold_norm in pred_norm
        gold_contains_answer = pred_norm in gold_norm

        precision, recall, f1 = token_scores(prediction, gold)
        score = f1

        if contains_gold:
            score = max(score, 0.95)
        if gold_contains_answer:
            score = max(score, 0.90)
        if exact:
            score = 1.0

        if score > best["score"]:
            best = {
                "gold": gold,
                "exact": bool(exact),
                "contains_gold": bool(contains_gold),
                "gold_contains_answer": bool(gold_contains_answer),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "score": float(score),
            }

    return best


# ---------------------------------------------------------------------------
# Evaluación answer-level
# ---------------------------------------------------------------------------

def claim_alignment_metrics(row: pd.Series) -> dict[str, Any]:
    n_claims = safe_int(row.get("s4_num_claims", 0), default=0)
    n_supported = safe_int(row.get("s4_num_supported_claims", 0), default=0)
    n_refuted = safe_int(row.get("s4_num_refuted_claims", 0), default=0)
    n_nei = safe_int(row.get("s4_num_nei_claims", 0), default=0)

    if n_claims > 0:
        support_rate = n_supported / n_claims
        refuted_rate = n_refuted / n_claims
        nei_rate = n_nei / n_claims
    else:
        support_rate = 0.0
        refuted_rate = 0.0
        nei_rate = 0.0

    return {
        "eval_claim_support_rate": round(float(support_rate), 4),
        "eval_claim_nei_rate": round(float(nei_rate), 4),
        "eval_claim_refuted_rate": round(float(refuted_rate), 4),
        "eval_all_claims_supported": bool(n_claims > 0 and n_supported == n_claims),
        "eval_any_claim_nei": bool(n_nei > 0),
        "eval_any_claim_refuted": bool(n_refuted > 0),
    }


def evaluate_answer_content(row: pd.Series, answer: str, f1_threshold: float) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []

    if is_mmlu_row(row):
        pred = normalize_mmlu_answer(answer)
        gold_candidates = get_gold_answers(row)
        gold = normalize_mmlu_answer(gold_candidates[0]) if gold_candidates else ""

        correct = bool(pred in VALID_MMLU_OPTIONS and gold in VALID_MMLU_OPTIONS and pred == gold)

        if not gold or gold not in VALID_MMLU_OPTIONS:
            notes.append("Gold MMLU ausente o inválido.")
        if pred not in VALID_MMLU_OPTIONS:
            notes.append("Respuesta MMLU parseada inválida.")

        return {
            "eval_exact_match": correct,
            "eval_contains_gold_answer": correct,
            "eval_gold_contains_answer": correct,
            "eval_token_precision": 1.0 if correct else 0.0,
            "eval_token_recall": 1.0 if correct else 0.0,
            "eval_token_f1": 1.0 if correct else 0.0,
            "eval_answer_correct": correct,
        }, notes

    gold_answers = get_gold_answers(row)
    match = best_gold_match(answer, gold_answers)

    exact = bool(match["exact"])
    contains_gold = bool(match["contains_gold"])
    gold_contains_answer = bool(match["gold_contains_answer"])
    precision = float(match["precision"])
    recall = float(match["recall"])
    f1 = float(match["f1"])

    if not gold_answers:
        # Sin gold no marcamos automáticamente mal si el comportamiento fue correcto.
        # Dejamos answer_correct como None-equivalente práctico: False + nota.
        notes.append("No hay gold_answer/expected_answer para comparar contenido.")
        answer_correct = False
    else:
        notes.append(f"best_gold_match={match['gold'][:120]!r}")
        answer_correct = bool(exact or contains_gold or gold_contains_answer or f1 >= f1_threshold)

    return {
        "eval_exact_match": exact,
        "eval_contains_gold_answer": contains_gold,
        "eval_gold_contains_answer": gold_contains_answer,
        "eval_token_precision": round(precision, 4),
        "eval_token_recall": round(recall, 4),
        "eval_token_f1": round(f1, 4),
        "eval_answer_correct": answer_correct,
    }, notes


def evaluate_row(row: pd.Series, *, f1_threshold: float) -> dict[str, Any]:
    answer = get_s4_answer(row)
    source = get_source_answer(row)
    behavior = expected_final_behavior(row)

    parsed_abstained = coerce_bool(row.get("parsed_abstained"), default=False)
    s4_abstained = coerce_bool(row.get("s4_abstained"), default=False)
    abstained = bool(parsed_abstained or s4_abstained or infer_abstention(answer))
    clarified = bool(infer_clarification(answer))

    source_parsed_abstained = coerce_bool(row.get("source_parsed_abstained"), default=False)
    source_abstained = bool(source_parsed_abstained or infer_abstention(source))

    changed_answer = bool(normalize_text(answer) != normalize_text(source)) if answer and source else False
    changed_behavior = bool(abstained != source_abstained)

    valid_format = coerce_bool(row.get("valid_answer_format", row.get("valid_format")), default=False)
    run_error = coerce_bool(row.get("run_error_present"), default=False)
    parse_error = clean_text(row.get("parse_error", ""))
    error_text = clean_text(row.get("error", ""))

    claim_metrics = claim_alignment_metrics(row)
    final_decision = normalize_label(row.get("s4_final_decision", ""))

    notes: list[str] = []

    base_answer_metrics = {
        "eval_exact_match": False,
        "eval_contains_gold_answer": False,
        "eval_gold_contains_answer": False,
        "eval_token_precision": 0.0,
        "eval_token_recall": 0.0,
        "eval_token_f1": 0.0,
        "eval_answer_correct": False,
    }

    if behavior == "answer":
        content_metrics, content_notes = evaluate_answer_content(row, answer, f1_threshold)
        base_answer_metrics.update(content_metrics)
        notes.extend(content_notes)

        behavior_correct = bool(not abstained and not clarified)
        # Si no hay gold, pero todos los claims están soportados y S4 no se abstuvo,
        # marcamos final_correct por comportamiento+claims, pero dejamos nota.
        has_gold = bool(get_gold_answers(row))
        if has_gold:
            final_correct = bool(behavior_correct and base_answer_metrics["eval_answer_correct"])
        else:
            final_correct = bool(behavior_correct and claim_metrics["eval_all_claims_supported"])
            if final_correct:
                notes.append("Sin gold textual; final_correct inferido por comportamiento correcto + todos los claims soportados.")

    elif behavior == "abstain":
        behavior_correct = bool(abstained)
        base_answer_metrics["eval_answer_correct"] = bool(abstained)
        final_correct = bool(abstained)
        if not abstained:
            notes.append("Se esperaba abstención, pero S4 no se abstuvo.")

    elif behavior == "clarify":
        behavior_correct = bool(clarified)
        base_answer_metrics["eval_answer_correct"] = bool(clarified)
        final_correct = bool(clarified)
        if not clarified:
            notes.append("Se esperaba aclaración, pero S4 no pidió aclaración explícita.")

    else:
        behavior_correct = False
        final_correct = False
        notes.append(f"Comportamiento esperado desconocido: {behavior}")

    wrong_abstention = bool(behavior == "answer" and abstained)
    wrong_clarification = bool(behavior == "answer" and clarified)
    missing_abstention = bool(behavior == "abstain" and not abstained)
    missing_clarification = bool(behavior == "clarify" and not clarified)

    needs_manual_review = False

    if run_error or error_text:
        error_type = "run_error"
        needs_manual_review = True
        notes.append("S4 tuvo error de ejecución.")
    elif parse_error:
        error_type = "parse_warning"
        needs_manual_review = True
        notes.append(f"Parse warning: {parse_error}")
    elif not valid_format:
        error_type = "format_error"
        needs_manual_review = True
        notes.append("Formato S4 inválido.")
    elif final_correct:
        error_type = "correct"
    elif wrong_abstention:
        error_type = "wrong_abstention"
        needs_manual_review = True
    elif wrong_clarification:
        error_type = "wrong_clarification"
        needs_manual_review = True
    elif missing_abstention:
        error_type = "missing_abstention"
        needs_manual_review = True
    elif missing_clarification:
        error_type = "missing_clarification"
        needs_manual_review = True
    elif claim_metrics["eval_any_claim_refuted"]:
        error_type = "claim_refuted"
        needs_manual_review = True
    elif claim_metrics["eval_any_claim_nei"]:
        error_type = "claim_nei"
        needs_manual_review = True
    elif behavior == "answer":
        error_type = "answer_mismatch"
        needs_manual_review = True
    else:
        error_type = "behavior_error"
        needs_manual_review = True

    if final_decision == "abstained" and behavior == "answer":
        notes.append("S4 se abstuvo en un caso donde se esperaba respuesta.")
    if changed_answer:
        notes.append("La respuesta final S4 cambió respecto de la respuesta fuente.")
    if claim_metrics["eval_any_claim_nei"]:
        notes.append("Hay al menos un claim not_enough_info.")
    if claim_metrics["eval_any_claim_refuted"]:
        notes.append("Hay al menos un claim refutado.")

    return {
        "eval_expected_final_behavior": behavior,
        "eval_behavior_correct": bool(behavior_correct),
        **base_answer_metrics,
        "eval_final_correct": bool(final_correct),
        "eval_abstained": bool(abstained),
        "eval_clarified": bool(clarified),
        "eval_wrong_abstention": wrong_abstention,
        "eval_wrong_clarification": wrong_clarification,
        "eval_missing_abstention": missing_abstention,
        "eval_missing_clarification": missing_clarification,
        "eval_source_answer": source,
        "eval_source_abstained": bool(source_abstained),
        "eval_s4_changed_answer": bool(changed_answer),
        "eval_s4_changed_behavior": bool(changed_behavior),
        **claim_metrics,
        "eval_error_type": error_type,
        "eval_needs_manual_review": bool(needs_manual_review),
        "eval_notes": " ".join(notes),
    }


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------

def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}

    summary = {
        "n": int(len(df)),
        "final_accuracy": mean_bool(df["eval_final_correct"]) if "eval_final_correct" in df else None,
        "behavior_accuracy": mean_bool(df["eval_behavior_correct"]) if "eval_behavior_correct" in df else None,
        "answer_accuracy": mean_bool(df["eval_answer_correct"]) if "eval_answer_correct" in df else None,
        "exact_match_rate": mean_bool(df["eval_exact_match"]) if "eval_exact_match" in df else None,
        "contains_gold_answer_rate": mean_bool(df["eval_contains_gold_answer"]) if "eval_contains_gold_answer" in df else None,
        "avg_token_f1": mean_numeric(df["eval_token_f1"]) if "eval_token_f1" in df else None,

        "valid_answer_format_rate": mean_bool(df["valid_answer_format"]) if "valid_answer_format" in df else None,
        "run_error_rate": mean_bool(df["run_error_present"]) if "run_error_present" in df else None,

        "s4_abstention_rate": mean_bool(df["eval_abstained"]) if "eval_abstained" in df else None,
        "s4_clarification_rate": mean_bool(df["eval_clarified"]) if "eval_clarified" in df else None,
        "wrong_abstention_rate": mean_bool(df["eval_wrong_abstention"]) if "eval_wrong_abstention" in df else None,
        "wrong_clarification_rate": mean_bool(df["eval_wrong_clarification"]) if "eval_wrong_clarification" in df else None,
        "missing_abstention_rate": mean_bool(df["eval_missing_abstention"]) if "eval_missing_abstention" in df else None,
        "missing_clarification_rate": mean_bool(df["eval_missing_clarification"]) if "eval_missing_clarification" in df else None,

        "source_abstention_rate": mean_bool(df["eval_source_abstained"]) if "eval_source_abstained" in df else None,
        "s4_changed_answer_rate": mean_bool(df["eval_s4_changed_answer"]) if "eval_s4_changed_answer" in df else None,
        "s4_changed_behavior_rate": mean_bool(df["eval_s4_changed_behavior"]) if "eval_s4_changed_behavior" in df else None,
        "manual_review_rate": mean_bool(df["eval_needs_manual_review"]) if "eval_needs_manual_review" in df else None,

        "avg_claim_support_rate": mean_numeric(df["eval_claim_support_rate"]) if "eval_claim_support_rate" in df else None,
        "avg_claim_nei_rate": mean_numeric(df["eval_claim_nei_rate"]) if "eval_claim_nei_rate" in df else None,
        "avg_claim_refuted_rate": mean_numeric(df["eval_claim_refuted_rate"]) if "eval_claim_refuted_rate" in df else None,
        "all_claims_supported_rate": mean_bool(df["eval_all_claims_supported"]) if "eval_all_claims_supported" in df else None,
        "any_claim_nei_rate": mean_bool(df["eval_any_claim_nei"]) if "eval_any_claim_nei" in df else None,
        "any_claim_refuted_rate": mean_bool(df["eval_any_claim_refuted"]) if "eval_any_claim_refuted" in df else None,

        "s4_correction_rate": mean_bool(df["s4_correction_applied"]) if "s4_correction_applied" in df else None,
        "avg_s4_claims": mean_numeric(df["s4_num_claims"]) if "s4_num_claims" in df else None,
        "avg_s4_supported_claims": mean_numeric(df["s4_num_supported_claims"]) if "s4_num_supported_claims" in df else None,
        "avg_s4_refuted_claims": mean_numeric(df["s4_num_refuted_claims"]) if "s4_num_refuted_claims" in df else None,
        "avg_s4_nei_claims": mean_numeric(df["s4_num_nei_claims"]) if "s4_num_nei_claims" in df else None,
        "avg_s4_verification_rounds": mean_numeric(df["s4_num_verification_rounds"]) if "s4_num_verification_rounds" in df else None,
        "avg_s4_retrieval_rounds": mean_numeric(df["s4_num_retrieval_rounds"]) if "s4_num_retrieval_rounds" in df else None,
        "avg_s4_chunks_retrieved": mean_numeric(df["s4_num_chunks_retrieved_total"]) if "s4_num_chunks_retrieved_total" in df else None,
        "avg_s4_total_tokens": mean_numeric(df["s4_total_tokens"]) if "s4_total_tokens" in df else None,
        "avg_s4_latency_seconds": mean_numeric(df["s4_latency_seconds"]) if "s4_latency_seconds" in df else None,
    }

    for col, key in [
        ("eval_expected_final_behavior", "expected_final_behavior_counts"),
        ("s4_final_decision", "s4_final_decision_counts"),
        ("eval_error_type", "error_type_counts"),
        ("s4_claim_strategy", "s4_claim_strategy_counts"),
        ("s4_verification_strategy", "s4_verification_strategy_counts"),
    ]:
        if col in df.columns:
            summary[key] = {
                clean_text(k): int(v)
                for k, v in df[col].value_counts(dropna=False).items()
            }

    return summary


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"group_type": "overall", "group": "all", **summarize_subset(df)})

    group_cols = [
        "s2_case_type",
        "dataset",
        "case_type",
        "expected_route",
        "eval_expected_final_behavior",
        "s4_final_decision",
        "s4_claim_strategy",
        "s4_verification_strategy",
        "s4_query_strategy",
        "s4_repair_strategy",
        "source_system_for_s4",
        "generation_policy",
    ]

    for col in group_cols:
        if col in df.columns:
            for value, subset in df.groupby(col, dropna=False):
                rows.append({
                    "group_type": col,
                    "group": clean_text(value),
                    **summarize_subset(subset),
                })

    if "s2_case_type" in df.columns and "s4_final_decision" in df.columns:
        for (case_type, decision), subset in df.groupby(["s2_case_type", "s4_final_decision"], dropna=False):
            rows.append({
                "group_type": "s2_case_type::s4_final_decision",
                "group": f"{clean_text(case_type)}::{clean_text(decision)}",
                **summarize_subset(subset),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("El CSV S4 debe tener columna 'id'.")

    if "parsed_answer" not in df.columns and "s4_answer" not in df.columns:
        raise ValueError(
            "El CSV S4 debe tener parsed_answer o s4_answer. "
            "Ejecutá primero parse_s4_outputs.py."
        )

    if "valid_answer_format" not in df.columns and "valid_format" not in df.columns:
        raise ValueError(
            "El CSV S4 debe tener valid_answer_format o valid_format. "
            "Ejecutá primero parse_s4_outputs.py."
        )


def evaluate_file(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    group_summary_path: Path,
    f1_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    eval_rows = [evaluate_row(row, f1_threshold=f1_threshold) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(eval_rows)

    base_cols = [col for col in df.columns if col not in EVAL_COLUMNS]
    out_df = pd.concat([df[base_cols].reset_index(drop=True), eval_df], axis=1)

    group_summary_df = build_group_summary(out_df)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "thresholds": {"f1_threshold": f1_threshold},
        "overall": summarize_subset(out_df),
    }

    for group_col in [
        "s2_case_type",
        "dataset",
        "case_type",
        "expected_route",
        "eval_expected_final_behavior",
        "s4_final_decision",
        "s4_claim_strategy",
        "s4_verification_strategy",
        "source_system_for_s4",
    ]:
        if group_col in out_df.columns:
            summary[f"by_{group_col}"] = {
                clean_text(value): summarize_subset(subset)
                for value, subset in out_df.groupby(group_col, dropna=False)
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_summary_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_df, summary, group_summary_df


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResumen evaluación answer-level S4")
    print("----------------------------------")
    overall = summary.get("overall", {})

    print(f"Filas evaluadas: {overall.get('n', 0)}")
    print(f"Final accuracy: {fmt(overall.get('final_accuracy'))}")
    print(f"Behavior accuracy: {fmt(overall.get('behavior_accuracy'))}")
    print(f"Answer accuracy: {fmt(overall.get('answer_accuracy'))}")
    print(f"Avg token F1: {fmt(overall.get('avg_token_f1'))}")
    print(f"Valid answer format rate: {fmt(overall.get('valid_answer_format_rate'))}")
    print(f"Run error rate: {fmt(overall.get('run_error_rate'))}")

    print("\nComportamiento S4:")
    print(f"S4 abstention rate: {fmt(overall.get('s4_abstention_rate'))}")
    print(f"S4 clarification rate: {fmt(overall.get('s4_clarification_rate'))}")
    print(f"Wrong abstention rate: {fmt(overall.get('wrong_abstention_rate'))}")
    print(f"Wrong clarification rate: {fmt(overall.get('wrong_clarification_rate'))}")
    print(f"Missing abstention rate: {fmt(overall.get('missing_abstention_rate'))}")
    print(f"Missing clarification rate: {fmt(overall.get('missing_clarification_rate'))}")

    print("\nComparación contra fuente:")
    print(f"Source abstention rate: {fmt(overall.get('source_abstention_rate'))}")
    print(f"S4 changed answer rate: {fmt(overall.get('s4_changed_answer_rate'))}")
    print(f"S4 changed behavior rate: {fmt(overall.get('s4_changed_behavior_rate'))}")
    print(f"S4 correction rate: {fmt(overall.get('s4_correction_rate'))}")
    print(f"Manual review rate: {fmt(overall.get('manual_review_rate'))}")

    print("\nClaims agregados:")
    print(f"Avg claim support rate: {fmt(overall.get('avg_claim_support_rate'))}")
    print(f"Avg claim NEI rate: {fmt(overall.get('avg_claim_nei_rate'))}")
    print(f"Avg claim refuted rate: {fmt(overall.get('avg_claim_refuted_rate'))}")
    print(f"All claims supported rate: {fmt(overall.get('all_claims_supported_rate'))}")
    print(f"Any claim NEI rate: {fmt(overall.get('any_claim_nei_rate'))}")
    print(f"Any claim refuted rate: {fmt(overall.get('any_claim_refuted_rate'))}")

    print("\nCosto S4:")
    print(f"Avg S4 claims: {fmt(overall.get('avg_s4_claims'))}")
    print(f"Avg S4 verification rounds: {fmt(overall.get('avg_s4_verification_rounds'))}")
    print(f"Avg S4 retrieval rounds: {fmt(overall.get('avg_s4_retrieval_rounds'))}")
    print(f"Avg S4 chunks retrieved: {fmt(overall.get('avg_s4_chunks_retrieved'))}")
    print(f"Avg S4 total tokens: {fmt(overall.get('avg_s4_total_tokens'))}")
    print(f"Avg S4 latency seconds: {fmt(overall.get('avg_s4_latency_seconds'))}")

    for title, key in [
        ("Comportamientos esperados", "expected_final_behavior_counts"),
        ("Decisiones finales S4", "s4_final_decision_counts"),
        ("Tipos de resultado/error", "error_type_counts"),
    ]:
        counts = overall.get(key, {})
        if counts:
            print(f"\n{title}:")
            for k, v in counts.items():
                print(f"- {k}: {v}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa respuestas finales de S4 FIRE-like.")

    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--group-summary-path", type=Path, default=DEFAULT_GROUP_SUMMARY_PATH)
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.70,
        help="Umbral token-F1 para respuestas abiertas cuando hay gold_answer.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _, summary, _ = evaluate_file(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        group_summary_path=args.group_summary_path,
        f1_threshold=args.f1_threshold,
    )

    print(f"Resultados S4 answer-level guardados en: {args.output_path}")
    print(f"Resumen JSON guardado en: {args.summary_path}")
    print(f"Resumen por grupos guardado en: {args.group_summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluate_s3_answers.py

Evaluador para S3: FLARE-like / active retrieval.

Entrada por defecto:
    outputs/s3/generation/flare_like_s3_parsed.csv

Salidas por defecto:
    outputs/s3/evaluation/flare_like_s3_answer_results.csv
    outputs/s3/evaluation/flare_like_s3_answer_summary.json
    outputs/s3/evaluation/flare_like_s3_answer_summary_by_group.csv

Diseño:
- No llama al LLM.
- Evalúa respuestas parseadas de S3.
- Evalúa formato, comportamiento final, multiple-choice, respuestas abiertas,
  uso de retrieval activo, soporte por evidencia y costos.
- A diferencia de S2, S3 no tiene router previo. Se infiere la acción usada:
  clarify / abstain / retrieve / direct.
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


DEFAULT_INPUT_PATH = Path("outputs/s3/generation/flare_like_s3_parsed.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s3/evaluation/flare_like_s3_answer_results.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/s3/evaluation/flare_like_s3_answer_summary.json")
DEFAULT_GROUP_SUMMARY_PATH = Path("outputs/s3/evaluation/flare_like_s3_answer_summary_by_group.csv")

VALID_MMLU_OPTIONS = {"A", "B", "C", "D"}
VALID_ROUTES = {"direct", "retrieve", "abstain", "clarify"}
SUPPORTED_STATUSES = {"supported", "corrected"}

EVALUATION_COLUMNS = [
    "eval_expected_final_behavior",
    "eval_s3_action",
    "eval_expected_route",
    "eval_route_exact_match",
    "eval_route_acceptable",
    "eval_behavior_correct",
    "eval_answer_correct",
    "eval_final_correct",
    "eval_exact_match",
    "eval_contains_gold_answer",
    "eval_gold_contains_answer",
    "eval_token_f1",
    "eval_token_precision",
    "eval_token_recall",
    "eval_abstained",
    "eval_clarified",
    "eval_active_retrieval_triggered",
    "eval_wrong_abstention",
    "eval_wrong_clarification",
    "eval_missing_abstention",
    "eval_missing_clarification",
    "eval_over_retrieval",
    "eval_under_retrieval",
    "eval_retrieval_decision_correct",
    "eval_retrieval_n_gold",
    "eval_retrieval_n_retrieved",
    "eval_retrieval_matched_gold_ids",
    "eval_retrieval_hit_gold",
    "eval_retrieval_full_recall",
    "eval_retrieval_recall",
    "eval_retrieval_precision",
    "eval_final_support_status",
    "eval_any_supported_or_corrected",
    "eval_any_not_enough_info_or_refuted",
    "eval_retrieval_supported",
    "eval_generation_steps",
    "eval_retrieval_steps",
    "eval_chunks_retrieved",
    "eval_total_tokens",
    "eval_latency_seconds",
    "eval_error_type",
    "eval_needs_manual_review",
    "eval_notes",
]

ABSTENTION_MARKERS = [
    "no se", "no sé", "no estoy seguro", "no tengo informacion suficiente",
    "no tengo información suficiente", "no hay informacion suficiente",
    "no hay información suficiente", "informacion insuficiente", "información insuficiente",
    "insufficient information", "not enough information", "i don't know", "i do not know",
    "cannot determine", "can't determine", "not enough evidence", "insufficient evidence",
    "no evidence", "not supported by the context", "not supported by the available context",
    "not supported by the recovered evidence", "the retrieved evidence does not provide",
    "the recovered evidence does not provide",
]

CLARIFICATION_MARKERS = [
    "necesito una aclaracion", "necesito una aclaración", "necesitaria una aclaracion",
    "necesitaría una aclaración", "podrias aclarar", "podrías aclarar", "por favor aclar",
    "falta aclarar", "pregunta ambigua", "no esta claro", "no está claro",
    "need clarification", "needs clarification", "please clarify", "could you clarify",
    "ambiguous", "unclear", "not clear who", "not clear what", "not clear which",
]

EXPECTED_ABSTAIN_VALUES = {"abstain", "abstention", "no_answer", "not_enough_information", "insufficient_information"}
EXPECTED_CLARIFY_VALUES = {"clarify", "ask_clarification", "needs_clarification", "clarification"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "it", "its",
    "as", "at", "from", "but", "not", "does", "do", "did", "can", "could", "would",
    "should", "will", "may", "might", "have", "has", "had", "if", "then", "than",
    "into", "about", "which", "who", "what", "where", "when",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de", "del",
    "en", "con", "por", "para", "es", "son", "fue", "eran", "ser", "que", "este",
    "esta", "esto", "como", "hay", "si", "no", "se", "su", "sus",
}


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


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    text = clean_text(value).lower()
    if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def coerce_int(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: Any) -> str:
    text = strip_accents(clean_text(text).lower())
    text = text.replace("’", "'")
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(value: Any) -> str:
    return normalize_text(value).replace(" ", "_").replace("-", "_")


def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def split_pipe(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


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
        if isinstance(parsed, str):
            return [parsed] if parsed else []
    except json.JSONDecodeError:
        pass
    if "|" in text:
        return split_pipe(text)
    return [text]


def mean_bool(series: pd.Series) -> float | None:
    values: list[bool] = []
    for value in series.dropna():
        coerced = coerce_bool(value)
        if coerced is not None:
            values.append(coerced)
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def infer_abstention(answer: str) -> bool:
    lower = normalize_text(answer)
    return any(normalize_text(marker) in lower for marker in ABSTENTION_MARKERS)


def infer_clarification(answer: str) -> bool:
    lower = normalize_text(answer)
    return any(normalize_text(marker) in lower for marker in CLARIFICATION_MARKERS)


def expected_final_behavior(row: pd.Series) -> str:
    for col in ["expected_final_behavior", "expected_behavior"]:
        value = normalize_label(row.get(col, ""))
        if value:
            if value in EXPECTED_ABSTAIN_VALUES:
                return "abstain"
            if value in EXPECTED_CLARIFY_VALUES:
                return "clarify"
            return "answer"
    return "answer"


def normalize_route(value: Any) -> str:
    text = normalize_label(value)
    aliases = {
        "no_retrieval": "direct", "no_retrieve": "direct", "direct_answer": "direct",
        "answer_directly": "direct", "rag": "retrieve", "retrieval": "retrieve",
        "use_retrieval": "retrieve", "use_rag": "retrieve", "active": "retrieve",
        "active_retrieval": "retrieve", "ask_clarification": "clarify",
        "needs_clarification": "clarify", "clarification": "clarify",
        "not_enough_information": "abstain", "insufficient_information": "abstain",
        "no_answer": "abstain", "refuse": "abstain",
    }
    return aliases.get(text, text)


def get_expected_route(row: pd.Series) -> str:
    value = normalize_route(row.get("expected_route", ""))
    return value if value in VALID_ROUTES else ""


def route_acceptable(row: pd.Series, route: str) -> bool | None:
    acceptable_routes = [normalize_route(x) for x in parse_json_list(row.get("acceptable_routes_json", ""))]
    acceptable_routes = [x for x in acceptable_routes if x in VALID_ROUTES]
    expected_route = get_expected_route(row)
    if acceptable_routes:
        return route in acceptable_routes
    if expected_route:
        return route == expected_route
    return None


def infer_s3_action(*, abstained: bool, clarified: bool, active_retrieval: bool) -> str:
    if clarified:
        return "clarify"
    if abstained:
        return "abstain"
    if active_retrieval:
        return "retrieve"
    return "direct"


def token_scores(prediction: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_gold_answers(row: pd.Series) -> list[str]:
    candidates: list[str] = []
    for col in ["expected_answer", "gold_answer", "gold_answer_text", "best_answer"]:
        value = clean_text(row.get(col, ""))
        if value and value not in candidates:
            candidates.append(value)
    for value in parse_json_list(row.get("correct_answers_json", "")):
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def best_answer_match(prediction: str, gold_answers: list[str]) -> dict[str, Any]:
    pred_norm = normalize_text(prediction)
    best = {
        "gold": "", "exact": False, "contains_gold": False,
        "gold_contains_answer": False, "precision": 0.0, "recall": 0.0,
        "f1": 0.0, "score": 0.0,
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
                "gold": gold, "exact": bool(exact), "contains_gold": bool(contains_gold),
                "gold_contains_answer": bool(gold_contains_answer), "precision": float(precision),
                "recall": float(recall), "f1": float(f1), "score": float(score),
            }
    return best


def option_from_index(value: Any) -> str:
    if is_missing(value):
        return ""
    try:
        idx = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return ""
    if idx in {0, 1, 2, 3}:
        return "ABCD"[idx]
    if idx in {1, 2, 3, 4}:
        return "ABCD"[idx - 1]
    return ""


def normalize_mmlu_answer(value: Any) -> str:
    text = clean_text(value).upper().strip()
    if not text:
        return ""
    if text in VALID_MMLU_OPTIONS:
        return text
    idx_option = option_from_index(text)
    if idx_option:
        return idx_option
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()
    return ""


def choice_from_text(row: pd.Series, text_value: Any) -> str:
    target = normalize_text(text_value)
    if not target:
        return ""
    for label in ["A", "B", "C", "D"]:
        option_text = normalize_text(row.get(label, ""))
        if option_text and target == option_text:
            return label
    raw_choices = clean_text(row.get("answer_choices_json", ""))
    if raw_choices:
        try:
            parsed = json.loads(raw_choices)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for label in ["A", "B", "C", "D"]:
                option_text = normalize_text(parsed.get(label, ""))
                if option_text and target == option_text:
                    return label
        elif isinstance(parsed, list):
            for label, option in zip(["A", "B", "C", "D"], parsed):
                option_text = normalize_text(option)
                if option_text and target == option_text:
                    return label
    return ""


def get_mc_gold(row: pd.Series) -> str:
    for col in ["gold_answer_idx", "answer_idx", "label"]:
        option = option_from_index(row.get(col, ""))
        if option:
            return option
    for col in ["gold_answer", "expected_answer", "gold_answer_text", "best_answer"]:
        direct = normalize_mmlu_answer(row.get(col, ""))
        if direct:
            return direct
        from_text = choice_from_text(row, row.get(col, ""))
        if from_text:
            return from_text
    return ""


def get_mc_prediction(row: pd.Series) -> str:
    for col in ["parsed_choice", "predicted_choice"]:
        value = normalize_mmlu_answer(row.get(col, ""))
        if value:
            return value
    return normalize_mmlu_answer(row.get("parsed_answer", ""))


def is_mmlu_or_mc_row(row: pd.Series) -> bool:
    dataset = normalize_label(row.get("dataset", ""))
    source_dataset = normalize_label(row.get("source_dataset", ""))
    case_type = normalize_label(row.get("case_type", ""))
    s2_case_type = normalize_label(row.get("s2_case_type", ""))
    task_type = normalize_label(row.get("task_type", ""))
    question_format = normalize_label(row.get("question_format", ""))
    return (
        dataset == "mmlu" or source_dataset == "mmlu" or s2_case_type == "direct_mmlu"
        or case_type in {"multiple_choice", "multiplechoice"}
        or task_type == "multiple_choice" or question_format == "multiple_choice"
    )


def retrieval_metrics(row: pd.Series) -> dict[str, Any]:
    gold_ids = set(split_pipe(row.get("gold_evidence_ids", "")))
    retrieved_ids = split_pipe(row.get("parsed_retrieved_chunk_ids", ""))
    if not retrieved_ids:
        retrieved_ids = split_pipe(row.get("retrieved_chunk_ids", ""))
    retrieved_set = set(retrieved_ids)
    matched = gold_ids & retrieved_set
    n_gold = len(gold_ids)
    n_retrieved = len(retrieved_ids)
    hit = bool(matched)
    full_recall = bool(n_gold > 0 and matched == gold_ids)
    recall = len(matched) / n_gold if n_gold else 0.0
    precision = len(matched) / n_retrieved if n_retrieved else 0.0
    return {
        "eval_retrieval_n_gold": int(n_gold),
        "eval_retrieval_n_retrieved": int(n_retrieved),
        "eval_retrieval_matched_gold_ids": "|".join(sorted(matched)),
        "eval_retrieval_hit_gold": bool(hit),
        "eval_retrieval_full_recall": bool(full_recall),
        "eval_retrieval_recall": round(float(recall), 4),
        "eval_retrieval_precision": round(float(precision), 4),
    }


def evaluate_answer_case(row: pd.Series, *, f1_threshold: float) -> tuple[dict[str, Any], list[str]]:
    parsed_answer = clean_text(row.get("parsed_answer", ""))
    notes: list[str] = []
    if is_mmlu_or_mc_row(row):
        pred = get_mc_prediction(row)
        gold = get_mc_gold(row)
        correct = bool(pred in VALID_MMLU_OPTIONS and gold in VALID_MMLU_OPTIONS and pred == gold)
        if not gold:
            notes.append("Gold answer multiple-choice ausente o inválido.")
        if not pred:
            notes.append("Respuesta multiple-choice parseada inválida.")
        return {
            "eval_exact_match": correct,
            "eval_contains_gold_answer": correct,
            "eval_gold_contains_answer": correct,
            "eval_token_f1": 1.0 if correct else 0.0,
            "eval_token_precision": 1.0 if correct else 0.0,
            "eval_token_recall": 1.0 if correct else 0.0,
            "eval_answer_correct": correct,
        }, notes
    gold_answers = get_gold_answers(row)
    match = best_answer_match(parsed_answer, gold_answers)
    exact = bool(match["exact"])
    contains_gold = bool(match["contains_gold"])
    gold_contains = bool(match["gold_contains_answer"])
    f1 = float(match["f1"])
    answer_correct = bool(exact or contains_gold or gold_contains or f1 >= f1_threshold)
    if not gold_answers:
        notes.append("No se encontraron respuestas gold para comparar.")
    elif match["gold"]:
        notes.append(f"best_gold_match={match['gold'][:120]!r}")
    return {
        "eval_exact_match": exact,
        "eval_contains_gold_answer": contains_gold,
        "eval_gold_contains_answer": gold_contains,
        "eval_token_f1": round(f1, 4),
        "eval_token_precision": round(float(match["precision"]), 4),
        "eval_token_recall": round(float(match["recall"]), 4),
        "eval_answer_correct": answer_correct,
    }, notes


def evaluate_row(row: pd.Series, *, f1_threshold: float) -> dict[str, Any]:
    parsed_answer = clean_text(row.get("parsed_answer", ""))
    abstained = bool(coerce_bool(row.get("parsed_abstained"))) or infer_abstention(parsed_answer)
    clarified = bool(coerce_bool(row.get("parsed_clarification"))) or infer_clarification(parsed_answer)
    active_retrieval = bool(coerce_bool(row.get("parsed_active_retrieval_triggered")))
    valid_s3_format = coerce_bool(row.get("valid_s3_format"))
    valid_answer_format = coerce_bool(row.get("valid_answer_format"))
    run_error = bool(coerce_bool(row.get("run_error_present")))
    valid_s3_format = True if valid_s3_format is None else bool(valid_s3_format)
    valid_answer_format = True if valid_answer_format is None else bool(valid_answer_format)

    behavior = expected_final_behavior(row)
    s3_action = infer_s3_action(abstained=abstained, clarified=clarified, active_retrieval=active_retrieval)
    expected_route = get_expected_route(row)
    acceptable = route_acceptable(row, s3_action) if s3_action else None
    route_exact = (s3_action == expected_route) if expected_route else None
    retrieval = retrieval_metrics(row)

    final_support_status = clean_text(row.get("parsed_final_support_status", "")).lower()
    any_supported = bool(coerce_bool(row.get("parsed_any_supported_or_corrected")))
    any_not_enough_or_refuted = bool(coerce_bool(row.get("parsed_any_not_enough_info_or_refuted")))
    retrieval_supported = bool(active_retrieval and (final_support_status in SUPPORTED_STATUSES or any_supported))

    notes: list[str] = []
    needs_manual_review = False
    base_metrics = {
        "eval_exact_match": False,
        "eval_contains_gold_answer": False,
        "eval_gold_contains_answer": False,
        "eval_token_f1": 0.0,
        "eval_token_precision": 0.0,
        "eval_token_recall": 0.0,
        "eval_answer_correct": False,
    }

    if behavior == "answer":
        answer_metrics, answer_notes = evaluate_answer_case(row, f1_threshold=f1_threshold)
        base_metrics.update(answer_metrics)
        notes.extend(answer_notes)
        behavior_correct = bool(not abstained and not clarified)
        final_correct = bool(behavior_correct and base_metrics["eval_answer_correct"])
    elif behavior == "abstain":
        behavior_correct = bool(abstained)
        base_metrics["eval_answer_correct"] = bool(abstained)
        final_correct = bool(abstained)
        if not abstained:
            notes.append("Se esperaba abstención, pero el sistema respondió o no marcó abstención.")
    elif behavior == "clarify":
        behavior_correct = bool(clarified)
        base_metrics["eval_answer_correct"] = bool(clarified)
        final_correct = bool(clarified)
        if not clarified:
            notes.append("Se esperaba aclaración, pero el sistema no pidió aclaración explícita.")
    else:
        behavior_correct = False
        final_correct = False
        notes.append(f"expected_final_behavior desconocido: {behavior}")
        needs_manual_review = True

    wrong_abstention = bool(behavior == "answer" and abstained)
    wrong_clarification = bool(behavior == "answer" and clarified)
    missing_abstention = bool(behavior == "abstain" and not abstained)
    missing_clarification = bool(behavior == "clarify" and not clarified)
    over_retrieval = bool(expected_route and expected_route != "retrieve" and active_retrieval)
    under_retrieval = bool(expected_route == "retrieve" and not active_retrieval)

    if expected_route == "retrieve":
        retrieval_decision_correct = active_retrieval
    elif expected_route == "direct":
        retrieval_decision_correct = not active_retrieval
    elif expected_route == "abstain":
        retrieval_decision_correct = abstained
    elif expected_route == "clarify":
        retrieval_decision_correct = clarified
    else:
        retrieval_decision_correct = None

    if run_error:
        error_type = "run_error"
        notes.append("Hubo error de ejecución.")
        needs_manual_review = True
    elif not valid_s3_format or not valid_answer_format:
        error_type = "format_error"
        notes.append("Formato S3 inválido o respuesta con formato inválido.")
        needs_manual_review = True
    elif final_correct:
        error_type = "correct"
    elif wrong_abstention:
        error_type = "wrong_abstention"
    elif wrong_clarification:
        error_type = "wrong_clarification"
    elif missing_abstention:
        error_type = "missing_abstention"
    elif missing_clarification:
        error_type = "missing_clarification"
    elif under_retrieval:
        error_type = "under_retrieval"
        notes.append("S3 no activó retrieval aunque expected_route=retrieve.")
    elif over_retrieval:
        error_type = "over_retrieval"
        notes.append("S3 activó retrieval aunque expected_route no era retrieve.")
    elif behavior == "answer" and active_retrieval and retrieval["eval_retrieval_n_gold"] > 0 and not retrieval["eval_retrieval_hit_gold"]:
        error_type = "retrieval_miss"
        needs_manual_review = True
    elif behavior == "answer" and active_retrieval and retrieval["eval_retrieval_n_gold"] > 0 and not retrieval["eval_retrieval_full_recall"]:
        error_type = "partial_retrieval_or_generation_error"
        needs_manual_review = True
    elif behavior == "answer":
        error_type = "answer_mismatch"
        needs_manual_review = True
    else:
        error_type = "behavior_error"
        needs_manual_review = True

    return {
        "eval_expected_final_behavior": behavior,
        "eval_s3_action": s3_action,
        "eval_expected_route": expected_route,
        "eval_route_exact_match": route_exact,
        "eval_route_acceptable": acceptable,
        "eval_behavior_correct": bool(behavior_correct),
        **base_metrics,
        "eval_final_correct": bool(final_correct),
        "eval_abstained": bool(abstained),
        "eval_clarified": bool(clarified),
        "eval_active_retrieval_triggered": bool(active_retrieval),
        "eval_wrong_abstention": wrong_abstention,
        "eval_wrong_clarification": wrong_clarification,
        "eval_missing_abstention": missing_abstention,
        "eval_missing_clarification": missing_clarification,
        "eval_over_retrieval": over_retrieval,
        "eval_under_retrieval": under_retrieval,
        "eval_retrieval_decision_correct": retrieval_decision_correct,
        **retrieval,
        "eval_final_support_status": final_support_status,
        "eval_any_supported_or_corrected": any_supported,
        "eval_any_not_enough_info_or_refuted": any_not_enough_or_refuted,
        "eval_retrieval_supported": retrieval_supported,
        "eval_generation_steps": coerce_int(row.get("parsed_num_generation_steps")),
        "eval_retrieval_steps": coerce_int(row.get("parsed_num_retrieval_steps")),
        "eval_chunks_retrieved": coerce_int(row.get("parsed_num_chunks_retrieved_total")),
        "eval_total_tokens": coerce_int(row.get("parsed_total_tokens")),
        "eval_latency_seconds": coerce_float(row.get("parsed_latency_seconds")),
        "eval_error_type": error_type,
        "eval_needs_manual_review": bool(needs_manual_review),
        "eval_notes": " ".join(notes),
    }


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
        "valid_s3_format_rate": mean_bool(df["valid_s3_format"]) if "valid_s3_format" in df else None,
        "valid_answer_format_rate": mean_bool(df["valid_answer_format"]) if "valid_answer_format" in df else None,
        "run_error_rate": mean_bool(df["run_error_present"]) if "run_error_present" in df else None,
        "abstention_rate": mean_bool(df["eval_abstained"]) if "eval_abstained" in df else None,
        "clarification_rate": mean_bool(df["eval_clarified"]) if "eval_clarified" in df else None,
        "active_retrieval_rate": mean_bool(df["eval_active_retrieval_triggered"]) if "eval_active_retrieval_triggered" in df else None,
        "retrieval_decision_accuracy": mean_bool(df["eval_retrieval_decision_correct"]) if "eval_retrieval_decision_correct" in df else None,
        "route_exact_rate": mean_bool(df["eval_route_exact_match"]) if "eval_route_exact_match" in df else None,
        "route_acceptable_rate": mean_bool(df["eval_route_acceptable"]) if "eval_route_acceptable" in df else None,
        "over_retrieval_rate": mean_bool(df["eval_over_retrieval"]) if "eval_over_retrieval" in df else None,
        "under_retrieval_rate": mean_bool(df["eval_under_retrieval"]) if "eval_under_retrieval" in df else None,
        "wrong_abstention_rate": mean_bool(df["eval_wrong_abstention"]) if "eval_wrong_abstention" in df else None,
        "wrong_clarification_rate": mean_bool(df["eval_wrong_clarification"]) if "eval_wrong_clarification" in df else None,
        "missing_abstention_rate": mean_bool(df["eval_missing_abstention"]) if "eval_missing_abstention" in df else None,
        "missing_clarification_rate": mean_bool(df["eval_missing_clarification"]) if "eval_missing_clarification" in df else None,
        "manual_review_rate": mean_bool(df["eval_needs_manual_review"]) if "eval_needs_manual_review" in df else None,
        "retrieval_hit_gold_rate": mean_bool(df["eval_retrieval_hit_gold"]) if "eval_retrieval_hit_gold" in df else None,
        "retrieval_full_recall_rate": mean_bool(df["eval_retrieval_full_recall"]) if "eval_retrieval_full_recall" in df else None,
        "avg_retrieval_recall": mean_numeric(df["eval_retrieval_recall"]) if "eval_retrieval_recall" in df else None,
        "avg_retrieval_precision": mean_numeric(df["eval_retrieval_precision"]) if "eval_retrieval_precision" in df else None,
        "retrieval_supported_rate": mean_bool(df["eval_retrieval_supported"]) if "eval_retrieval_supported" in df else None,
        "avg_confidence": mean_numeric(df["parsed_confidence"]) if "parsed_confidence" in df else None,
        "avg_generation_steps": mean_numeric(df["eval_generation_steps"]) if "eval_generation_steps" in df else None,
        "avg_retrieval_steps": mean_numeric(df["eval_retrieval_steps"]) if "eval_retrieval_steps" in df else None,
        "avg_chunks_retrieved": mean_numeric(df["eval_chunks_retrieved"]) if "eval_chunks_retrieved" in df else None,
        "avg_total_tokens": mean_numeric(df["eval_total_tokens"]) if "eval_total_tokens" in df else None,
        "avg_latency_seconds": mean_numeric(df["eval_latency_seconds"]) if "eval_latency_seconds" in df else None,
    }
    if "eval_error_type" in df.columns:
        summary["error_type_counts"] = {clean_text(k): int(v) for k, v in df["eval_error_type"].value_counts(dropna=False).items()}
    if "eval_expected_final_behavior" in df.columns:
        summary["expected_final_behavior_counts"] = {clean_text(k): int(v) for k, v in df["eval_expected_final_behavior"].value_counts(dropna=False).items()}
    if "eval_s3_action" in df.columns:
        summary["s3_action_counts"] = {clean_text(k): int(v) for k, v in df["eval_s3_action"].value_counts(dropna=False).items()}
    return summary


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [{"group_type": "overall", "group": "all", **summarize_subset(df)}]
    group_cols = [
        "s2_case_type", "task_type", "question_format", "expected_route",
        "eval_s3_action", "eval_expected_final_behavior", "generation_policy",
        "source_dataset", "dataset", "topic",
    ]
    for group_col in group_cols:
        if group_col in df.columns:
            for value, subset in df.groupby(group_col, dropna=False):
                rows.append({"group_type": group_col, "group": clean_text(value), **summarize_subset(subset)})
    if "s2_case_type" in df.columns and "eval_expected_final_behavior" in df.columns:
        for (case_type, behavior), subset in df.groupby(["s2_case_type", "eval_expected_final_behavior"], dropna=False):
            rows.append({"group_type": "s2_case_type_behavior", "group": f"{clean_text(case_type)}::{clean_text(behavior)}", **summarize_subset(subset)})
    if "task_type" in df.columns and "expected_route" in df.columns:
        for (task_type, expected_route), subset in df.groupby(["task_type", "expected_route"], dropna=False):
            rows.append({"group_type": "task_type_expected_route", "group": f"{clean_text(task_type)}::{clean_text(expected_route)}", **summarize_subset(subset)})
    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame) -> None:
    required = {"id", "parsed_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Faltan columnas obligatorias en el CSV parseado: " + ", ".join(sorted(missing)))
    if "valid_s3_format" not in df.columns and "valid_answer_format" not in df.columns:
        raise ValueError("El CSV debe tener valid_s3_format o valid_answer_format. Asegurate de haber corrido parse_s3_outputs.py.")


def evaluate_file(*, input_path: Path, output_path: Path, summary_path: Path, group_summary_path: Path, f1_threshold: float) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")
    df = pd.read_csv(input_path)
    validate_input(df)
    eval_rows = [evaluate_row(row, f1_threshold=f1_threshold) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(eval_rows)
    base_cols = [col for col in df.columns if col not in EVALUATION_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), eval_df], axis=1)
    group_summary_df = build_group_summary(output_df)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "thresholds": {"f1_threshold": f1_threshold},
        "overall": summarize_subset(output_df),
    }
    for group_col in ["s2_case_type", "task_type", "question_format", "expected_route", "eval_s3_action", "eval_expected_final_behavior", "generation_policy", "source_dataset", "dataset"]:
        if group_col in output_df.columns:
            summary[f"by_{group_col}"] = {clean_text(value): summarize_subset(subset) for value, subset in output_df.groupby(group_col, dropna=False)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return output_df, summary, group_summary_df


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResumen evaluación respuestas S3")
    print("--------------------------------")
    overall = summary.get("overall", {})
    print(f"Filas totales: {overall.get('n', 0)}")
    print(f"Final accuracy: {fmt_metric(overall.get('final_accuracy'))}")
    print(f"Behavior accuracy: {fmt_metric(overall.get('behavior_accuracy'))}")
    print(f"Answer accuracy: {fmt_metric(overall.get('answer_accuracy'))}")
    print(f"Exact match rate: {fmt_metric(overall.get('exact_match_rate'))}")
    print(f"Contains gold answer rate: {fmt_metric(overall.get('contains_gold_answer_rate'))}")
    print(f"Avg token F1: {fmt_metric(overall.get('avg_token_f1'))}")
    print(f"Valid S3 format rate: {fmt_metric(overall.get('valid_s3_format_rate'))}")
    print(f"Run error rate: {fmt_metric(overall.get('run_error_rate'))}")
    print(f"Abstention rate: {fmt_metric(overall.get('abstention_rate'))}")
    print(f"Clarification rate: {fmt_metric(overall.get('clarification_rate'))}")
    print(f"Active retrieval rate: {fmt_metric(overall.get('active_retrieval_rate'))}")
    print(f"Retrieval decision accuracy: {fmt_metric(overall.get('retrieval_decision_accuracy'))}")
    print(f"Over-retrieval rate: {fmt_metric(overall.get('over_retrieval_rate'))}")
    print(f"Under-retrieval rate: {fmt_metric(overall.get('under_retrieval_rate'))}")
    print(f"Retrieval hit gold rate: {fmt_metric(overall.get('retrieval_hit_gold_rate'))}")
    print(f"Retrieval full recall rate: {fmt_metric(overall.get('retrieval_full_recall_rate'))}")
    print(f"Retrieval supported rate: {fmt_metric(overall.get('retrieval_supported_rate'))}")
    print(f"Avg generation steps: {fmt_metric(overall.get('avg_generation_steps'))}")
    print(f"Avg retrieval steps: {fmt_metric(overall.get('avg_retrieval_steps'))}")
    print(f"Avg chunks retrieved: {fmt_metric(overall.get('avg_chunks_retrieved'))}")
    print(f"Avg total tokens: {fmt_metric(overall.get('avg_total_tokens'))}")
    print(f"Avg latency seconds: {fmt_metric(overall.get('avg_latency_seconds'))}")
    for title, key in [("Comportamientos esperados", "expected_final_behavior_counts"), ("Acciones S3 inferidas", "s3_action_counts"), ("Tipos de resultado/error", "error_type_counts")]:
        counts = overall.get(key, {})
        if counts:
            print(f"\n{title}:")
            for k, v in counts.items():
                print(f"- {k}: {v}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa respuestas parseadas de S3 FLARE-like active retrieval.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="CSV parseado de entrada, generado por parse_s3_outputs.py.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV evaluado de salida.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="JSON con resumen global y por grupo.")
    parser.add_argument("--group-summary-path", type=Path, default=DEFAULT_GROUP_SUMMARY_PATH, help="CSV con resumen por grupos.")
    parser.add_argument("--f1-threshold", type=float, default=0.70, help="Umbral de token F1 para respuestas abiertas.")
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
    print(f"Resultados de respuestas S3 evaluados guardados en: {args.output_path}")
    print(f"Resumen JSON guardado en: {args.summary_path}")
    print(f"Resumen por grupos guardado en: {args.group_summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

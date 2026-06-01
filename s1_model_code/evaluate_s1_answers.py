#!/usr/bin/env python3
"""
evaluate_s1_answers.py

Evalúa las respuestas parseadas de S1 RAG básico sobre HotpotQA-mini.

Entrada por defecto:
    outputs/s1/generation/hotpotqa_mini_s1_parsed.csv

Salida por defecto:
    outputs/s1/evaluation/hotpotqa_mini_s1_answer_results.csv
    outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary.json
    outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary_by_group.csv

Diseño:
    - No llama al LLM.
    - Evalúa calidad de respuesta con métricas heurísticas reproducibles:
        exact_match, contains_gold_answer, token_f1, answer_correct.
    - Evalúa abstenciones incorrectas.
    - Cruza respuesta con diagnóstico de retrieval usando gold_evidence_ids y
      retrieved_chunk_ids.
    - Separa errores potenciales de retrieval y generación.
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


DEFAULT_INPUT_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_parsed.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s1/evaluation/hotpotqa_mini_s1_answer_results.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary.json")
DEFAULT_GROUP_SUMMARY_PATH = Path("outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary_by_group.csv")

EVALUATION_COLUMNS = [
    "eval_exact_match",
    "eval_contains_gold_answer",
    "eval_gold_contains_answer",
    "eval_token_f1",
    "eval_token_precision",
    "eval_token_recall",
    "eval_answer_correct",
    "eval_abstained",
    "eval_wrong_abstention",
    "eval_retrieval_n_gold",
    "eval_retrieval_n_retrieved",
    "eval_retrieval_matched_gold_ids",
    "eval_retrieval_hit_gold",
    "eval_retrieval_full_recall",
    "eval_retrieval_recall",
    "eval_retrieval_precision",
    "eval_error_type",
    "eval_needs_manual_review",
    "eval_notes",
]

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
    "insufficient information",
    "not enough information",
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
]

EXPECTED_ABSTAIN_VALUES = {
    "abstain",
    "abstention",
    "no_answer",
    "not_enough_information",
    "insufficient_information",
    "clarify",
    "ask_clarification",
}

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "it", "its", "as", "at", "from", "but", "not", "does", "do", "did", "can",
    "could", "would", "should", "will", "may", "might", "have", "has", "had",
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de",
    "del", "en", "con", "por", "para", "es", "son", "fue", "eran", "ser",
    "que", "este", "esta", "esto", "como", "hay",
}


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
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
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


def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def token_scores(prediction: str, gold: str) -> tuple[float, float, float]:
    """Devuelve precision, recall y F1 entre respuesta predicha y gold."""
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
    except json.JSONDecodeError:
        pass

    return [text]


def infer_abstention(answer: str) -> bool:
    lower = normalize_text(answer)
    return any(normalize_text(marker) in lower for marker in ABSTENTION_MARKERS)


def expected_is_abstention(row: pd.Series) -> bool:
    value = normalize_text(row.get("expected_behavior", ""))
    return value in EXPECTED_ABSTAIN_VALUES


def get_gold_answers(row: pd.Series) -> list[str]:
    """Construye una lista de respuestas gold/correctas posibles."""
    candidates: list[str] = []

    for col in ["expected_answer", "gold_answer", "gold_answer_text"]:
        value = clean_text(row.get(col, ""))
        if value and value not in candidates:
            candidates.append(value)

    for value in parse_json_list(row.get("correct_answers_json", "")):
        if value and value not in candidates:
            candidates.append(value)

    return candidates


def best_answer_match(prediction: str, gold_answers: list[str]) -> dict[str, Any]:
    """Calcula la mejor coincidencia entre prediction y lista de gold answers."""
    pred_norm = normalize_text(prediction)

    best = {
        "gold": "",
        "exact": False,
        "contains_gold": False,
        "gold_contains_answer": False,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    if not pred_norm or not gold_answers:
        return best

    for gold in gold_answers:
        gold_norm = normalize_text(gold)
        if not gold_norm:
            continue

        exact = pred_norm == gold_norm
        contains_gold = len(gold_norm) >= 1 and gold_norm in pred_norm
        gold_contains_answer = len(pred_norm) >= 1 and pred_norm in gold_norm
        precision, recall, f1 = token_scores(prediction, gold)

        # Ranking conservador: exact > substring > F1.
        score = f1
        if contains_gold:
            score = max(score, 0.95)
        if gold_contains_answer:
            score = max(score, 0.90)
        if exact:
            score = 1.0

        best_score = best["f1"]
        if best["contains_gold"]:
            best_score = max(best_score, 0.95)
        if best["gold_contains_answer"]:
            best_score = max(best_score, 0.90)
        if best["exact"]:
            best_score = 1.0

        if score > best_score:
            best = {
                "gold": gold,
                "exact": bool(exact),
                "contains_gold": bool(contains_gold),
                "gold_contains_answer": bool(gold_contains_answer),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best


def retrieval_metrics(row: pd.Series) -> dict[str, Any]:
    gold_ids = set(split_pipe(row.get("gold_evidence_ids", "")))
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
        "eval_retrieval_recall": float(recall),
        "eval_retrieval_precision": float(precision),
    }


# ---------------------------------------------------------------------------
# Evaluación por fila
# ---------------------------------------------------------------------------


def evaluate_row(row: pd.Series, *, f1_threshold: float) -> dict[str, Any]:
    parsed_answer = clean_text(row.get("parsed_answer", ""))
    parsed_abstained = coerce_bool(row.get("parsed_abstained"))
    abstained = bool(parsed_abstained) or infer_abstention(parsed_answer)
    expected_abstain = expected_is_abstention(row)
    valid_format = bool(coerce_bool(row.get("valid_format")))
    run_error = bool(coerce_bool(row.get("run_error_present")))

    gold_answers = get_gold_answers(row)
    match = best_answer_match(parsed_answer, gold_answers)
    retrieval = retrieval_metrics(row)

    exact_match = bool(match["exact"])
    contains_gold = bool(match["contains_gold"])
    gold_contains_answer = bool(match["gold_contains_answer"])
    token_precision = float(match["precision"])
    token_recall = float(match["recall"])
    token_f1 = float(match["f1"])

    if expected_abstain:
        answer_correct = abstained
    else:
        answer_correct = bool(
            not abstained
            and not run_error
            and (
                exact_match
                or contains_gold
                or gold_contains_answer
                or token_f1 >= f1_threshold
            )
        )

    wrong_abstention = bool(not expected_abstain and abstained)

    notes: list[str] = []
    needs_manual_review = False

    if run_error:
        error_type = "run_error"
        notes.append("Hubo error de ejecución.")
    elif not valid_format:
        error_type = "format_error"
        notes.append("Formato inválido o parseado por fallback.")
        needs_manual_review = True
    elif answer_correct:
        error_type = "correct"
    elif wrong_abstention:
        if retrieval["eval_retrieval_full_recall"]:
            error_type = "wrong_abstention_despite_full_retrieval"
        elif retrieval["eval_retrieval_hit_gold"]:
            error_type = "wrong_abstention_with_partial_retrieval"
        else:
            error_type = "wrong_abstention_no_gold_retrieved"
        notes.append("El modelo se abstuvo aunque se esperaba respuesta.")
    elif not retrieval["eval_retrieval_hit_gold"]:
        error_type = "retrieval_miss"
        notes.append("No se recuperó ninguna evidencia gold.")
    elif not retrieval["eval_retrieval_full_recall"]:
        error_type = "partial_retrieval_or_generation_error"
        notes.append("Se recuperó evidencia parcial, pero la respuesta no coincide con el gold.")
        needs_manual_review = True
    else:
        error_type = "generation_error_after_full_retrieval"
        notes.append("Se recuperó toda la evidencia gold, pero la respuesta no coincide.")
        needs_manual_review = True

    if match["gold"]:
        notes.append(f"best_gold_match={match['gold'][:120]!r}")

    return {
        "eval_exact_match": exact_match,
        "eval_contains_gold_answer": contains_gold,
        "eval_gold_contains_answer": gold_contains_answer,
        "eval_token_f1": round(token_f1, 4),
        "eval_token_precision": round(token_precision, 4),
        "eval_token_recall": round(token_recall, 4),
        "eval_answer_correct": answer_correct,
        "eval_abstained": bool(abstained),
        "eval_wrong_abstention": wrong_abstention,
        **retrieval,
        "eval_error_type": error_type,
        "eval_needs_manual_review": bool(needs_manual_review),
        "eval_notes": " ".join(notes),
    }


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------


def mean_bool(series: pd.Series) -> float | None:
    if series.empty:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.astype(bool).mean())


def mean_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}

    summary = {
        "n": int(len(df)),
        "answer_accuracy": mean_bool(df["eval_answer_correct"]) if "eval_answer_correct" in df else None,
        "exact_match_rate": mean_bool(df["eval_exact_match"]) if "eval_exact_match" in df else None,
        "contains_gold_answer_rate": mean_bool(df["eval_contains_gold_answer"]) if "eval_contains_gold_answer" in df else None,
        "avg_token_f1": mean_numeric(df["eval_token_f1"]) if "eval_token_f1" in df else None,
        "valid_format_rate": mean_bool(df["valid_format"]) if "valid_format" in df else None,
        "run_error_rate": mean_bool(df["run_error_present"]) if "run_error_present" in df else None,
        "abstention_rate": mean_bool(df["eval_abstained"]) if "eval_abstained" in df else None,
        "wrong_abstention_rate": mean_bool(df["eval_wrong_abstention"]) if "eval_wrong_abstention" in df else None,
        "manual_review_rate": mean_bool(df["eval_needs_manual_review"]) if "eval_needs_manual_review" in df else None,
        "retrieval_hit_gold_rate": mean_bool(df["eval_retrieval_hit_gold"]) if "eval_retrieval_hit_gold" in df else None,
        "retrieval_full_recall_rate": mean_bool(df["eval_retrieval_full_recall"]) if "eval_retrieval_full_recall" in df else None,
        "avg_retrieval_recall": mean_numeric(df["eval_retrieval_recall"]) if "eval_retrieval_recall" in df else None,
        "avg_retrieval_precision": mean_numeric(df["eval_retrieval_precision"]) if "eval_retrieval_precision" in df else None,
        "avg_confidence": mean_numeric(df["parsed_confidence"]) if "parsed_confidence" in df else None,
        "avg_latency_seconds": mean_numeric(df["latency_seconds"]) if "latency_seconds" in df else None,
        "avg_input_tokens": mean_numeric(df["input_tokens"]) if "input_tokens" in df else None,
        "avg_output_tokens": mean_numeric(df["output_tokens"]) if "output_tokens" in df else None,
        "avg_total_tokens": mean_numeric(df["total_tokens"]) if "total_tokens" in df else None,
    }

    if "eval_error_type" in df.columns:
        summary["error_type_counts"] = {
            clean_text(k): int(v)
            for k, v in df["eval_error_type"].value_counts(dropna=False).items()
        }

    return summary


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"group_type": "overall", "group": "all", **summarize_subset(df)})

    for group_col in ["topic", "hotpot_type", "level", "top_k"]:
        if group_col in df.columns:
            for value, subset in df.groupby(group_col, dropna=False):
                rows.append({
                    "group_type": group_col,
                    "group": clean_text(value),
                    **summarize_subset(subset),
                })

    if "topic" in df.columns and "hotpot_type" in df.columns:
        for (topic, hotpot_type), subset in df.groupby(["topic", "hotpot_type"], dropna=False):
            rows.append({
                "group_type": "topic_hotpot_type",
                "group": f"{clean_text(topic)}::{clean_text(hotpot_type)}",
                **summarize_subset(subset),
            })

    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame) -> None:
    required = {"id", "parsed_answer", "valid_format"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV parseado: "
            + ", ".join(sorted(missing))
        )

    if not ({"expected_answer", "gold_answer", "gold_answer_text"} & set(df.columns)):
        raise ValueError(
            "El CSV debe tener al menos una columna de respuesta gold: "
            "expected_answer, gold_answer o gold_answer_text."
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
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    eval_rows = [
        evaluate_row(row, f1_threshold=f1_threshold)
        for _, row in df.iterrows()
    ]
    eval_df = pd.DataFrame(eval_rows)

    # Evita duplicar columnas si se reevalúa un archivo ya evaluado.
    base_cols = [col for col in df.columns if col not in EVALUATION_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), eval_df], axis=1)

    group_summary_df = build_group_summary(output_df)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "thresholds": {
            "f1_threshold": f1_threshold,
        },
        "overall": summarize_subset(output_df),
    }

    for group_col in ["topic", "hotpot_type", "level", "top_k"]:
        if group_col in output_df.columns:
            summary[f"by_{group_col}"] = {
                clean_text(value): summarize_subset(subset)
                for value, subset in output_df.groupby(group_col, dropna=False)
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_summary_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_df, summary, group_summary_df


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResumen de evaluación S1")
    print("------------------------")
    overall = summary.get("overall", {})
    print(f"Filas totales: {overall.get('n', 0)}")
    print(f"Answer accuracy: {fmt_metric(overall.get('answer_accuracy'))}")
    print(f"Exact match rate: {fmt_metric(overall.get('exact_match_rate'))}")
    print(f"Contains gold answer rate: {fmt_metric(overall.get('contains_gold_answer_rate'))}")
    print(f"Avg token F1: {fmt_metric(overall.get('avg_token_f1'))}")
    print(f"Valid format rate: {fmt_metric(overall.get('valid_format_rate'))}")
    print(f"Run error rate: {fmt_metric(overall.get('run_error_rate'))}")
    print(f"Abstention rate: {fmt_metric(overall.get('abstention_rate'))}")
    print(f"Wrong abstention rate: {fmt_metric(overall.get('wrong_abstention_rate'))}")
    print(f"Retrieval hit gold rate: {fmt_metric(overall.get('retrieval_hit_gold_rate'))}")
    print(f"Retrieval full recall rate: {fmt_metric(overall.get('retrieval_full_recall_rate'))}")
    print(f"Avg retrieval recall: {fmt_metric(overall.get('avg_retrieval_recall'))}")
    print(f"Avg total tokens: {fmt_metric(overall.get('avg_total_tokens'))}")
    print(f"Avg latency seconds: {fmt_metric(overall.get('avg_latency_seconds'))}")

    error_counts = overall.get("error_type_counts", {})
    if error_counts:
        print("\nTipos de resultado/error:")
        for key, value in error_counts.items():
            print(f"- {key}: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa respuestas parseadas de S1 RAG básico."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV parseado de entrada, generado por parse_s1_outputs.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV evaluado de salida.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON con resumen global y por grupo.",
    )
    parser.add_argument(
        "--group-summary-path",
        type=Path,
        default=DEFAULT_GROUP_SUMMARY_PATH,
        help="CSV con resumen por topic, hotpot_type, level y top_k.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.70,
        help="Umbral de token F1 para marcar una respuesta como correcta si no hubo match exacto/substring.",
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

    print(f"Resultados evaluados guardados en: {args.output_path}")
    print(f"Resumen JSON guardado en: {args.summary_path}")
    print(f"Resumen por grupos guardado en: {args.group_summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

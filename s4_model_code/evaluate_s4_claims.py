#!/usr/bin/env python3
"""
evaluate_s4_claims.py

Evaluador claim-level para S4: FIRE-like / FIRE-inspired claim verification.

Entrada típica:
    outputs/s4/generation/fire_like_s4_parsed.csv

Salidas típicas:
    outputs/s4/evaluation/fire_like_s4_claim_results.csv
    outputs/s4/evaluation/fire_like_s4_claim_summary.json
    outputs/s4/evaluation/fire_like_s4_claim_summary_by_group.csv

Objetivo
--------
Evaluar la etapa de verificación de claims de S4.

Este archivo:
- No llama al LLM.
- No vuelve a recuperar evidencia.
- No evalúa todavía la respuesta final contra gold_answer. Eso va en evaluate_s4_answers.py.
- Explota s4_claim_results_json a una fila por claim.
- Resume:
    - claims supported/refuted/not_enough_info;
    - claims que requieren evidencia;
    - claims con evidencia soporte;
    - claims con evidencia refutadora;
    - uso de gold evidence si está disponible;
    - rondas de verificación;
    - rondas de retrieval;
    - chunks recuperados;
    - claims que requieren revisión manual;
    - métricas por s2_case_type, expected_route, final_decision, etc.

Uso rápido
----------

python s4_model_code/evaluate_s4_claims.py \
  --input-path outputs/s4/generation/fire_like_s4_parsed_retrieve_test_5_hybrid_claims_llm_verify.csv \
  --claim-output-path outputs/s4/evaluation/fire_like_s4_claim_results_retrieve_test_5_hybrid_llm.csv \
  --summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_retrieve_test_5_hybrid_llm.json \
  --group-summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_by_group_retrieve_test_5_hybrid_llm.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s4/generation/fire_like_s4_parsed.csv")
DEFAULT_CLAIM_OUTPUT_PATH = Path("outputs/s4/evaluation/fire_like_s4_claim_results.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/s4/evaluation/fire_like_s4_claim_summary.json")
DEFAULT_GROUP_SUMMARY_PATH = Path("outputs/s4/evaluation/fire_like_s4_claim_summary_by_group.csv")

VALID_VERDICTS = {"supported", "refuted", "not_enough_info"}
NON_FACTUAL_CLAIM_TYPES = {"abstention", "clarification", "meta", "answer_choice"}

QUESTION_METADATA_COLUMNS = [
    "id",
    "source_system_for_s4",
    "source_question_id",
    "source_dataset",
    "dataset",
    "case_type",
    "s2_case_type",
    "task_type",
    "question_format",
    "expected_route",
    "expected_behavior",
    "expected_final_behavior",
    "requires_retrieval",
    "retrieval_mode",
    "generation_policy",
    "s4_question_type",
    "s4_expected_behavior",
    "s4_final_decision",
    "s4_abstained",
    "s4_correction_applied",
    "s4_claim_strategy",
    "s4_verification_strategy",
    "s4_query_strategy",
    "s4_repair_strategy",
    "s4_num_claims",
    "s4_num_supported_claims",
    "s4_num_refuted_claims",
    "s4_num_nei_claims",
    "s4_num_verification_rounds",
    "s4_num_retrieval_rounds",
    "s4_num_chunks_retrieved_total",
    "s4_total_tokens",
    "s4_latency_seconds",
    "s4_retrieval_latency_seconds",
    "run_error_present",
    "valid_answer_format",
    "parse_error",
    "error",
]

GOLD_EVIDENCE_COLUMNS = [
    "gold_evidence_ids",
    "context_chunk_ids",
    "expected_evidence_ids",
    "supporting_fact_chunk_ids",
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


def normalize_label(value: Any) -> str:
    return clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")


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


def safe_float(value: Any, default: float | None = None) -> float | None:
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


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def parse_list(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]

    text = clean_text(value)
    if not text:
        return []

    parsed = safe_json_loads(text)
    if isinstance(parsed, list):
        return [clean_text(x) for x in parsed if clean_text(x)]

    if "|" in text:
        return [clean_text(x) for x in text.split("|") if clean_text(x)]

    return [text]


def unique_preserve_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def join_pipe(values: list[Any]) -> str:
    return "|".join(unique_preserve_order(values))


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


def normalize_verdict(value: Any) -> str:
    text = normalize_label(value)

    aliases = {
        "support": "supported",
        "supports": "supported",
        "entailment": "supported",
        "entailed": "supported",
        "entails": "supported",
        "contradiction": "refuted",
        "contradicted": "refuted",
        "unsupported": "not_enough_info",
        "not_enough": "not_enough_info",
        "nei": "not_enough_info",
        "unknown": "not_enough_info",
        "insufficient": "not_enough_info",
    }

    text = aliases.get(text, text)
    return text if text in VALID_VERDICTS else text


def normalize_claim_type(value: Any) -> str:
    text = normalize_label(value)
    if not text:
        return "factual"
    return text


# ---------------------------------------------------------------------------
# Extracción de claims desde una fila S4
# ---------------------------------------------------------------------------

def get_claim_results(row: pd.Series) -> list[dict[str, Any]]:
    """
    Prioriza s4_claim_results_json. Si no existe, intenta raw_output.
    """
    parsed = safe_json_loads(row.get("s4_claim_results_json", ""))
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]

    raw = safe_json_loads(row.get("raw_output", ""))
    if isinstance(raw, dict):
        claim_results = raw.get("claim_results", [])
        if isinstance(claim_results, list):
            return [x for x in claim_results if isinstance(x, dict)]

    return []


def get_claims(row: pd.Series) -> list[dict[str, Any]]:
    parsed = safe_json_loads(row.get("s4_claims_json", ""))
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]

    raw = safe_json_loads(row.get("raw_output", ""))
    if isinstance(raw, dict):
        claims = raw.get("claims", [])
        if isinstance(claims, list):
            return [x for x in claims if isinstance(x, dict)]

    return []


def get_gold_evidence_ids(row: pd.Series) -> list[str]:
    values: list[str] = []

    for col in GOLD_EVIDENCE_COLUMNS:
        if col in row.index:
            values.extend(parse_list(row.get(col, "")))

    return unique_preserve_order(values)


def collect_retrieved_from_rounds(rounds_trace: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for round_item in rounds_trace:
        if not isinstance(round_item, dict):
            continue
        ids.extend(parse_list(round_item.get("retrieved_chunk_ids", [])))
    return unique_preserve_order(ids)


def collect_queries_from_rounds(rounds_trace: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for round_item in rounds_trace:
        if not isinstance(round_item, dict):
            continue
        query = clean_text(round_item.get("retrieval_query", ""))
        if query:
            queries.append(query)
    return unique_preserve_order(queries)


def collect_verdicts_from_rounds(rounds_trace: list[dict[str, Any]]) -> list[str]:
    verdicts: list[str] = []
    for round_item in rounds_trace:
        if not isinstance(round_item, dict):
            continue
        verdict = normalize_verdict(round_item.get("verdict", ""))
        if verdict:
            verdicts.append(verdict)
    return verdicts


def claim_manual_review_reason(
    *,
    verdict: str,
    claim_type: str,
    requires_evidence: bool,
    s4_final_decision: str,
    expected_route: str,
    supporting_chunk_ids: list[str],
    refuting_chunk_ids: list[str],
    gold_evidence_ids: list[str],
    uses_gold_evidence: bool,
    confidence: float | None,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if verdict not in VALID_VERDICTS:
        reasons.append("unknown_verdict")

    if requires_evidence and verdict == "not_enough_info":
        reasons.append("requires_evidence_but_nei")

    if verdict == "supported" and not supporting_chunk_ids and requires_evidence:
        reasons.append("supported_without_supporting_chunk")

    if verdict == "refuted" and not refuting_chunk_ids and requires_evidence:
        reasons.append("refuted_without_refuting_chunk")

    if verdict == "refuted":
        reasons.append("claim_refuted")

    if s4_final_decision == "unchanged" and verdict in {"refuted", "not_enough_info"} and requires_evidence:
        reasons.append("unchanged_despite_unsupported_claim")

    if expected_route == "retrieve" and verdict == "supported" and gold_evidence_ids and not uses_gold_evidence:
        reasons.append("supported_without_gold_evidence")

    if confidence is not None and confidence < 0.50 and verdict == "supported":
        reasons.append("low_confidence_supported")

    if claim_type in NON_FACTUAL_CLAIM_TYPES and requires_evidence:
        reasons.append("non_factual_marked_requires_evidence")

    return bool(reasons), "|".join(reasons)


def explode_claims(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for question_index, row in df.iterrows():
        question_id = clean_text(row.get("id", f"row_{question_index}"))
        claim_results = get_claim_results(row)
        claims = get_claims(row)
        gold_evidence_ids = get_gold_evidence_ids(row)

        # Fallback: si hay claims pero no claim_results, crear filas NEI marcadas.
        if not claim_results and claims:
            claim_results = [
                {
                    **claim,
                    "verdict": "not_enough_info",
                    "confidence": 0.0,
                    "rationale": "No hay claim_results disponibles; fallback del evaluador.",
                    "supporting_chunk_ids": [],
                    "refuting_chunk_ids": [],
                    "evidence_ids": [],
                    "retrieved_chunk_ids": [],
                    "rounds_trace": [],
                    "rounds": 0,
                }
                for claim in claims
            ]

        # Fallback: si no hay claims, crear una fila sintética por pregunta para no perder el caso.
        if not claim_results:
            claim_results = [
                {
                    "claim_id": "no_claims",
                    "claim_text": "",
                    "claim_type": "none",
                    "requires_evidence": False,
                    "importance": "none",
                    "verdict": "not_enough_info",
                    "confidence": 0.0,
                    "rationale": "No se encontraron claims ni claim_results.",
                    "supporting_chunk_ids": [],
                    "refuting_chunk_ids": [],
                    "evidence_ids": [],
                    "retrieved_chunk_ids": [],
                    "rounds_trace": [],
                    "rounds": 0,
                }
            ]

        for claim_position, claim in enumerate(claim_results, start=1):
            if not isinstance(claim, dict):
                continue

            claim_id = clean_text(claim.get("claim_id", f"c{claim_position}")) or f"c{claim_position}"
            claim_text = clean_text(claim.get("claim_text", ""))
            claim_type = normalize_claim_type(claim.get("claim_type", "factual"))
            requires_evidence = bool(coerce_bool(claim.get("requires_evidence"), default=claim_type not in NON_FACTUAL_CLAIM_TYPES))
            importance = normalize_label(claim.get("importance", "core")) or "core"

            verdict = normalize_verdict(claim.get("verdict", "not_enough_info"))
            confidence = safe_float(claim.get("confidence", None), default=None)
            rationale = clean_text(claim.get("rationale", ""))

            supporting_chunk_ids = parse_list(claim.get("supporting_chunk_ids", []))
            refuting_chunk_ids = parse_list(claim.get("refuting_chunk_ids", []))
            evidence_ids = parse_list(claim.get("evidence_ids", []))
            retrieved_chunk_ids = parse_list(claim.get("retrieved_chunk_ids", []))

            rounds_trace = claim.get("rounds_trace", [])
            if not isinstance(rounds_trace, list):
                rounds_trace = []

            retrieved_from_rounds = collect_retrieved_from_rounds(rounds_trace)
            retrieval_queries = collect_queries_from_rounds(rounds_trace)
            round_verdicts = collect_verdicts_from_rounds(rounds_trace)

            if not retrieved_chunk_ids:
                retrieved_chunk_ids = retrieved_from_rounds

            if not evidence_ids:
                evidence_ids = unique_preserve_order(
                    supporting_chunk_ids + refuting_chunk_ids + retrieved_chunk_ids
                )

            rounds = safe_int(claim.get("rounds", len(rounds_trace)), default=len(rounds_trace))
            if rounds <= 0 and rounds_trace:
                rounds = len(rounds_trace)

            num_retrieval_rounds_for_claim = sum(
                1 for item in rounds_trace
                if isinstance(item, dict) and parse_list(item.get("retrieved_chunk_ids", []))
            )

            num_chunks_retrieved_for_claim = len(unique_preserve_order(retrieved_chunk_ids))
            num_evidence_ids_for_claim = len(unique_preserve_order(evidence_ids))
            num_supporting_chunks = len(unique_preserve_order(supporting_chunk_ids))
            num_refuting_chunks = len(unique_preserve_order(refuting_chunk_ids))

            supported = verdict == "supported"
            refuted = verdict == "refuted"
            nei = verdict == "not_enough_info"

            gold_set = set(gold_evidence_ids)
            supporting_set = set(supporting_chunk_ids)
            evidence_set = set(evidence_ids)
            retrieved_set = set(retrieved_chunk_ids)

            gold_supporting_overlap = sorted(gold_set & supporting_set)
            gold_evidence_overlap = sorted(gold_set & evidence_set)
            gold_retrieved_overlap = sorted(gold_set & retrieved_set)

            uses_gold_supporting = bool(gold_supporting_overlap)
            uses_gold_evidence = bool(gold_evidence_overlap)
            retrieved_gold_evidence = bool(gold_retrieved_overlap)

            expected_route = normalize_label(row.get("expected_route", ""))
            s4_final_decision = normalize_label(row.get("s4_final_decision", ""))

            needs_manual_review, manual_review_reason = claim_manual_review_reason(
                verdict=verdict,
                claim_type=claim_type,
                requires_evidence=requires_evidence,
                s4_final_decision=s4_final_decision,
                expected_route=expected_route,
                supporting_chunk_ids=supporting_chunk_ids,
                refuting_chunk_ids=refuting_chunk_ids,
                gold_evidence_ids=gold_evidence_ids,
                uses_gold_evidence=uses_gold_evidence,
                confidence=confidence,
            )

            out: dict[str, Any] = {}

            for col in QUESTION_METADATA_COLUMNS:
                if col in df.columns:
                    out[col] = row.get(col, "")

            out.update(
                {
                    "question_row_index": int(question_index),
                    "question_id": question_id,
                    "claim_position": int(claim_position),
                    "claim_id": claim_id,
                    "claim_global_id": f"{question_id}::{claim_id}",
                    "claim_text": claim_text,
                    "claim_type": claim_type,
                    "claim_requires_evidence": requires_evidence,
                    "claim_importance": importance,
                    "claim_verdict": verdict,
                    "claim_confidence": confidence,
                    "claim_rationale": rationale,
                    "claim_supported": supported,
                    "claim_refuted": refuted,
                    "claim_nei": nei,
                    "claim_is_factual": claim_type == "factual",
                    "claim_is_non_factual": claim_type in NON_FACTUAL_CLAIM_TYPES,
                    "claim_is_core": importance == "core",
                    "claim_is_supporting": importance == "supporting",
                    "claim_is_low_importance": importance == "low",

                    "claim_rounds": int(rounds),
                    "claim_retrieval_rounds": int(num_retrieval_rounds_for_claim),
                    "claim_num_supporting_chunks": int(num_supporting_chunks),
                    "claim_num_refuting_chunks": int(num_refuting_chunks),
                    "claim_num_evidence_ids": int(num_evidence_ids_for_claim),
                    "claim_num_retrieved_chunks": int(num_chunks_retrieved_for_claim),

                    "claim_supporting_chunk_ids": join_pipe(supporting_chunk_ids),
                    "claim_refuting_chunk_ids": join_pipe(refuting_chunk_ids),
                    "claim_evidence_ids": join_pipe(evidence_ids),
                    "claim_retrieved_chunk_ids": join_pipe(retrieved_chunk_ids),
                    "claim_retrieval_queries": safe_json_dumps(retrieval_queries),
                    "claim_round_verdicts": join_pipe(round_verdicts),
                    "claim_rounds_trace_json": safe_json_dumps(rounds_trace),
                    "claim_raw_json": safe_json_dumps(claim),

                    "gold_evidence_ids_joined": join_pipe(gold_evidence_ids),
                    "claim_gold_supporting_overlap": join_pipe(gold_supporting_overlap),
                    "claim_gold_evidence_overlap": join_pipe(gold_evidence_overlap),
                    "claim_gold_retrieved_overlap": join_pipe(gold_retrieved_overlap),
                    "claim_uses_gold_supporting": uses_gold_supporting,
                    "claim_uses_gold_evidence": uses_gold_evidence,
                    "claim_retrieved_gold_evidence": retrieved_gold_evidence,

                    "claim_needs_manual_review": needs_manual_review,
                    "claim_manual_review_reason": manual_review_reason,
                }
            )

            rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------

def summarize_claim_subset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_claim_rows": 0}

    summary = {
        "n_claim_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()) if "question_id" in df.columns else None,

        "supported_rate": mean_bool(df["claim_supported"]) if "claim_supported" in df.columns else None,
        "refuted_rate": mean_bool(df["claim_refuted"]) if "claim_refuted" in df.columns else None,
        "nei_rate": mean_bool(df["claim_nei"]) if "claim_nei" in df.columns else None,

        "requires_evidence_rate": mean_bool(df["claim_requires_evidence"]) if "claim_requires_evidence" in df.columns else None,
        "factual_claim_rate": mean_bool(df["claim_is_factual"]) if "claim_is_factual" in df.columns else None,
        "non_factual_claim_rate": mean_bool(df["claim_is_non_factual"]) if "claim_is_non_factual" in df.columns else None,
        "core_claim_rate": mean_bool(df["claim_is_core"]) if "claim_is_core" in df.columns else None,

        "avg_claim_confidence": mean_numeric(df["claim_confidence"]) if "claim_confidence" in df.columns else None,
        "avg_claim_rounds": mean_numeric(df["claim_rounds"]) if "claim_rounds" in df.columns else None,
        "avg_claim_retrieval_rounds": mean_numeric(df["claim_retrieval_rounds"]) if "claim_retrieval_rounds" in df.columns else None,
        "avg_claim_supporting_chunks": mean_numeric(df["claim_num_supporting_chunks"]) if "claim_num_supporting_chunks" in df.columns else None,
        "avg_claim_refuting_chunks": mean_numeric(df["claim_num_refuting_chunks"]) if "claim_num_refuting_chunks" in df.columns else None,
        "avg_claim_evidence_ids": mean_numeric(df["claim_num_evidence_ids"]) if "claim_num_evidence_ids" in df.columns else None,
        "avg_claim_retrieved_chunks": mean_numeric(df["claim_num_retrieved_chunks"]) if "claim_num_retrieved_chunks" in df.columns else None,

        "claim_uses_gold_supporting_rate": mean_bool(df["claim_uses_gold_supporting"]) if "claim_uses_gold_supporting" in df.columns else None,
        "claim_uses_gold_evidence_rate": mean_bool(df["claim_uses_gold_evidence"]) if "claim_uses_gold_evidence" in df.columns else None,
        "claim_retrieved_gold_evidence_rate": mean_bool(df["claim_retrieved_gold_evidence"]) if "claim_retrieved_gold_evidence" in df.columns else None,

        "claim_manual_review_rate": mean_bool(df["claim_needs_manual_review"]) if "claim_needs_manual_review" in df.columns else None,
    }

    for col, key in [
        ("claim_verdict", "claim_verdict_counts"),
        ("claim_type", "claim_type_counts"),
        ("claim_importance", "claim_importance_counts"),
        ("claim_manual_review_reason", "manual_review_reason_counts"),
        ("s4_final_decision", "s4_final_decision_counts"),
        ("s4_claim_strategy", "s4_claim_strategy_counts"),
        ("s4_verification_strategy", "s4_verification_strategy_counts"),
    ]:
        if col in df.columns:
            summary[key] = {
                clean_text(k): int(v)
                for k, v in df[col].value_counts(dropna=False).items()
            }

    return summary


def summarize_questions_from_claims(claim_df: pd.DataFrame) -> dict[str, Any]:
    """
    Resume a nivel pregunta a partir del claim-level.
    """
    if claim_df.empty:
        return {
            "n_questions": 0,
        }

    grouped = claim_df.groupby("question_id", dropna=False)

    question_rows: list[dict[str, Any]] = []
    for question_id, subset in grouped:
        n_claims = int(len(subset))
        n_supported = int(subset["claim_supported"].sum()) if "claim_supported" in subset else 0
        n_refuted = int(subset["claim_refuted"].sum()) if "claim_refuted" in subset else 0
        n_nei = int(subset["claim_nei"].sum()) if "claim_nei" in subset else 0
        n_requires_evidence = int(subset["claim_requires_evidence"].sum()) if "claim_requires_evidence" in subset else 0

        question_rows.append(
            {
                "question_id": clean_text(question_id),
                "n_claims": n_claims,
                "all_claims_supported": bool(n_claims > 0 and n_supported == n_claims),
                "all_evidence_required_claims_supported": bool(
                    n_requires_evidence == 0
                    or (
                        n_requires_evidence > 0
                        and int((subset["claim_supported"] & subset["claim_requires_evidence"]).sum()) == n_requires_evidence
                    )
                ),
                "any_claim_refuted": bool(n_refuted > 0),
                "any_claim_nei": bool(n_nei > 0),
                "any_claim_needs_manual_review": bool(subset["claim_needs_manual_review"].any()) if "claim_needs_manual_review" in subset else False,
                "claim_support_rate": n_supported / n_claims if n_claims else 0.0,
                "claim_refuted_rate": n_refuted / n_claims if n_claims else 0.0,
                "claim_nei_rate": n_nei / n_claims if n_claims else 0.0,
                "avg_claim_confidence": mean_numeric(subset["claim_confidence"]) if "claim_confidence" in subset else None,
            }
        )

    qdf = pd.DataFrame(question_rows)

    return {
        "n_questions": int(len(qdf)),
        "question_all_claims_supported_rate": mean_bool(qdf["all_claims_supported"]),
        "question_all_evidence_required_claims_supported_rate": mean_bool(qdf["all_evidence_required_claims_supported"]),
        "question_any_claim_refuted_rate": mean_bool(qdf["any_claim_refuted"]),
        "question_any_claim_nei_rate": mean_bool(qdf["any_claim_nei"]),
        "question_manual_review_rate": mean_bool(qdf["any_claim_needs_manual_review"]),
        "avg_question_claim_support_rate": mean_numeric(qdf["claim_support_rate"]),
        "avg_question_claim_nei_rate": mean_numeric(qdf["claim_nei_rate"]),
        "avg_question_claim_refuted_rate": mean_numeric(qdf["claim_refuted_rate"]),
        "avg_question_claim_confidence": mean_numeric(qdf["avg_claim_confidence"]),
    }


def build_group_summary(claim_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "group_type": "overall",
            "group": "all",
            **summarize_claim_subset(claim_df),
        }
    )

    group_cols = [
        "s2_case_type",
        "dataset",
        "case_type",
        "expected_route",
        "s4_final_decision",
        "s4_claim_strategy",
        "s4_verification_strategy",
        "s4_query_strategy",
        "s4_repair_strategy",
        "claim_type",
        "claim_importance",
        "claim_verdict",
        "claim_requires_evidence",
        "source_system_for_s4",
    ]

    for col in group_cols:
        if col in claim_df.columns:
            for value, subset in claim_df.groupby(col, dropna=False):
                rows.append(
                    {
                        "group_type": col,
                        "group": clean_text(value),
                        **summarize_claim_subset(subset),
                    }
                )

    if "s2_case_type" in claim_df.columns and "claim_verdict" in claim_df.columns:
        for (case_type, verdict), subset in claim_df.groupby(["s2_case_type", "claim_verdict"], dropna=False):
            rows.append(
                {
                    "group_type": "s2_case_type::claim_verdict",
                    "group": f"{clean_text(case_type)}::{clean_text(verdict)}",
                    **summarize_claim_subset(subset),
                }
            )

    if "s4_final_decision" in claim_df.columns and "claim_verdict" in claim_df.columns:
        for (decision, verdict), subset in claim_df.groupby(["s4_final_decision", "claim_verdict"], dropna=False):
            rows.append(
                {
                    "group_type": "s4_final_decision::claim_verdict",
                    "group": f"{clean_text(decision)}::{clean_text(verdict)}",
                    **summarize_claim_subset(subset),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline de archivo
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("El CSV S4 debe tener columna 'id'.")

    if "s4_claim_results_json" not in df.columns and "raw_output" not in df.columns:
        raise ValueError(
            "El CSV S4 debe tener s4_claim_results_json o raw_output. "
            "Ejecutá primero parse_s4_outputs.py o usá salida de run_s4_fire_like.py."
        )


def evaluate_claims_file(
    *,
    input_path: Path,
    claim_output_path: Path,
    summary_path: Path,
    group_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró input-path: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    claim_df = explode_claims(df)
    group_summary_df = build_group_summary(claim_df)

    overall_claim_summary = summarize_claim_subset(claim_df)
    question_summary = summarize_questions_from_claims(claim_df)

    summary = {
        "input_path": str(input_path),
        "claim_output_path": str(claim_output_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "overall_claim_level": overall_claim_summary,
        "overall_question_level_from_claims": question_summary,
    }

    for group_col in [
        "s2_case_type",
        "dataset",
        "case_type",
        "expected_route",
        "s4_final_decision",
        "s4_claim_strategy",
        "s4_verification_strategy",
        "claim_type",
        "claim_importance",
        "claim_verdict",
        "claim_requires_evidence",
    ]:
        if group_col in claim_df.columns:
            summary[f"by_{group_col}"] = {
                clean_text(value): summarize_claim_subset(subset)
                for value, subset in claim_df.groupby(group_col, dropna=False)
            }

    claim_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_summary_path.parent.mkdir(parents=True, exist_ok=True)

    claim_df.to_csv(claim_output_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return claim_df, summary, group_summary_df


def print_summary(summary: dict[str, Any]) -> None:
    print("\nResumen evaluación claim-level S4")
    print("---------------------------------")

    claim = summary.get("overall_claim_level", {})
    q = summary.get("overall_question_level_from_claims", {})

    print(f"Claims evaluados: {claim.get('n_claim_rows', 0)}")
    print(f"Preguntas evaluadas: {claim.get('n_questions', q.get('n_questions', 0))}")

    print("\nClaim-level:")
    print(f"Supported rate: {fmt(claim.get('supported_rate'))}")
    print(f"Refuted rate: {fmt(claim.get('refuted_rate'))}")
    print(f"NEI rate: {fmt(claim.get('nei_rate'))}")
    print(f"Requires evidence rate: {fmt(claim.get('requires_evidence_rate'))}")
    print(f"Factual claim rate: {fmt(claim.get('factual_claim_rate'))}")
    print(f"Avg confidence: {fmt(claim.get('avg_claim_confidence'))}")
    print(f"Avg rounds: {fmt(claim.get('avg_claim_rounds'))}")
    print(f"Avg retrieval rounds: {fmt(claim.get('avg_claim_retrieval_rounds'))}")
    print(f"Avg retrieved chunks: {fmt(claim.get('avg_claim_retrieved_chunks'))}")
    print(f"Manual review rate: {fmt(claim.get('claim_manual_review_rate'))}")

    print("\nGold evidence usage, si hay gold evidence:")
    print(f"Uses gold supporting rate: {fmt(claim.get('claim_uses_gold_supporting_rate'))}")
    print(f"Uses gold evidence rate: {fmt(claim.get('claim_uses_gold_evidence_rate'))}")
    print(f"Retrieved gold evidence rate: {fmt(claim.get('claim_retrieved_gold_evidence_rate'))}")

    print("\nQuestion-level desde claims:")
    print(f"All claims supported rate: {fmt(q.get('question_all_claims_supported_rate'))}")
    print(f"All evidence-required claims supported rate: {fmt(q.get('question_all_evidence_required_claims_supported_rate'))}")
    print(f"Any claim refuted rate: {fmt(q.get('question_any_claim_refuted_rate'))}")
    print(f"Any claim NEI rate: {fmt(q.get('question_any_claim_nei_rate'))}")
    print(f"Question manual review rate: {fmt(q.get('question_manual_review_rate'))}")

    counts = claim.get("claim_verdict_counts", {})
    if counts:
        print("\nConteo por veredicto:")
        for key, value in counts.items():
            print(f"- {key}: {value}")

    type_counts = claim.get("claim_type_counts", {})
    if type_counts:
        print("\nConteo por tipo de claim:")
        for key, value in type_counts.items():
            print(f"- {key}: {value}")

    review_counts = claim.get("manual_review_reason_counts", {})
    if review_counts:
        print("\nMotivos de revisión manual:")
        for key, value in review_counts.items():
            print(f"- {key}: {value}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa claims verificados por S4 FIRE-like."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV S4 parseado o raw con s4_claim_results_json/raw_output.",
    )
    parser.add_argument(
        "--claim-output-path",
        type=Path,
        default=DEFAULT_CLAIM_OUTPUT_PATH,
        help="CSV de salida con una fila por claim.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON con resumen claim-level.",
    )
    parser.add_argument(
        "--group-summary-path",
        type=Path,
        default=DEFAULT_GROUP_SUMMARY_PATH,
        help="CSV con resumen por grupos.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _, summary, _ = evaluate_claims_file(
        input_path=args.input_path,
        claim_output_path=args.claim_output_path,
        summary_path=args.summary_path,
        group_summary_path=args.group_summary_path,
    )

    print(f"Claims S4 evaluados guardados en: {args.claim_output_path}")
    print(f"Resumen JSON guardado en: {args.summary_path}")
    print(f"Resumen por grupos guardado en: {args.group_summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

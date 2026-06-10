#!/usr/bin/env python3
"""
verify_s4_results.py

Verificador final / sanity-check para resultados S4.

Este archivo corresponde al paso 9 del pipeline S4:

    run_s4_fire_like.py
        -> parse_s4_outputs.py
            -> evaluate_s4_claims.py
                -> evaluate_s4_answers.py
                    -> verify_s4_results.py

Objetivo
--------
Revisar de forma automática si los outputs de S4 son consistentes y útiles
antes de pasar a la comparación S2 vs S4 / S3 vs S4.

Este script:
- No llama al LLM.
- No modifica outputs previos.
- Lee, si existen:
    1. CSV parseado de S4.
    2. CSV claim-level evaluado.
    3. CSV answer-level evaluado.
- Genera:
    - JSON con resumen final.
    - TXT legible con conclusiones.
    - CSV de problemas por pregunta.
    - CSV de problemas por claim.
    - CSV de sanity checks.

Uso rápido
----------

Caso retrieve test 5 hybrid+llm:

python s4_model_code/verify_s4_results.py \
  --s4-parsed-path outputs/s4/generation/fire_like_s4_parsed_retrieve_test_5_hybrid_claims_llm_verify.csv \
  --claim-results-path outputs/s4/evaluation/fire_like_s4_claim_results_retrieve_test_5_hybrid_llm.csv \
  --answer-results-path outputs/s4/evaluation/fire_like_s4_answer_results_retrieve_test_5_hybrid_llm.csv \
  --output-dir outputs/s4/verification \
  --prefix retrieve_test_5_hybrid_llm

Salidas:
    outputs/s4/verification/retrieve_test_5_hybrid_llm_summary.json
    outputs/s4/verification/retrieve_test_5_hybrid_llm_summary.txt
    outputs/s4/verification/retrieve_test_5_hybrid_llm_problematic_questions.csv
    outputs/s4/verification/retrieve_test_5_hybrid_llm_problematic_claims.csv
    outputs/s4/verification/retrieve_test_5_hybrid_llm_sanity_checks.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_S4_PARSED_PATH = Path("outputs/s4/generation/fire_like_s4_parsed.csv")
DEFAULT_CLAIM_RESULTS_PATH = Path("outputs/s4/evaluation/fire_like_s4_claim_results.csv")
DEFAULT_ANSWER_RESULTS_PATH = Path("outputs/s4/evaluation/fire_like_s4_answer_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/s4/verification")
DEFAULT_PREFIX = "verify_s4"

DEFAULT_LATENCY_WARN_SECONDS = 30.0
DEFAULT_TOTAL_TOKENS_WARN = 1500
DEFAULT_CHUNKS_WARN = 15
DEFAULT_RETRIEVAL_ROUNDS_WARN = 5
DEFAULT_VERIFICATION_ROUNDS_WARN = 8


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
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def mean_bool(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.astype(bool).mean())


def mean_numeric(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def value_counts_dict(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df is None or df.empty or col not in df.columns:
        return {}
    return {
        clean_text(k): int(v)
        for k, v in df[col].value_counts(dropna=False).items()
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def read_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"No se encontró archivo requerido: {path}")
    return None


def add_issue(
    issues: list[dict[str, Any]],
    *,
    level: str,
    source: str,
    item_id: str,
    category: str,
    message: str,
    detail: str = "",
    suggested_action: str = "",
) -> None:
    issues.append(
        {
            "level": level,
            "source": source,
            "id": item_id,
            "category": category,
            "message": message,
            "detail": detail,
            "suggested_action": suggested_action,
        }
    )


def add_check(
    checks: list[dict[str, Any]],
    *,
    check_name: str,
    status: str,
    message: str,
    value: Any = "",
    threshold: Any = "",
) -> None:
    checks.append(
        {
            "check_name": check_name,
            "status": status,
            "message": message,
            "value": value,
            "threshold": threshold,
        }
    )


# ---------------------------------------------------------------------------
# Resúmenes base
# ---------------------------------------------------------------------------

def summarize_parsed(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "n": int(len(df)),
        "duplicate_id_count": int(df["id"].duplicated().sum()) if "id" in df.columns else None,
        "final_decision_counts": value_counts_dict(df, "s4_final_decision"),
        "parse_method_counts": value_counts_dict(df, "parse_method"),
        "valid_answer_format_rate": mean_bool(df["valid_answer_format"]) if "valid_answer_format" in df.columns else None,
        "run_error_rate": mean_bool(df["run_error_present"]) if "run_error_present" in df.columns else None,
        "s4_abstention_rate": mean_bool(df["s4_abstained"]) if "s4_abstained" in df.columns else None,
        "s4_correction_rate": mean_bool(df["s4_correction_applied"]) if "s4_correction_applied" in df.columns else None,
        "avg_claims": mean_numeric(df["s4_num_claims"]) if "s4_num_claims" in df.columns else None,
        "avg_supported_claims": mean_numeric(df["s4_num_supported_claims"]) if "s4_num_supported_claims" in df.columns else None,
        "avg_refuted_claims": mean_numeric(df["s4_num_refuted_claims"]) if "s4_num_refuted_claims" in df.columns else None,
        "avg_nei_claims": mean_numeric(df["s4_num_nei_claims"]) if "s4_num_nei_claims" in df.columns else None,
        "avg_verification_rounds": mean_numeric(df["s4_num_verification_rounds"]) if "s4_num_verification_rounds" in df.columns else None,
        "avg_retrieval_rounds": mean_numeric(df["s4_num_retrieval_rounds"]) if "s4_num_retrieval_rounds" in df.columns else None,
        "avg_chunks_retrieved": mean_numeric(df["s4_num_chunks_retrieved_total"]) if "s4_num_chunks_retrieved_total" in df.columns else None,
        "avg_total_tokens": mean_numeric(df["s4_total_tokens"]) if "s4_total_tokens" in df.columns else None,
        "avg_latency_seconds": mean_numeric(df["s4_latency_seconds"]) if "s4_latency_seconds" in df.columns else None,
    }

    return summary


def summarize_claims(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "n_claims": int(len(df)),
        "n_questions": int(df["question_id"].nunique()) if "question_id" in df.columns else None,
        "verdict_counts": value_counts_dict(df, "claim_verdict"),
        "claim_type_counts": value_counts_dict(df, "claim_type"),
        "supported_rate": mean_bool(df["claim_supported"]) if "claim_supported" in df.columns else None,
        "refuted_rate": mean_bool(df["claim_refuted"]) if "claim_refuted" in df.columns else None,
        "nei_rate": mean_bool(df["claim_nei"]) if "claim_nei" in df.columns else None,
        "requires_evidence_rate": mean_bool(df["claim_requires_evidence"]) if "claim_requires_evidence" in df.columns else None,
        "manual_review_rate": mean_bool(df["claim_needs_manual_review"]) if "claim_needs_manual_review" in df.columns else None,
        "uses_gold_supporting_rate": mean_bool(df["claim_uses_gold_supporting"]) if "claim_uses_gold_supporting" in df.columns else None,
        "uses_gold_evidence_rate": mean_bool(df["claim_uses_gold_evidence"]) if "claim_uses_gold_evidence" in df.columns else None,
        "retrieved_gold_evidence_rate": mean_bool(df["claim_retrieved_gold_evidence"]) if "claim_retrieved_gold_evidence" in df.columns else None,
        "avg_confidence": mean_numeric(df["claim_confidence"]) if "claim_confidence" in df.columns else None,
        "avg_rounds": mean_numeric(df["claim_rounds"]) if "claim_rounds" in df.columns else None,
        "avg_retrieval_rounds": mean_numeric(df["claim_retrieval_rounds"]) if "claim_retrieval_rounds" in df.columns else None,
        "avg_retrieved_chunks": mean_numeric(df["claim_num_retrieved_chunks"]) if "claim_num_retrieved_chunks" in df.columns else None,
    }

    if "question_id" in df.columns:
        grouped = df.groupby("question_id", dropna=False)
        question_rows: list[dict[str, Any]] = []

        for question_id, subset in grouped:
            n = int(len(subset))
            supported = int(subset["claim_supported"].sum()) if "claim_supported" in subset else 0
            refuted = int(subset["claim_refuted"].sum()) if "claim_refuted" in subset else 0
            nei = int(subset["claim_nei"].sum()) if "claim_nei" in subset else 0
            manual = bool(subset["claim_needs_manual_review"].any()) if "claim_needs_manual_review" in subset else False

            question_rows.append(
                {
                    "question_id": clean_text(question_id),
                    "all_supported": bool(n > 0 and supported == n),
                    "any_refuted": bool(refuted > 0),
                    "any_nei": bool(nei > 0),
                    "manual_review": manual,
                    "support_rate": supported / n if n else 0.0,
                }
            )

        qdf = pd.DataFrame(question_rows)
        summary.update(
            {
                "question_all_claims_supported_rate": mean_bool(qdf["all_supported"]),
                "question_any_refuted_rate": mean_bool(qdf["any_refuted"]),
                "question_any_nei_rate": mean_bool(qdf["any_nei"]),
                "question_manual_review_rate": mean_bool(qdf["manual_review"]),
                "avg_question_support_rate": mean_numeric(qdf["support_rate"]),
            }
        )

    return summary


def summarize_answers(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "n": int(len(df)),
        "final_accuracy": mean_bool(df["eval_final_correct"]) if "eval_final_correct" in df.columns else None,
        "behavior_accuracy": mean_bool(df["eval_behavior_correct"]) if "eval_behavior_correct" in df.columns else None,
        "answer_accuracy": mean_bool(df["eval_answer_correct"]) if "eval_answer_correct" in df.columns else None,
        "avg_token_f1": mean_numeric(df["eval_token_f1"]) if "eval_token_f1" in df.columns else None,
        "error_type_counts": value_counts_dict(df, "eval_error_type"),
        "expected_behavior_counts": value_counts_dict(df, "eval_expected_final_behavior"),
        "s4_final_decision_counts": value_counts_dict(df, "s4_final_decision"),
        "manual_review_rate": mean_bool(df["eval_needs_manual_review"]) if "eval_needs_manual_review" in df.columns else None,
        "wrong_abstention_rate": mean_bool(df["eval_wrong_abstention"]) if "eval_wrong_abstention" in df.columns else None,
        "wrong_clarification_rate": mean_bool(df["eval_wrong_clarification"]) if "eval_wrong_clarification" in df.columns else None,
        "missing_abstention_rate": mean_bool(df["eval_missing_abstention"]) if "eval_missing_abstention" in df.columns else None,
        "missing_clarification_rate": mean_bool(df["eval_missing_clarification"]) if "eval_missing_clarification" in df.columns else None,
        "s4_changed_answer_rate": mean_bool(df["eval_s4_changed_answer"]) if "eval_s4_changed_answer" in df.columns else None,
        "s4_changed_behavior_rate": mean_bool(df["eval_s4_changed_behavior"]) if "eval_s4_changed_behavior" in df.columns else None,
        "all_claims_supported_rate": mean_bool(df["eval_all_claims_supported"]) if "eval_all_claims_supported" in df.columns else None,
        "any_claim_nei_rate": mean_bool(df["eval_any_claim_nei"]) if "eval_any_claim_nei" in df.columns else None,
        "any_claim_refuted_rate": mean_bool(df["eval_any_claim_refuted"]) if "eval_any_claim_refuted" in df.columns else None,
    }

    return summary


# ---------------------------------------------------------------------------
# Checks de consistencia
# ---------------------------------------------------------------------------

def verify_file_availability(
    *,
    parsed_path: Path,
    claims_path: Path,
    answers_path: Path,
    parsed_df: pd.DataFrame | None,
    claim_df: pd.DataFrame | None,
    answer_df: pd.DataFrame | None,
    checks: list[dict[str, Any]],
) -> None:
    add_check(
        checks,
        check_name="parsed_file_available",
        status="PASS" if parsed_df is not None else "FAIL",
        message="Archivo parseado S4 disponible." if parsed_df is not None else "Falta archivo parseado S4.",
        value=str(parsed_path),
    )
    add_check(
        checks,
        check_name="claim_results_available",
        status="PASS" if claim_df is not None else "WARN",
        message="Archivo claim-level disponible." if claim_df is not None else "No se encontró archivo claim-level; se omiten checks de claims.",
        value=str(claims_path),
    )
    add_check(
        checks,
        check_name="answer_results_available",
        status="PASS" if answer_df is not None else "WARN",
        message="Archivo answer-level disponible." if answer_df is not None else "No se encontró archivo answer-level; se omiten checks de respuestas.",
        value=str(answers_path),
    )


def verify_row_counts(
    parsed_df: pd.DataFrame | None,
    claim_df: pd.DataFrame | None,
    answer_df: pd.DataFrame | None,
    checks: list[dict[str, Any]],
    question_issues: list[dict[str, Any]],
) -> None:
    if parsed_df is None:
        return

    n_parsed = len(parsed_df)

    if "id" in parsed_df.columns:
        dup_count = int(parsed_df["id"].duplicated().sum())
        add_check(
            checks,
            check_name="duplicate_question_ids",
            status="PASS" if dup_count == 0 else "FAIL",
            message="No hay IDs duplicados." if dup_count == 0 else "Hay IDs duplicados en S4 parseado.",
            value=dup_count,
        )

        if dup_count > 0:
            duplicated = parsed_df[parsed_df["id"].duplicated(keep=False)]
            for _, row in duplicated.iterrows():
                add_issue(
                    question_issues,
                    level="critical",
                    source="parsed",
                    item_id=clean_text(row.get("id", "")),
                    category="duplicate_id",
                    message="ID duplicado en archivo parseado S4.",
                    suggested_action="Revisar generación/resume del runner S4.",
                )

    if answer_df is not None:
        n_answer = len(answer_df)
        add_check(
            checks,
            check_name="parsed_answer_row_count_match",
            status="PASS" if n_parsed == n_answer else "FAIL",
            message="La cantidad de filas parseadas y answer-level coincide."
            if n_parsed == n_answer
            else "La cantidad de filas parseadas y answer-level no coincide.",
            value=f"parsed={n_parsed}, answers={n_answer}",
        )

    if claim_df is not None and "question_id" in claim_df.columns:
        n_claim_questions = int(claim_df["question_id"].nunique())
        add_check(
            checks,
            check_name="parsed_claim_question_count_match",
            status="PASS" if n_claim_questions == n_parsed else "WARN",
            message="La cantidad de preguntas con claims coincide con S4 parseado."
            if n_claim_questions == n_parsed
            else "La cantidad de preguntas con claims no coincide exactamente con S4 parseado.",
            value=f"parsed={n_parsed}, claim_questions={n_claim_questions}",
        )


def verify_parsed_rows(
    parsed_df: pd.DataFrame | None,
    checks: list[dict[str, Any]],
    question_issues: list[dict[str, Any]],
    *,
    latency_warn_seconds: float,
    total_tokens_warn: int,
    chunks_warn: int,
    retrieval_rounds_warn: int,
    verification_rounds_warn: int,
) -> None:
    if parsed_df is None:
        return

    required_cols = ["id", "parsed_answer", "valid_answer_format", "s4_final_decision"]
    missing = [col for col in required_cols if col not in parsed_df.columns]
    add_check(
        checks,
        check_name="parsed_required_columns",
        status="PASS" if not missing else "FAIL",
        message="Columnas mínimas parseadas presentes." if not missing else "Faltan columnas mínimas en S4 parseado.",
        value=",".join(missing),
    )

    if missing:
        return

    for _, row in parsed_df.iterrows():
        qid = clean_text(row.get("id", ""))

        valid = coerce_bool(row.get("valid_answer_format"), default=False)
        if not valid:
            add_issue(
                question_issues,
                level="critical",
                source="parsed",
                item_id=qid,
                category="invalid_answer_format",
                message="Respuesta S4 con formato inválido.",
                detail=clean_text(row.get("parsed_answer", ""))[:300],
                suggested_action="Revisar raw_output y parse_s4_outputs.py.",
            )

        run_error = coerce_bool(row.get("run_error_present"), default=False)
        error_text = clean_text(row.get("error", ""))
        if run_error or error_text:
            add_issue(
                question_issues,
                level="critical",
                source="parsed",
                item_id=qid,
                category="run_error",
                message="S4 reportó error de ejecución.",
                detail=error_text,
                suggested_action="Revisar trace en raw_output/error.",
            )

        parse_error = clean_text(row.get("parse_error", ""))
        if parse_error:
            add_issue(
                question_issues,
                level="warning",
                source="parsed",
                item_id=qid,
                category="parse_error",
                message="Hay parse_error o parse warning.",
                detail=parse_error,
                suggested_action="Revisar parse_s4_outputs.py y raw_output.",
            )

        final_decision = normalize_label(row.get("s4_final_decision", ""))
        abstained = coerce_bool(row.get("s4_abstained"), default=False)
        if final_decision == "abstained" and not abstained:
            add_issue(
                question_issues,
                level="warning",
                source="parsed",
                item_id=qid,
                category="decision_abstention_inconsistency",
                message="final_decision es abstained pero s4_abstained no es True.",
                suggested_action="Revisar campos s4_final_decision/s4_abstained.",
            )

        n_claims = safe_int(row.get("s4_num_claims", 0))
        n_supported = safe_int(row.get("s4_num_supported_claims", 0))
        n_refuted = safe_int(row.get("s4_num_refuted_claims", 0))
        n_nei = safe_int(row.get("s4_num_nei_claims", 0))

        if n_claims > 0 and n_supported + n_refuted + n_nei > n_claims:
            add_issue(
                question_issues,
                level="warning",
                source="parsed",
                item_id=qid,
                category="claim_count_inconsistency",
                message="La suma de claims supported/refuted/NEI supera num_claims.",
                detail=f"num_claims={n_claims}, supported={n_supported}, refuted={n_refuted}, nei={n_nei}",
                suggested_action="Revisar parse_s4_outputs.py o controller S4.",
            )

        if final_decision == "unchanged" and (n_refuted > 0 or n_nei > 0):
            add_issue(
                question_issues,
                level="critical",
                source="parsed",
                item_id=qid,
                category="unchanged_with_unsupported_claims",
                message="S4 mantuvo la respuesta aunque hay claims refutados o NEI.",
                detail=f"refuted={n_refuted}, nei={n_nei}",
                suggested_action="Revisar fire_controller_s4.py / repair policy.",
            )

        latency = safe_float(row.get("s4_latency_seconds"), default=None)
        if latency is not None and latency > latency_warn_seconds:
            add_issue(
                question_issues,
                level="info",
                source="parsed",
                item_id=qid,
                category="high_latency",
                message="Latencia S4 alta.",
                detail=f"latency_seconds={latency:.3f}",
                suggested_action="Considerar reducir max_rounds/top_k o limitar S4 a casos retrieve.",
            )

        tokens = safe_int(row.get("s4_total_tokens", 0))
        if tokens > total_tokens_warn:
            add_issue(
                question_issues,
                level="info",
                source="parsed",
                item_id=qid,
                category="high_token_usage",
                message="Uso de tokens alto.",
                detail=f"s4_total_tokens={tokens}",
                suggested_action="Considerar estrategias rules/hybrid o top_k menor.",
            )

        chunks = safe_int(row.get("s4_num_chunks_retrieved_total", 0))
        if chunks > chunks_warn:
            add_issue(
                question_issues,
                level="info",
                source="parsed",
                item_id=qid,
                category="many_chunks_retrieved",
                message="S4 recuperó muchos chunks.",
                detail=f"chunks={chunks}",
                suggested_action="Revisar deduplicación/reuso de evidencia.",
            )

        retrieval_rounds = safe_int(row.get("s4_num_retrieval_rounds", 0))
        if retrieval_rounds > retrieval_rounds_warn:
            add_issue(
                question_issues,
                level="info",
                source="parsed",
                item_id=qid,
                category="many_retrieval_rounds",
                message="S4 usó muchas rondas de retrieval.",
                detail=f"retrieval_rounds={retrieval_rounds}",
                suggested_action="Revisar max_rounds_per_claim o política de stop.",
            )

        verification_rounds = safe_int(row.get("s4_num_verification_rounds", 0))
        if verification_rounds > verification_rounds_warn:
            add_issue(
                question_issues,
                level="info",
                source="parsed",
                item_id=qid,
                category="many_verification_rounds",
                message="S4 usó muchas rondas de verificación.",
                detail=f"verification_rounds={verification_rounds}",
                suggested_action="Revisar max_rounds_per_claim.",
            )


def verify_answer_rows(
    answer_df: pd.DataFrame | None,
    question_issues: list[dict[str, Any]],
) -> None:
    if answer_df is None:
        return

    for _, row in answer_df.iterrows():
        qid = clean_text(row.get("id", ""))

        final_correct = coerce_bool(row.get("eval_final_correct"), default=False)
        if not final_correct:
            add_issue(
                question_issues,
                level="critical",
                source="answer_eval",
                item_id=qid,
                category="final_answer_incorrect",
                message="La evaluación answer-level marcó la respuesta final como incorrecta.",
                detail=clean_text(row.get("eval_error_type", "")),
                suggested_action="Revisar respuesta final, gold answer y claims.",
            )

        needs_review = coerce_bool(row.get("eval_needs_manual_review"), default=False)
        if needs_review:
            add_issue(
                question_issues,
                level="warning",
                source="answer_eval",
                item_id=qid,
                category="answer_manual_review",
                message="La evaluación answer-level recomienda revisión manual.",
                detail=clean_text(row.get("eval_notes", "")),
                suggested_action="Inspeccionar CSV answer-level.",
            )

        wrong_abstention = coerce_bool(row.get("eval_wrong_abstention"), default=False)
        if wrong_abstention:
            add_issue(
                question_issues,
                level="critical",
                source="answer_eval",
                item_id=qid,
                category="wrong_abstention",
                message="S4 se abstuvo cuando se esperaba respuesta.",
                detail=clean_text(row.get("eval_notes", "")),
                suggested_action="Revisar evidencia recuperada y verifier.",
            )

        wrong_clarification = coerce_bool(row.get("eval_wrong_clarification"), default=False)
        if wrong_clarification:
            add_issue(
                question_issues,
                level="critical",
                source="answer_eval",
                item_id=qid,
                category="wrong_clarification",
                message="S4 pidió aclaración cuando se esperaba respuesta.",
                detail=clean_text(row.get("eval_notes", "")),
                suggested_action="Revisar expected_behavior y final answer.",
            )

        any_nei = coerce_bool(row.get("eval_any_claim_nei"), default=False)
        any_refuted = coerce_bool(row.get("eval_any_claim_refuted"), default=False)
        final_decision = normalize_label(row.get("s4_final_decision", ""))

        if final_decision == "unchanged" and (any_nei or any_refuted):
            add_issue(
                question_issues,
                level="critical",
                source="answer_eval",
                item_id=qid,
                category="unchanged_with_bad_claims",
                message="Answer-level detecta respuesta unchanged con claims NEI/refutados.",
                detail=f"any_nei={any_nei}, any_refuted={any_refuted}",
                suggested_action="Revisar política FIRE de reparación.",
            )


def verify_claim_rows(
    claim_df: pd.DataFrame | None,
    claim_issues: list[dict[str, Any]],
) -> None:
    if claim_df is None:
        return

    for _, row in claim_df.iterrows():
        claim_gid = clean_text(row.get("claim_global_id", ""))
        if not claim_gid:
            claim_gid = f"{clean_text(row.get('question_id', ''))}::{clean_text(row.get('claim_id', ''))}"

        claim_text = clean_text(row.get("claim_text", ""))
        if not claim_text and clean_text(row.get("claim_id", "")) != "no_claims":
            add_issue(
                claim_issues,
                level="warning",
                source="claim_eval",
                item_id=claim_gid,
                category="empty_claim_text",
                message="Claim sin texto.",
                suggested_action="Revisar extractor de claims.",
            )

        verdict = normalize_label(row.get("claim_verdict", ""))
        if verdict not in {"supported", "refuted", "not_enough_info"}:
            add_issue(
                claim_issues,
                level="critical",
                source="claim_eval",
                item_id=claim_gid,
                category="unknown_claim_verdict",
                message="Veredicto de claim desconocido.",
                detail=verdict,
                suggested_action="Revisar fire_verifier_s4.py.",
            )

        needs_review = coerce_bool(row.get("claim_needs_manual_review"), default=False)
        if needs_review:
            add_issue(
                claim_issues,
                level="warning",
                source="claim_eval",
                item_id=claim_gid,
                category="claim_manual_review",
                message="Claim marcado para revisión manual.",
                detail=clean_text(row.get("claim_manual_review_reason", "")),
                suggested_action="Inspeccionar evidencia y rationale del claim.",
            )

        requires_evidence = coerce_bool(row.get("claim_requires_evidence"), default=False)
        supported = coerce_bool(row.get("claim_supported"), default=False)
        refuted = coerce_bool(row.get("claim_refuted"), default=False)
        nei = coerce_bool(row.get("claim_nei"), default=False)

        support_ids = clean_text(row.get("claim_supporting_chunk_ids", ""))
        refuting_ids = clean_text(row.get("claim_refuting_chunk_ids", ""))

        if requires_evidence and supported and not support_ids:
            add_issue(
                claim_issues,
                level="critical",
                source="claim_eval",
                item_id=claim_gid,
                category="supported_without_evidence",
                message="Claim supported sin chunk soporte.",
                suggested_action="Revisar output del verifier.",
            )

        if requires_evidence and refuted and not refuting_ids:
            add_issue(
                claim_issues,
                level="critical",
                source="claim_eval",
                item_id=claim_gid,
                category="refuted_without_evidence",
                message="Claim refuted sin chunk refutador.",
                suggested_action="Revisar output del verifier.",
            )

        if requires_evidence and nei:
            add_issue(
                claim_issues,
                level="warning",
                source="claim_eval",
                item_id=claim_gid,
                category="claim_nei",
                message="Claim requiere evidencia pero quedó not_enough_info.",
                detail=claim_text[:300],
                suggested_action="Revisar retrieval/verifier o aceptar abstención.",
            )


def verify_cross_consistency(
    parsed_df: pd.DataFrame | None,
    claim_df: pd.DataFrame | None,
    answer_df: pd.DataFrame | None,
    checks: list[dict[str, Any]],
    question_issues: list[dict[str, Any]],
) -> None:
    if parsed_df is None:
        return

    if claim_df is not None and "question_id" in claim_df.columns and "id" in parsed_df.columns:
        claim_counts = claim_df.groupby("question_id", dropna=False).agg(
            claim_count=("claim_id", "count"),
            supported=("claim_supported", lambda s: int(s.astype(bool).sum()) if len(s) else 0),
            refuted=("claim_refuted", lambda s: int(s.astype(bool).sum()) if len(s) else 0),
            nei=("claim_nei", lambda s: int(s.astype(bool).sum()) if len(s) else 0),
        ).reset_index()

        merged = parsed_df.merge(claim_counts, left_on="id", right_on="question_id", how="left")

        mismatches = 0
        for _, row in merged.iterrows():
            qid = clean_text(row.get("id", ""))
            if is_missing(row.get("claim_count")):
                continue

            expected_claims = safe_int(row.get("s4_num_claims", 0))
            actual_claims = safe_int(row.get("claim_count", 0))
            expected_supported = safe_int(row.get("s4_num_supported_claims", 0))
            actual_supported = safe_int(row.get("supported", 0))
            expected_refuted = safe_int(row.get("s4_num_refuted_claims", 0))
            actual_refuted = safe_int(row.get("refuted", 0))
            expected_nei = safe_int(row.get("s4_num_nei_claims", 0))
            actual_nei = safe_int(row.get("nei", 0))

            if (
                expected_claims != actual_claims
                or expected_supported != actual_supported
                or expected_refuted != actual_refuted
                or expected_nei != actual_nei
            ):
                mismatches += 1
                add_issue(
                    question_issues,
                    level="warning",
                    source="cross_check",
                    item_id=qid,
                    category="parsed_claim_eval_mismatch",
                    message="Conteos parseados de claims no coinciden con claim-level CSV.",
                    detail=(
                        f"parsed claims/support/refuted/nei="
                        f"{expected_claims}/{expected_supported}/{expected_refuted}/{expected_nei}; "
                        f"claim_eval={actual_claims}/{actual_supported}/{actual_refuted}/{actual_nei}"
                    ),
                    suggested_action="Re-ejecutar parse_s4_outputs.py y evaluate_s4_claims.py.",
                )

        add_check(
            checks,
            check_name="parsed_vs_claim_counts_match",
            status="PASS" if mismatches == 0 else "WARN",
            message="Conteos de claims entre parsed y claim-level coinciden."
            if mismatches == 0
            else "Hay diferencias entre conteos parsed y claim-level.",
            value=mismatches,
        )

    if answer_df is not None and "id" in parsed_df.columns and "id" in answer_df.columns:
        common = set(parsed_df["id"].astype(str)) & set(answer_df["id"].astype(str))
        add_check(
            checks,
            check_name="parsed_answer_id_overlap",
            status="PASS" if len(common) == len(parsed_df) == len(answer_df) else "WARN",
            message="IDs de parsed y answer-level coinciden."
            if len(common) == len(parsed_df) == len(answer_df)
            else "IDs de parsed y answer-level no coinciden perfectamente.",
            value=f"overlap={len(common)}, parsed={len(parsed_df)}, answers={len(answer_df)}",
        )


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------

def compute_status(
    question_issues: list[dict[str, Any]],
    claim_issues: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> str:
    all_issues = question_issues + claim_issues

    if any(item["status"] == "FAIL" for item in checks):
        return "FAIL"
    if any(issue["level"] == "critical" for issue in all_issues):
        return "FAIL"
    if any(item["status"] == "WARN" for item in checks):
        return "PASS_WITH_WARNINGS"
    if any(issue["level"] == "warning" for issue in all_issues):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def build_summary(
    *,
    parsed_path: Path,
    claims_path: Path,
    answers_path: Path,
    parsed_df: pd.DataFrame | None,
    claim_df: pd.DataFrame | None,
    answer_df: pd.DataFrame | None,
    question_issues: list[dict[str, Any]],
    claim_issues: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    status = compute_status(question_issues, claim_issues, checks)

    all_issues = question_issues + claim_issues
    issue_counts_by_level: dict[str, int] = {}
    issue_counts_by_category: dict[str, int] = {}

    for issue in all_issues:
        level = clean_text(issue.get("level", ""))
        category = clean_text(issue.get("category", ""))
        issue_counts_by_level[level] = issue_counts_by_level.get(level, 0) + 1
        issue_counts_by_category[category] = issue_counts_by_category.get(category, 0) + 1

    check_counts = {}
    for check in checks:
        status_key = clean_text(check.get("status", ""))
        check_counts[status_key] = check_counts.get(status_key, 0) + 1

    return {
        "overall_status": status,
        "inputs": {
            "s4_parsed_path": str(parsed_path),
            "claim_results_path": str(claims_path),
            "answer_results_path": str(answers_path),
        },
        "parsed_summary": summarize_parsed(parsed_df),
        "claim_summary": summarize_claims(claim_df),
        "answer_summary": summarize_answers(answer_df),
        "verification": {
            "num_checks": len(checks),
            "check_counts": check_counts,
            "num_question_issues": len(question_issues),
            "num_claim_issues": len(claim_issues),
            "num_total_issues": len(all_issues),
            "issue_counts_by_level": issue_counts_by_level,
            "issue_counts_by_category": issue_counts_by_category,
        },
    }


def make_text_report(summary: dict[str, Any], question_issues: pd.DataFrame, claim_issues: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("Verificación final S4")
    lines.append("=====================")
    lines.append("")
    lines.append(f"Estado general: {summary.get('overall_status', 'UNKNOWN')}")
    lines.append("")

    parsed = summary.get("parsed_summary", {})
    claims = summary.get("claim_summary", {})
    answers = summary.get("answer_summary", {})
    verification = summary.get("verification", {})

    lines.append("Resumen parseado")
    lines.append("----------------")
    if parsed.get("available"):
        lines.append(f"Filas S4 parseadas: {parsed.get('n')}")
        lines.append(f"Valid answer format rate: {fmt(parsed.get('valid_answer_format_rate'))}")
        lines.append(f"Run error rate: {fmt(parsed.get('run_error_rate'))}")
        lines.append(f"S4 abstention rate: {fmt(parsed.get('s4_abstention_rate'))}")
        lines.append(f"S4 correction rate: {fmt(parsed.get('s4_correction_rate'))}")
        lines.append(f"Avg claims: {fmt(parsed.get('avg_claims'))}")
        lines.append(f"Avg supported claims: {fmt(parsed.get('avg_supported_claims'))}")
        lines.append(f"Avg NEI claims: {fmt(parsed.get('avg_nei_claims'))}")
        lines.append(f"Avg latency seconds: {fmt(parsed.get('avg_latency_seconds'))}")
        lines.append(f"Avg total tokens: {fmt(parsed.get('avg_total_tokens'))}")
        final_decisions = parsed.get("final_decision_counts", {})
        if final_decisions:
            lines.append(f"Final decisions: {final_decisions}")
    else:
        lines.append("Archivo parseado no disponible.")
    lines.append("")

    lines.append("Resumen claim-level")
    lines.append("-------------------")
    if claims.get("available"):
        lines.append(f"Claims evaluados: {claims.get('n_claims')}")
        lines.append(f"Preguntas con claims: {claims.get('n_questions')}")
        lines.append(f"Supported rate: {fmt(claims.get('supported_rate'))}")
        lines.append(f"Refuted rate: {fmt(claims.get('refuted_rate'))}")
        lines.append(f"NEI rate: {fmt(claims.get('nei_rate'))}")
        lines.append(f"Manual review rate: {fmt(claims.get('manual_review_rate'))}")
        lines.append(f"Uses gold evidence rate: {fmt(claims.get('uses_gold_evidence_rate'))}")
        lines.append(f"Question all claims supported rate: {fmt(claims.get('question_all_claims_supported_rate'))}")
    else:
        lines.append("Archivo claim-level no disponible.")
    lines.append("")

    lines.append("Resumen answer-level")
    lines.append("--------------------")
    if answers.get("available"):
        lines.append(f"Filas evaluadas: {answers.get('n')}")
        lines.append(f"Final accuracy: {fmt(answers.get('final_accuracy'))}")
        lines.append(f"Behavior accuracy: {fmt(answers.get('behavior_accuracy'))}")
        lines.append(f"Answer accuracy: {fmt(answers.get('answer_accuracy'))}")
        lines.append(f"Manual review rate: {fmt(answers.get('manual_review_rate'))}")
        lines.append(f"Wrong abstention rate: {fmt(answers.get('wrong_abstention_rate'))}")
        lines.append(f"Any claim NEI rate: {fmt(answers.get('any_claim_nei_rate'))}")
        lines.append(f"Any claim refuted rate: {fmt(answers.get('any_claim_refuted_rate'))}")
        error_counts = answers.get("error_type_counts", {})
        if error_counts:
            lines.append(f"Error/result types: {error_counts}")
    else:
        lines.append("Archivo answer-level no disponible.")
    lines.append("")

    lines.append("Problemas detectados")
    lines.append("--------------------")
    lines.append(f"Checks: {verification.get('check_counts', {})}")
    lines.append(f"Issues totales: {verification.get('num_total_issues', 0)}")
    lines.append(f"Issues por nivel: {verification.get('issue_counts_by_level', {})}")
    lines.append(f"Issues por categoría: {verification.get('issue_counts_by_category', {})}")
    lines.append("")

    if not question_issues.empty:
        lines.append("Primeros problemas por pregunta:")
        for _, row in question_issues.head(10).iterrows():
            lines.append(
                f"- [{row.get('level')}] {row.get('id')} :: {row.get('category')} :: {row.get('message')}"
            )
        lines.append("")

    if not claim_issues.empty:
        lines.append("Primeros problemas por claim:")
        for _, row in claim_issues.head(10).iterrows():
            lines.append(
                f"- [{row.get('level')}] {row.get('id')} :: {row.get('category')} :: {row.get('message')}"
            )
        lines.append("")

    lines.append("Lectura recomendada")
    lines.append("-------------------")
    status = summary.get("overall_status")
    if status == "PASS":
        lines.append("El pipeline S4 luce consistente para estos archivos. Se puede avanzar a comparar S2 vs S4.")
    elif status == "PASS_WITH_WARNINGS":
        lines.append("El pipeline S4 corre, pero hay advertencias. Revisar CSVs de problemas antes de usarlo para conclusiones.")
    else:
        lines.append("Hay problemas críticos. Conviene corregirlos antes de comparar modelos.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def verify_results(
    *,
    s4_parsed_path: Path,
    claim_results_path: Path,
    answer_results_path: Path,
    output_dir: Path,
    prefix: str,
    latency_warn_seconds: float,
    total_tokens_warn: int,
    chunks_warn: int,
    retrieval_rounds_warn: int,
    verification_rounds_warn: int,
) -> dict[str, Path]:
    parsed_df = read_csv_if_exists(s4_parsed_path, required=True)
    claim_df = read_csv_if_exists(claim_results_path, required=False)
    answer_df = read_csv_if_exists(answer_results_path, required=False)

    checks: list[dict[str, Any]] = []
    question_issues: list[dict[str, Any]] = []
    claim_issues: list[dict[str, Any]] = []

    verify_file_availability(
        parsed_path=s4_parsed_path,
        claims_path=claim_results_path,
        answers_path=answer_results_path,
        parsed_df=parsed_df,
        claim_df=claim_df,
        answer_df=answer_df,
        checks=checks,
    )
    verify_row_counts(parsed_df, claim_df, answer_df, checks, question_issues)
    verify_parsed_rows(
        parsed_df,
        checks,
        question_issues,
        latency_warn_seconds=latency_warn_seconds,
        total_tokens_warn=total_tokens_warn,
        chunks_warn=chunks_warn,
        retrieval_rounds_warn=retrieval_rounds_warn,
        verification_rounds_warn=verification_rounds_warn,
    )
    verify_answer_rows(answer_df, question_issues)
    verify_claim_rows(claim_df, claim_issues)
    verify_cross_consistency(parsed_df, claim_df, answer_df, checks, question_issues)

    summary = build_summary(
        parsed_path=s4_parsed_path,
        claims_path=claim_results_path,
        answers_path=answer_results_path,
        parsed_df=parsed_df,
        claim_df=claim_df,
        answer_df=answer_df,
        question_issues=question_issues,
        claim_issues=claim_issues,
        checks=checks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / f"{prefix}_summary.json"
    summary_txt_path = output_dir / f"{prefix}_summary.txt"
    question_issues_path = output_dir / f"{prefix}_problematic_questions.csv"
    claim_issues_path = output_dir / f"{prefix}_problematic_claims.csv"
    sanity_checks_path = output_dir / f"{prefix}_sanity_checks.csv"

    question_issues_df = pd.DataFrame(question_issues)
    claim_issues_df = pd.DataFrame(claim_issues)
    checks_df = pd.DataFrame(checks)

    # Asegurar columnas aunque no haya problemas.
    issue_cols = ["level", "source", "id", "category", "message", "detail", "suggested_action"]
    check_cols = ["check_name", "status", "message", "value", "threshold"]

    if question_issues_df.empty:
        question_issues_df = pd.DataFrame(columns=issue_cols)
    if claim_issues_df.empty:
        claim_issues_df = pd.DataFrame(columns=issue_cols)
    if checks_df.empty:
        checks_df = pd.DataFrame(columns=check_cols)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_txt_path.write_text(
        make_text_report(summary, question_issues_df, claim_issues_df),
        encoding="utf-8",
    )

    question_issues_df.to_csv(question_issues_path, index=False)
    claim_issues_df.to_csv(claim_issues_path, index=False)
    checks_df.to_csv(sanity_checks_path, index=False)

    return {
        "summary_json": summary_json_path,
        "summary_txt": summary_txt_path,
        "problematic_questions": question_issues_path,
        "problematic_claims": claim_issues_path,
        "sanity_checks": sanity_checks_path,
    }


def print_final(paths: dict[str, Path]) -> None:
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    print("\nVerificación final S4")
    print("=====================")
    print(f"Estado general: {summary.get('overall_status')}")
    print()

    verification = summary.get("verification", {})
    print(f"Checks: {verification.get('check_counts', {})}")
    print(f"Issues totales: {verification.get('num_total_issues', 0)}")
    print(f"Issues por nivel: {verification.get('issue_counts_by_level', {})}")
    print()

    answers = summary.get("answer_summary", {})
    if answers.get("available"):
        print("Answer-level:")
        print(f"- Final accuracy: {fmt(answers.get('final_accuracy'))}")
        print(f"- Behavior accuracy: {fmt(answers.get('behavior_accuracy'))}")
        print(f"- Manual review rate: {fmt(answers.get('manual_review_rate'))}")

    claims = summary.get("claim_summary", {})
    if claims.get("available"):
        print("Claim-level:")
        print(f"- Supported rate: {fmt(claims.get('supported_rate'))}")
        print(f"- NEI rate: {fmt(claims.get('nei_rate'))}")
        print(f"- Refuted rate: {fmt(claims.get('refuted_rate'))}")

    print()
    print(f"Resumen JSON: {paths['summary_json']}")
    print(f"Resumen TXT: {paths['summary_txt']}")
    print(f"Problemas por pregunta: {paths['problematic_questions']}")
    print(f"Problemas por claim: {paths['problematic_claims']}")
    print(f"Sanity checks: {paths['sanity_checks']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifica outputs finales S4 y genera reportes de sanity-check."
    )

    parser.add_argument(
        "--s4-parsed-path",
        type=Path,
        default=DEFAULT_S4_PARSED_PATH,
        help="CSV parseado de S4.",
    )
    parser.add_argument(
        "--claim-results-path",
        type=Path,
        default=DEFAULT_CLAIM_RESULTS_PATH,
        help="CSV claim-level generado por evaluate_s4_claims.py.",
    )
    parser.add_argument(
        "--answer-results-path",
        type=Path,
        default=DEFAULT_ANSWER_RESULTS_PATH,
        help="CSV answer-level generado por evaluate_s4_answers.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para reportes de verificación.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Prefijo de archivos de salida.",
    )

    parser.add_argument("--latency-warn-seconds", type=float, default=DEFAULT_LATENCY_WARN_SECONDS)
    parser.add_argument("--total-tokens-warn", type=int, default=DEFAULT_TOTAL_TOKENS_WARN)
    parser.add_argument("--chunks-warn", type=int, default=DEFAULT_CHUNKS_WARN)
    parser.add_argument("--retrieval-rounds-warn", type=int, default=DEFAULT_RETRIEVAL_ROUNDS_WARN)
    parser.add_argument("--verification-rounds-warn", type=int, default=DEFAULT_VERIFICATION_ROUNDS_WARN)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    paths = verify_results(
        s4_parsed_path=args.s4_parsed_path,
        claim_results_path=args.claim_results_path,
        answer_results_path=args.answer_results_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
        latency_warn_seconds=args.latency_warn_seconds,
        total_tokens_warn=args.total_tokens_warn,
        chunks_warn=args.chunks_warn,
        retrieval_rounds_warn=args.retrieval_rounds_warn,
        verification_rounds_warn=args.verification_rounds_warn,
    )

    print_final(paths)


if __name__ == "__main__":
    main()

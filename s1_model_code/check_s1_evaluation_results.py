#!/usr/bin/env python3
"""
check_s1_evaluation_results.py

Verifica y resume los resultados ya parseados/evaluados del pipeline S1 RAG.

No llama al LLM.
No modifica los outputs originales.

Entradas por defecto:
    outputs/s1/generation/hotpotqa_mini_s1_parsed.csv
    outputs/s1/evaluation/hotpotqa_mini_s1_answer_results.csv
    outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary.json   # opcional

Salidas por defecto:
    verificacion_de_resultados/s1_verificacion_resumen.txt
    verificacion_de_resultados/s1_verificacion_resumen.json
    verificacion_de_resultados/s1_casos_problematicos.csv
    verificacion_de_resultados/s1_muestra_respuestas.csv
    verificacion_de_resultados/s1_metricas_por_grupo.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PARSED_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_parsed.csv")
DEFAULT_EVALUATED_PATH = Path("outputs/s1/evaluation/hotpotqa_mini_s1_answer_results.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/s1/evaluation/hotpotqa_mini_s1_answer_summary.json")
DEFAULT_OUTPUT_DIR = Path("verificacion_de_resultados")


PARSED_EXPECTED_COLUMNS = [
    "id",
    "raw_output",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "valid_format",
    "parse_method",
    "parse_error",
]

EVALUATED_EXPECTED_COLUMNS = [
    "id",
    "original_question",
    "expected_answer",
    "parsed_answer",
    "parsed_abstained",
    "valid_format",
    "eval_answer_correct",
    "eval_exact_match",
    "eval_contains_gold_answer",
    "eval_token_f1",
    "eval_abstained",
    "eval_wrong_abstention",
    "eval_retrieval_hit_gold",
    "eval_retrieval_full_recall",
    "eval_retrieval_recall",
    "eval_error_type",
]

PROBLEM_PRIORITY_COLUMNS = [
    "id",
    "topic",
    "hotpot_type",
    "level",
    "original_question",
    "expected_answer",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "valid_format",
    "eval_answer_correct",
    "eval_exact_match",
    "eval_contains_gold_answer",
    "eval_token_f1",
    "eval_wrong_abstention",
    "eval_retrieval_hit_gold",
    "eval_retrieval_full_recall",
    "eval_retrieval_recall",
    "eval_error_type",
    "retrieved_titles",
    "retrieved_chunk_ids",
    "gold_evidence_ids",
    "matched_gold_chunk_ids",
    "parse_error",
    "error",
    "raw_output",
]

SAMPLE_PRIORITY_COLUMNS = [
    "id",
    "topic",
    "hotpot_type",
    "original_question",
    "expected_answer",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "eval_answer_correct",
    "eval_token_f1",
    "eval_error_type",
    "retrieved_titles",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convierte una serie con bools/strings/números a bool, preservando NaN."""
    def convert(value: Any) -> bool | float:
        if value is None:
            return math.nan
        try:
            if pd.isna(value):
                return math.nan
        except TypeError:
            pass

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
            return math.nan

        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1", "si", "sí"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return math.nan

    return series.apply(convert)


def bool_rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    converted = coerce_bool_series(df[col]).dropna()
    if converted.empty:
        return None
    return float(converted.astype(bool).mean())


def numeric_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def numeric_min(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def numeric_max(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def count_true(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    converted = coerce_bool_series(df[col]).dropna()
    if converted.empty:
        return 0
    return int(converted.astype(bool).sum())


def count_false(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    converted = coerce_bool_series(df[col]).dropna()
    if converted.empty:
        return 0
    return int((~converted.astype(bool)).sum())


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def value_counts_dict(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].fillna("<NA>").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def safe_round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, digits)
    return value


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------


def build_problem_mask(df: pd.DataFrame) -> pd.Series:
    """Marca filas que conviene inspeccionar manualmente."""
    problem = pd.Series(False, index=df.index)

    if "error" in df.columns:
        problem |= df["error"].fillna("").astype(str).str.strip().ne("")

    if "run_error_present" in df.columns:
        problem |= coerce_bool_series(df["run_error_present"]).fillna(False).astype(bool)

    if "valid_format" in df.columns:
        valid = coerce_bool_series(df["valid_format"]).fillna(False).astype(bool)
        problem |= ~valid

    if "eval_answer_correct" in df.columns:
        correct = coerce_bool_series(df["eval_answer_correct"])
        # Se consideran problemáticas solo las filas decididas como incorrectas.
        problem |= correct.eq(False).fillna(False)

    if "eval_wrong_abstention" in df.columns:
        problem |= coerce_bool_series(df["eval_wrong_abstention"]).fillna(False).astype(bool)

    if "eval_needs_manual_review" in df.columns:
        problem |= coerce_bool_series(df["eval_needs_manual_review"]).fillna(False).astype(bool)

    if "eval_retrieval_hit_gold" in df.columns:
        hit = coerce_bool_series(df["eval_retrieval_hit_gold"]).fillna(True).astype(bool)
        problem |= ~hit

    return problem


def summarize_overall(parsed_df: pd.DataFrame, evaluated_df: pd.DataFrame, problem_df: pd.DataFrame) -> dict[str, Any]:
    """Resumen global de verificación."""
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_parsed_rows": int(len(parsed_df)),
        "n_evaluated_rows": int(len(evaluated_df)),
        "ids_match_between_parsed_and_evaluated": None,
        "n_problem_rows": int(len(problem_df)),
        "problem_rate": float(len(problem_df) / len(evaluated_df)) if len(evaluated_df) else None,
        "missing_columns": {
            "parsed": missing_columns(parsed_df, PARSED_EXPECTED_COLUMNS),
            "evaluated": missing_columns(evaluated_df, EVALUATED_EXPECTED_COLUMNS),
        },
        "parse": {
            "valid_format_rate": bool_rate(evaluated_df, "valid_format"),
            "valid_format_count": count_true(evaluated_df, "valid_format"),
            "invalid_format_count": count_false(evaluated_df, "valid_format"),
            "parse_method_counts": value_counts_dict(evaluated_df, "parse_method"),
            "run_error_rate": bool_rate(evaluated_df, "run_error_present"),
            "run_error_count": count_true(evaluated_df, "run_error_present"),
        },
        "answer_quality": {
            "answer_accuracy": bool_rate(evaluated_df, "eval_answer_correct"),
            "answer_correct_count": count_true(evaluated_df, "eval_answer_correct"),
            "answer_incorrect_count": count_false(evaluated_df, "eval_answer_correct"),
            "exact_match_rate": bool_rate(evaluated_df, "eval_exact_match"),
            "contains_gold_answer_rate": bool_rate(evaluated_df, "eval_contains_gold_answer"),
            "gold_contains_answer_rate": bool_rate(evaluated_df, "eval_gold_contains_answer"),
            "avg_token_f1": numeric_mean(evaluated_df, "eval_token_f1"),
            "min_token_f1": numeric_min(evaluated_df, "eval_token_f1"),
            "max_token_f1": numeric_max(evaluated_df, "eval_token_f1"),
            "avg_confidence": numeric_mean(evaluated_df, "parsed_confidence"),
        },
        "abstention": {
            "abstention_rate": bool_rate(evaluated_df, "eval_abstained"),
            "abstention_count": count_true(evaluated_df, "eval_abstained"),
            "wrong_abstention_rate": bool_rate(evaluated_df, "eval_wrong_abstention"),
            "wrong_abstention_count": count_true(evaluated_df, "eval_wrong_abstention"),
        },
        "retrieval": {
            "retrieval_hit_gold_rate": bool_rate(evaluated_df, "eval_retrieval_hit_gold"),
            "retrieval_full_recall_rate": bool_rate(evaluated_df, "eval_retrieval_full_recall"),
            "avg_retrieval_recall": numeric_mean(evaluated_df, "eval_retrieval_recall"),
            "avg_retrieval_precision": numeric_mean(evaluated_df, "eval_retrieval_precision"),
        },
        "execution_cost": {
            "avg_latency_seconds": numeric_mean(evaluated_df, "latency_seconds"),
            "max_latency_seconds": numeric_max(evaluated_df, "latency_seconds"),
            "avg_input_tokens": numeric_mean(evaluated_df, "input_tokens"),
            "avg_output_tokens": numeric_mean(evaluated_df, "output_tokens"),
            "avg_total_tokens": numeric_mean(evaluated_df, "total_tokens"),
            "max_total_tokens": numeric_max(evaluated_df, "total_tokens"),
            "avg_rag_prompt_chars": numeric_mean(evaluated_df, "rag_prompt_chars"),
            "max_rag_prompt_chars": numeric_max(evaluated_df, "rag_prompt_chars"),
        },
        "counts": {
            "eval_error_type_counts": value_counts_dict(evaluated_df, "eval_error_type"),
            "topic_counts": value_counts_dict(evaluated_df, "topic"),
            "hotpot_type_counts": value_counts_dict(evaluated_df, "hotpot_type"),
            "top_k_counts": value_counts_dict(evaluated_df, "top_k"),
        },
    }

    if "id" in parsed_df.columns and "id" in evaluated_df.columns:
        parsed_ids = set(parsed_df["id"].astype(str))
        evaluated_ids = set(evaluated_df["id"].astype(str))
        summary["ids_match_between_parsed_and_evaluated"] = parsed_ids == evaluated_ids
        summary["ids_only_in_parsed"] = sorted(parsed_ids - evaluated_ids)
        summary["ids_only_in_evaluated"] = sorted(evaluated_ids - parsed_ids)

    return summary


def build_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Construye métricas por grupos útiles para análisis posterior."""
    group_columns = []
    for col in ["topic", "hotpot_type", "level", "eval_error_type", "top_k"]:
        if col in df.columns:
            group_columns.append(col)

    rows: list[dict[str, Any]] = []

    def summarize_group(group_type: str, group_value: str, subset: pd.DataFrame) -> dict[str, Any]:
        return {
            "group_type": group_type,
            "group": group_value,
            "n": int(len(subset)),
            "answer_accuracy": bool_rate(subset, "eval_answer_correct"),
            "exact_match_rate": bool_rate(subset, "eval_exact_match"),
            "contains_gold_answer_rate": bool_rate(subset, "eval_contains_gold_answer"),
            "avg_token_f1": numeric_mean(subset, "eval_token_f1"),
            "abstention_rate": bool_rate(subset, "eval_abstained"),
            "wrong_abstention_rate": bool_rate(subset, "eval_wrong_abstention"),
            "retrieval_hit_gold_rate": bool_rate(subset, "eval_retrieval_hit_gold"),
            "retrieval_full_recall_rate": bool_rate(subset, "eval_retrieval_full_recall"),
            "avg_retrieval_recall": numeric_mean(subset, "eval_retrieval_recall"),
            "avg_total_tokens": numeric_mean(subset, "total_tokens"),
            "avg_latency_seconds": numeric_mean(subset, "latency_seconds"),
        }

    rows.append(summarize_group("overall", "all", df))

    for col in group_columns:
        for value, subset in df.groupby(col, dropna=False):
            rows.append(summarize_group(col, clean_text(value) or "<NA>", subset))

    return pd.DataFrame(rows)


def load_optional_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_text_report(
    *,
    report_path: Path,
    parsed_path: Path,
    evaluated_path: Path,
    summary_path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    problem_df: pd.DataFrame,
    group_df: pd.DataFrame,
    optional_eval_summary: dict[str, Any] | None,
) -> None:
    lines: list[str] = []

    lines.append("VERIFICACIÓN DE RESULTADOS S1")
    lines.append("============================")
    lines.append(f"Fecha UTC: {summary['created_at_utc']}")
    lines.append("")
    lines.append("Archivos de entrada")
    lines.append("-------------------")
    lines.append(f"Parsed:    {parsed_path}")
    lines.append(f"Evaluated: {evaluated_path}")
    lines.append(f"Summary:   {summary_path} ({'existe' if summary_path.exists() else 'no encontrado'})")
    lines.append("")

    lines.append("Chequeo estructural")
    lines.append("-------------------")
    lines.append(f"Filas parsed: {summary['n_parsed_rows']}")
    lines.append(f"Filas evaluated: {summary['n_evaluated_rows']}")
    lines.append(f"IDs coinciden entre parsed/evaluated: {summary['ids_match_between_parsed_and_evaluated']}")
    lines.append(f"Columnas faltantes parsed: {summary['missing_columns']['parsed']}")
    lines.append(f"Columnas faltantes evaluated: {summary['missing_columns']['evaluated']}")
    lines.append("")

    lines.append("Parseo / formato")
    lines.append("----------------")
    parse = summary["parse"]
    lines.append(f"Valid format rate: {format_metric(parse['valid_format_rate'])}")
    lines.append(f"Valid format count: {parse['valid_format_count']}")
    lines.append(f"Invalid format count: {parse['invalid_format_count']}")
    lines.append(f"Run error rate: {format_metric(parse['run_error_rate'])}")
    lines.append(f"Run error count: {parse['run_error_count']}")
    lines.append(f"Parse method counts: {json.dumps(parse['parse_method_counts'], ensure_ascii=False)}")
    lines.append("")

    lines.append("Calidad de respuesta")
    lines.append("--------------------")
    aq = summary["answer_quality"]
    lines.append(f"Answer accuracy: {format_metric(aq['answer_accuracy'])}")
    lines.append(f"Answer correct count: {aq['answer_correct_count']}")
    lines.append(f"Answer incorrect count: {aq['answer_incorrect_count']}")
    lines.append(f"Exact match rate: {format_metric(aq['exact_match_rate'])}")
    lines.append(f"Contains gold answer rate: {format_metric(aq['contains_gold_answer_rate'])}")
    lines.append(f"Gold contains answer rate: {format_metric(aq['gold_contains_answer_rate'])}")
    lines.append(f"Avg token F1: {format_metric(aq['avg_token_f1'])}")
    lines.append(f"Min token F1: {format_metric(aq['min_token_f1'])}")
    lines.append(f"Max token F1: {format_metric(aq['max_token_f1'])}")
    lines.append(f"Avg confidence: {format_metric(aq['avg_confidence'])}")
    lines.append("")

    lines.append("Abstención")
    lines.append("----------")
    abst = summary["abstention"]
    lines.append(f"Abstention rate: {format_metric(abst['abstention_rate'])}")
    lines.append(f"Abstention count: {abst['abstention_count']}")
    lines.append(f"Wrong abstention rate: {format_metric(abst['wrong_abstention_rate'])}")
    lines.append(f"Wrong abstention count: {abst['wrong_abstention_count']}")
    lines.append("")

    lines.append("Retrieval asociado a las respuestas")
    lines.append("-----------------------------------")
    ret = summary["retrieval"]
    lines.append(f"Retrieval hit gold rate: {format_metric(ret['retrieval_hit_gold_rate'])}")
    lines.append(f"Retrieval full recall rate: {format_metric(ret['retrieval_full_recall_rate'])}")
    lines.append(f"Avg retrieval recall: {format_metric(ret['avg_retrieval_recall'])}")
    lines.append(f"Avg retrieval precision: {format_metric(ret['avg_retrieval_precision'])}")
    lines.append("")

    lines.append("Costo / latencia")
    lines.append("----------------")
    cost = summary["execution_cost"]
    lines.append(f"Avg latency seconds: {format_metric(cost['avg_latency_seconds'])}")
    lines.append(f"Max latency seconds: {format_metric(cost['max_latency_seconds'])}")
    lines.append(f"Avg input tokens: {format_metric(cost['avg_input_tokens'])}")
    lines.append(f"Avg output tokens: {format_metric(cost['avg_output_tokens'])}")
    lines.append(f"Avg total tokens: {format_metric(cost['avg_total_tokens'])}")
    lines.append(f"Max total tokens: {format_metric(cost['max_total_tokens'])}")
    lines.append(f"Avg RAG prompt chars: {format_metric(cost['avg_rag_prompt_chars'])}")
    lines.append(f"Max RAG prompt chars: {format_metric(cost['max_rag_prompt_chars'])}")
    lines.append("")

    lines.append("Distribuciones importantes")
    lines.append("--------------------------")
    counts = summary["counts"]
    lines.append("eval_error_type_counts:")
    for key, value in counts["eval_error_type_counts"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("topic_counts:")
    for key, value in counts["topic_counts"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("hotpot_type_counts:")
    for key, value in counts["hotpot_type_counts"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("top_k_counts:")
    for key, value in counts["top_k_counts"].items():
        lines.append(f"  - {key}: {value}")
    lines.append("")

    lines.append("Casos problemáticos")
    lines.append("-------------------")
    lines.append(f"Cantidad: {summary['n_problem_rows']}")
    lines.append(f"Problem rate: {format_metric(summary['problem_rate'])}")
    if len(problem_df) > 0:
        preview_cols = existing_columns(problem_df, [
            "id",
            "expected_answer",
            "parsed_answer",
            "eval_answer_correct",
            "eval_token_f1",
            "eval_retrieval_recall",
            "eval_error_type",
        ])
        lines.append("")
        lines.append(problem_df[preview_cols].head(12).to_string(index=False))
    else:
        lines.append("No se detectaron filas problemáticas según las reglas del verificador.")
    lines.append("")

    if optional_eval_summary is not None:
        lines.append("Resumen JSON original de evaluate_s1_answers.py")
        lines.append("----------------------------------------------")
        lines.append("Se encontró y se cargó correctamente el summary original.")
        if "overall" in optional_eval_summary:
            lines.append("overall:")
            lines.append(json.dumps(optional_eval_summary["overall"], ensure_ascii=False, indent=2))
        lines.append("")

    lines.append("Archivos generados")
    lines.append("------------------")
    lines.append(f"- {output_dir / 's1_verificacion_resumen.txt'}")
    lines.append(f"- {output_dir / 's1_verificacion_resumen.json'}")
    lines.append(f"- {output_dir / 's1_casos_problematicos.csv'}")
    lines.append(f"- {output_dir / 's1_muestra_respuestas.csv'}")
    lines.append(f"- {output_dir / 's1_metricas_por_grupo.csv'}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una verificación compacta de resultados parseados/evaluados de S1."
    )
    parser.add_argument(
        "--parsed-path",
        type=Path,
        default=DEFAULT_PARSED_PATH,
        help="CSV parseado generado por parse_s1_outputs.py.",
    )
    parser.add_argument(
        "--evaluated-path",
        type=Path,
        default=DEFAULT_EVALUATED_PATH,
        help="CSV evaluado generado por evaluate_s1_answers.py.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON resumen generado por evaluate_s1_answers.py, si existe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde guardar archivos de verificación.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Cantidad máxima de filas para la muestra de respuestas.",
    )
    args = parser.parse_args()

    if not args.parsed_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV parseado: {args.parsed_path}")
    if not args.evaluated_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV evaluado: {args.evaluated_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed_df = pd.read_csv(args.parsed_path)
    evaluated_df = pd.read_csv(args.evaluated_path)

    problem_mask = build_problem_mask(evaluated_df)
    problem_df = evaluated_df[problem_mask].copy()

    summary = summarize_overall(parsed_df, evaluated_df, problem_df)
    optional_eval_summary = load_optional_summary(args.summary_path)

    group_df = build_group_metrics(evaluated_df)

    # Salidas tabulares.
    problem_cols = existing_columns(problem_df, PROBLEM_PRIORITY_COLUMNS)
    if problem_cols:
        problem_out = problem_df[problem_cols].copy()
    else:
        problem_out = problem_df.copy()

    sample_cols = existing_columns(evaluated_df, SAMPLE_PRIORITY_COLUMNS)
    sample_df = evaluated_df[sample_cols].head(args.sample_size).copy() if sample_cols else evaluated_df.head(args.sample_size).copy()

    problem_path = args.output_dir / "s1_casos_problematicos.csv"
    sample_path = args.output_dir / "s1_muestra_respuestas.csv"
    group_path = args.output_dir / "s1_metricas_por_grupo.csv"
    summary_json_path = args.output_dir / "s1_verificacion_resumen.json"
    report_path = args.output_dir / "s1_verificacion_resumen.txt"

    problem_out.to_csv(problem_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    group_df.to_csv(group_path, index=False)

    # JSON compacto, redondeando floats para legibilidad.
    def recursively_round(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: recursively_round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [recursively_round(v) for v in obj]
        return safe_round(obj, 5)

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(recursively_round(summary), f, ensure_ascii=False, indent=2)

    write_text_report(
        report_path=report_path,
        parsed_path=args.parsed_path,
        evaluated_path=args.evaluated_path,
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        summary=summary,
        problem_df=problem_df,
        group_df=group_df,
        optional_eval_summary=optional_eval_summary,
    )

    print("\nVerificación S1 generada correctamente")
    print("=====================================")
    print(f"Carpeta: {args.output_dir}")
    print(f"Resumen TXT: {report_path}")
    print(f"Resumen JSON: {summary_json_path}")
    print(f"Casos problemáticos: {problem_path}")
    print(f"Muestra de respuestas: {sample_path}")
    print(f"Métricas por grupo: {group_path}")
    print("")
    print("Resumen compacto")
    print("----------------")
    print(f"Filas parsed/evaluated: {len(parsed_df)}/{len(evaluated_df)}")
    print(f"Valid format rate: {format_metric(summary['parse']['valid_format_rate'])}")
    print(f"Answer accuracy: {format_metric(summary['answer_quality']['answer_accuracy'])}")
    print(f"Avg token F1: {format_metric(summary['answer_quality']['avg_token_f1'])}")
    print(f"Wrong abstention rate: {format_metric(summary['abstention']['wrong_abstention_rate'])}")
    print(f"Retrieval hit gold rate: {format_metric(summary['retrieval']['retrieval_hit_gold_rate'])}")
    print(f"Retrieval full recall rate: {format_metric(summary['retrieval']['retrieval_full_recall_rate'])}")
    print(f"Casos problemáticos: {summary['n_problem_rows']}")


if __name__ == "__main__":
    main()

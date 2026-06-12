#!/usr/bin/env python3
"""
evaluate_s2_routing.py

Evalúa el router de S2 Adaptive-RAG.

Entrada esperada:
    outputs/s2/generation/adaptive_rag_s2_parsed.csv

Salida por defecto:
    outputs/s2/evaluation/adaptive_rag_s2_routing_results.csv
    outputs/s2/evaluation/adaptive_rag_s2_routing_summary.json
    outputs/s2/evaluation/adaptive_rag_s2_routing_summary_by_group.csv

Diseño:
    - No llama al LLM.
    - Evalúa solamente la decisión de ruteo: direct / retrieve / abstain / clarify.
    - Compara ruta predicha contra expected_route.
    - También considera acceptable_routes_json para una métrica relajada.
    - Separa over-retrieval, under-retrieval, errores de abstención y errores de aclaración.
    - Genera métricas globales y por grupo para análisis del modelo S2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    S2_PARSED_OUTPUT_PATH,
    S2_ROUTING_GROUP_SUMMARY_PATH,
    S2_ROUTING_RESULTS_PATH,
    S2_ROUTING_SUMMARY_PATH,
)


DEFAULT_INPUT_PATH = S2_PARSED_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = S2_ROUTING_RESULTS_PATH
DEFAULT_SUMMARY_PATH = S2_ROUTING_SUMMARY_PATH
DEFAULT_GROUP_SUMMARY_PATH = S2_ROUTING_GROUP_SUMMARY_PATH

VALID_ROUTES = {"direct", "retrieve", "abstain", "clarify"}
VALID_RETRIEVAL_MODES = {"none", "single_step", "multi_step"}

EVALUATION_COLUMNS = [
    "eval_predicted_route",
    "eval_expected_route",
    "eval_acceptable_routes",
    "eval_route_exact_match",
    "eval_route_acceptable",
    "eval_valid_route",
    "eval_valid_retrieval_mode",
    "eval_retrieval_mode_exact_match",
    "eval_retrieval_mode_consistent",
    "eval_over_retrieval_strict",
    "eval_over_retrieval_unacceptable",
    "eval_under_retrieval",
    "eval_missed_clarify",
    "eval_missed_abstain",
    "eval_unnecessary_clarify",
    "eval_unnecessary_abstain",
    "eval_error_type",
    "eval_needs_manual_review",
    "eval_notes",
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
    return str(value).strip()


def normalize_label(value: Any) -> str:
    return clean_text(value).lower().strip().replace(" ", "_").replace("-", "_")


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


def safe_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None

    if 1.0 < number <= 100.0:
        number = number / 100.0

    return number


def parse_json_list(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, list):
        return [normalize_label(x) for x in value if clean_text(x)]

    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [normalize_label(x) for x in parsed if clean_text(x)]
    except json.JSONDecodeError:
        pass

    if "|" in text:
        return [normalize_label(x) for x in text.split("|") if clean_text(x)]

    return [normalize_label(text)]


def normalize_route(value: Any) -> str:
    text = normalize_label(value)

    aliases = {
        "no_retrieval": "direct",
        "no_retrieve": "direct",
        "direct_answer": "direct",
        "answer_directly": "direct",
        "rag": "retrieve",
        "retrieval": "retrieve",
        "use_retrieval": "retrieve",
        "use_rag": "retrieve",
        "ask_clarification": "clarify",
        "needs_clarification": "clarify",
        "clarification": "clarify",
        "not_enough_information": "abstain",
        "insufficient_information": "abstain",
        "no_answer": "abstain",
        "refuse": "abstain",
    }

    return aliases.get(text, text)


def normalize_retrieval_mode(value: Any) -> str:
    text = normalize_label(value)

    aliases = {
        "": "none",
        "no_retrieval": "none",
        "no_retrieve": "none",
        "single": "single_step",
        "one_step": "single_step",
        "simple": "single_step",
        "single_step_retrieval": "single_step",
        "multi": "multi_step",
        "multi_hop": "multi_step",
        "multihop": "multi_step",
        "iterative": "multi_step",
        "multi_step_retrieval": "multi_step",
    }

    return aliases.get(text, text)


def pick_predicted_route(row: pd.Series) -> str:
    """
    Usa parsed_route si existe; si no, predicted_route.

    Esto permite evaluar tanto:
    - outputs ya parseados por parse_s2_outputs.py
    - outputs crudos que ya tengan predicted_route del runner/router
    """
    for col in ["parsed_route", "predicted_route"]:
        if col in row.index:
            value = normalize_route(row.get(col, ""))
            if value:
                return value

    return ""


def pick_predicted_retrieval_mode(row: pd.Series) -> str:
    for col in ["parsed_retrieval_mode", "predicted_retrieval_mode"]:
        if col in row.index:
            value = normalize_retrieval_mode(row.get(col, ""))
            if value:
                return value
    return ""


def route_is_valid(route: str) -> bool:
    return route in VALID_ROUTES


def retrieval_mode_is_valid(mode: str) -> bool:
    return mode in VALID_RETRIEVAL_MODES


def retrieval_mode_consistent(route: str, mode: str) -> bool:
    if route == "retrieve":
        return mode in {"single_step", "multi_step"}
    return mode == "none"


# ---------------------------------------------------------------------------
# Evaluación por fila
# ---------------------------------------------------------------------------


def evaluate_row(row: pd.Series) -> dict[str, Any]:
    predicted_route = pick_predicted_route(row)
    predicted_mode = pick_predicted_retrieval_mode(row)

    expected_route = normalize_route(row.get("expected_route", ""))
    expected_mode = normalize_retrieval_mode(row.get("retrieval_mode", ""))

    acceptable_routes = parse_json_list(row.get("acceptable_routes_json", ""))
    if not acceptable_routes and expected_route:
        acceptable_routes = [expected_route]

    valid_route = route_is_valid(predicted_route)
    valid_mode = retrieval_mode_is_valid(predicted_mode)
    mode_consistent = valid_mode and retrieval_mode_consistent(predicted_route, predicted_mode)

    route_exact_match = bool(expected_route and predicted_route == expected_route)
    route_acceptable = bool(predicted_route and predicted_route in acceptable_routes)

    # Para expected_route=retrieve nos interesa también si el modo fue correcto.
    # Para rutas no retrieve, el modo correcto es none.
    if expected_route and predicted_route:
        retrieval_mode_exact_match = predicted_mode == expected_mode
    else:
        retrieval_mode_exact_match = False

    over_retrieval_strict = bool(expected_route != "retrieve" and predicted_route == "retrieve")
    over_retrieval_unacceptable = bool(over_retrieval_strict and not route_acceptable)
    under_retrieval = bool(expected_route == "retrieve" and predicted_route != "retrieve")

    missed_clarify = bool(expected_route == "clarify" and predicted_route != "clarify")
    missed_abstain = bool(expected_route == "abstain" and predicted_route != "abstain" and not route_acceptable)
    unnecessary_clarify = bool(expected_route != "clarify" and predicted_route == "clarify")
    unnecessary_abstain = bool(expected_route != "abstain" and predicted_route == "abstain")

    run_error_present = bool(coerce_bool(row.get("run_error_present")))
    router_valid_format = coerce_bool(row.get("valid_router_format"))
    router_error = clean_text(row.get("router_error", ""))

    notes: list[str] = []
    needs_manual_review = False

    if run_error_present:
        error_type = "run_error"
        notes.append("Hubo error de ejecución en el runner S2.")
        needs_manual_review = True
    elif router_error:
        error_type = "router_error"
        notes.append(f"Hubo error del router: {router_error[:160]}")
        needs_manual_review = True
    elif router_valid_format is False:
        error_type = "invalid_router_format"
        notes.append("El formato del router no fue válido.")
        needs_manual_review = True
    elif not valid_route:
        error_type = "invalid_route"
        notes.append(f"Ruta predicha inválida: {predicted_route!r}.")
        needs_manual_review = True
    elif not mode_consistent:
        error_type = "inconsistent_retrieval_mode"
        notes.append(
            f"Modo de recuperación inconsistente: route={predicted_route!r}, "
            f"retrieval_mode={predicted_mode!r}."
        )
        needs_manual_review = True
    elif route_exact_match:
        if expected_route == "retrieve" and not retrieval_mode_exact_match:
            error_type = "route_correct_mode_wrong"
            notes.append("La ruta retrieve fue correcta, pero el retrieval_mode no coincide con el esperado.")
        else:
            error_type = "correct"
    elif route_acceptable:
        error_type = "acceptable_alternative"
        notes.append("La ruta no coincide estrictamente, pero está dentro de acceptable_routes_json.")
    elif under_retrieval:
        error_type = "under_retrieval"
        notes.append("El router no recuperó aunque expected_route=retrieve.")
    elif over_retrieval_unacceptable:
        error_type = "over_retrieval"
        notes.append("El router recuperó cuando la recuperación no era aceptable.")
    elif missed_clarify:
        error_type = "missed_clarify"
        notes.append("El router no pidió aclaración cuando se esperaba clarify.")
    elif missed_abstain:
        error_type = "missed_abstain"
        notes.append("El router no se abstuvo cuando se esperaba abstain.")
    elif unnecessary_clarify:
        error_type = "unnecessary_clarify"
        notes.append("El router pidió aclaración cuando se esperaba otra ruta.")
    elif unnecessary_abstain:
        error_type = "unnecessary_abstain"
        notes.append("El router se abstuvo cuando se esperaba otra ruta.")
    else:
        error_type = "wrong_route"
        notes.append("La ruta predicha no coincide con la esperada ni es una alternativa aceptable.")
        needs_manual_review = True

    return {
        "eval_predicted_route": predicted_route,
        "eval_expected_route": expected_route,
        "eval_acceptable_routes": "|".join(acceptable_routes),
        "eval_route_exact_match": bool(route_exact_match),
        "eval_route_acceptable": bool(route_acceptable),
        "eval_valid_route": bool(valid_route),
        "eval_valid_retrieval_mode": bool(valid_mode),
        "eval_retrieval_mode_exact_match": bool(retrieval_mode_exact_match),
        "eval_retrieval_mode_consistent": bool(mode_consistent),
        "eval_over_retrieval_strict": bool(over_retrieval_strict),
        "eval_over_retrieval_unacceptable": bool(over_retrieval_unacceptable),
        "eval_under_retrieval": bool(under_retrieval),
        "eval_missed_clarify": bool(missed_clarify),
        "eval_missed_abstain": bool(missed_abstain),
        "eval_unnecessary_clarify": bool(unnecessary_clarify),
        "eval_unnecessary_abstain": bool(unnecessary_abstain),
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
    if series.empty:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def precision_recall_f1_for_route(df: pd.DataFrame, route: str) -> dict[str, Any]:
    expected = df["eval_expected_route"].astype(str) == route
    predicted = df["eval_predicted_route"].astype(str) == route

    tp = int((expected & predicted).sum())
    fp = int((~expected & predicted).sum())
    fn = int((expected & ~predicted).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_confusion_matrix(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}

    if df.empty:
        return matrix

    for expected, sub in df.groupby("eval_expected_route", dropna=False):
        expected_key = clean_text(expected)
        matrix[expected_key] = {
            clean_text(predicted): int(count)
            for predicted, count in sub["eval_predicted_route"].value_counts(dropna=False).items()
        }

    return matrix


def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}

    summary: dict[str, Any] = {
        "n": int(len(df)),
        "routing_accuracy_strict": mean_bool(df["eval_route_exact_match"]),
        "routing_accuracy_relaxed": mean_bool(df["eval_route_acceptable"]),
        "valid_route_rate": mean_bool(df["eval_valid_route"]),
        "valid_retrieval_mode_rate": mean_bool(df["eval_valid_retrieval_mode"]),
        "retrieval_mode_consistency_rate": mean_bool(df["eval_retrieval_mode_consistent"]),
        "retrieval_mode_exact_match_rate": mean_bool(df["eval_retrieval_mode_exact_match"]),
        "over_retrieval_strict_rate": mean_bool(df["eval_over_retrieval_strict"]),
        "over_retrieval_unacceptable_rate": mean_bool(df["eval_over_retrieval_unacceptable"]),
        "under_retrieval_rate": mean_bool(df["eval_under_retrieval"]),
        "missed_clarify_rate": mean_bool(df["eval_missed_clarify"]),
        "missed_abstain_rate": mean_bool(df["eval_missed_abstain"]),
        "unnecessary_clarify_rate": mean_bool(df["eval_unnecessary_clarify"]),
        "unnecessary_abstain_rate": mean_bool(df["eval_unnecessary_abstain"]),
        "manual_review_rate": mean_bool(df["eval_needs_manual_review"]),
        "avg_router_confidence": mean_numeric(df["router_confidence"]) if "router_confidence" in df else None,
        "avg_router_latency_seconds": mean_numeric(df["router_latency_seconds"]) if "router_latency_seconds" in df else None,
        "avg_router_total_tokens": mean_numeric(df["router_total_tokens"]) if "router_total_tokens" in df else None,
    }

    # Si vienen columnas parseadas, también reportamos calidad del parseo del router.
    if "valid_router_format" in df.columns:
        summary["valid_router_format_rate"] = mean_bool(df["valid_router_format"])
    if "run_error_present" in df.columns:
        summary["run_error_rate"] = mean_bool(df["run_error_present"])

    summary["expected_route_counts"] = {
        clean_text(k): int(v)
        for k, v in df["eval_expected_route"].value_counts(dropna=False).items()
    }
    summary["predicted_route_counts"] = {
        clean_text(k): int(v)
        for k, v in df["eval_predicted_route"].value_counts(dropna=False).items()
    }
    summary["error_type_counts"] = {
        clean_text(k): int(v)
        for k, v in df["eval_error_type"].value_counts(dropna=False).items()
    }

    summary["per_route"] = {
        route: precision_recall_f1_for_route(df, route)
        for route in sorted(VALID_ROUTES)
    }

    summary["confusion_matrix"] = build_confusion_matrix(df)

    return summary


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append({"group_type": "overall", "group": "all", **summarize_subset(df)})

    group_cols = [
        "s2_case_type",
        "expected_route",
        "retrieval_mode",
        "expected_behavior",
        "generation_policy",
        "router_strategy",
        "router_source",
        "source_dataset",
        "dataset",
    ]

    for group_col in group_cols:
        if group_col in df.columns:
            for value, subset in df.groupby(group_col, dropna=False):
                rows.append({
                    "group_type": group_col,
                    "group": clean_text(value),
                    **summarize_subset(subset),
                })

    if "s2_case_type" in df.columns and "expected_route" in df.columns:
        for (case_type, expected_route), subset in df.groupby(["s2_case_type", "expected_route"], dropna=False):
            rows.append({
                "group_type": "s2_case_type::expected_route",
                "group": f"{clean_text(case_type)}::{clean_text(expected_route)}",
                **summarize_subset(subset),
            })

    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame) -> None:
    required = {"id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(sorted(missing))
        )

    if "expected_route" not in df.columns:
        raise ValueError("El CSV debe tener la columna expected_route.")

    if "parsed_route" not in df.columns and "predicted_route" not in df.columns:
        raise ValueError("El CSV debe tener parsed_route o predicted_route.")


def evaluate_file(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    group_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    eval_rows = [evaluate_row(row) for _, row in df.iterrows()]
    eval_df = pd.DataFrame(eval_rows)

    base_cols = [col for col in df.columns if col not in EVALUATION_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), eval_df], axis=1)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "overall": summarize_subset(output_df),
    }

    for group_col in [
        "s2_case_type",
        "expected_route",
        "retrieval_mode",
        "expected_behavior",
        "generation_policy",
        "router_strategy",
        "router_source",
        "source_dataset",
        "dataset",
    ]:
        if group_col in output_df.columns:
            summary[f"by_{group_col}"] = {
                clean_text(value): summarize_subset(subset)
                for value, subset in output_df.groupby(group_col, dropna=False)
            }

    group_summary_df = build_group_summary(output_df)

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
    overall = summary.get("overall", {})

    print("\nResumen evaluación routing S2")
    print("-----------------------------")
    print(f"Filas totales: {overall.get('n', 0)}")
    print(f"Routing accuracy strict: {fmt_metric(overall.get('routing_accuracy_strict'))}")
    print(f"Routing accuracy relaxed: {fmt_metric(overall.get('routing_accuracy_relaxed'))}")
    print(f"Over-retrieval strict rate: {fmt_metric(overall.get('over_retrieval_strict_rate'))}")
    print(f"Over-retrieval unacceptable rate: {fmt_metric(overall.get('over_retrieval_unacceptable_rate'))}")
    print(f"Under-retrieval rate: {fmt_metric(overall.get('under_retrieval_rate'))}")
    print(f"Retrieval mode consistency rate: {fmt_metric(overall.get('retrieval_mode_consistency_rate'))}")
    print(f"Valid router format rate: {fmt_metric(overall.get('valid_router_format_rate'))}")
    print(f"Run error rate: {fmt_metric(overall.get('run_error_rate'))}")
    print(f"Avg router confidence: {fmt_metric(overall.get('avg_router_confidence'))}")

    print("\nRutas esperadas:")
    for route, count in overall.get("expected_route_counts", {}).items():
        print(f"- {route}: {count}")

    print("\nRutas predichas:")
    for route, count in overall.get("predicted_route_counts", {}).items():
        print(f"- {route}: {count}")

    print("\nTipos de resultado/error:")
    for error_type, count in overall.get("error_type_counts", {}).items():
        print(f"- {error_type}: {count}")

    per_route = overall.get("per_route", {})
    if per_route:
        print("\nPrecision/Recall/F1 por ruta:")
        for route in sorted(per_route):
            metrics = per_route[route]
            print(
                f"- {route}: "
                f"precision={fmt_metric(metrics.get('precision'))}, "
                f"recall={fmt_metric(metrics.get('recall'))}, "
                f"f1={fmt_metric(metrics.get('f1'))}, "
                f"tp={metrics.get('tp')}, fp={metrics.get('fp')}, fn={metrics.get('fn')}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa el routing de S2 Adaptive-RAG."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV parseado de S2, generado por parse_s2_outputs.py.",
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
        help="CSV con resumen por s2_case_type, expected_route, etc.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _, summary, _ = evaluate_file(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        group_summary_path=args.group_summary_path,
    )

    print(f"Resultados de routing evaluados guardados en: {args.output_path}")
    print(f"Resumen JSON guardado en: {args.summary_path}")
    print(f"Resumen por grupos guardado en: {args.group_summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()

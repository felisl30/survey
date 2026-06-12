#!/usr/bin/env python3
"""
parse_s2_outputs.py

Parsea las respuestas crudas generadas por run_s2_adaptive_rag.py para S2
Adaptive-RAG.

Entrada por defecto:
    outputs/s2/generation/adaptive_rag_s2_raw.csv

Salida por defecto:
    outputs/s2/generation/adaptive_rag_s2_parsed.csv

Diseño:
    - No llama al LLM.
    - Conserva todas las columnas originales del CSV crudo.
    - Agrega columnas normalizadas para evaluar router y respuesta final:
        run_error_present
        parsed_route
        parsed_retrieval_mode
        parsed_router_confidence
        parsed_router_reason
        valid_router_format
        router_parse_method_final
        router_parse_error_final
        parsed_router_json
        parsed_answer
        parsed_confidence
        parsed_abstained
        valid_answer_format
        valid_format
        parse_method
        parse_error
        parsed_json
    - Soporta JSON limpio, JSON dentro de bloques Markdown y JSON embebido.
    - En dry-run no marca el mensaje DRY_RUN como error real de ejecución.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import S2_PARSED_OUTPUT_PATH, S2_RAW_OUTPUT_PATH


DEFAULT_INPUT_PATH = S2_RAW_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = S2_PARSED_OUTPUT_PATH

VALID_ROUTES = {"direct", "retrieve", "abstain", "clarify"}
VALID_RETRIEVAL_MODES = {"none", "single_step", "multi_step"}

ADDITIONAL_COLUMNS = [
    "run_error_present",
    "parsed_route",
    "parsed_retrieval_mode",
    "parsed_router_confidence",
    "parsed_router_reason",
    "valid_router_format",
    "router_parse_method_final",
    "router_parse_error_final",
    "parsed_router_json",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "valid_answer_format",
    "valid_format",
    "parse_method",
    "parse_error",
    "parsed_json",
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


def to_json_string(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def coerce_confidence(value: Any) -> float | None:
    """Convierte confidence a float en [0, 1] cuando sea posible."""
    if is_missing(value):
        return None

    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if text.endswith("%"):
            try:
                number = float(text[:-1].strip()) / 100.0
                return min(max(number, 0.0), 1.0)
            except ValueError:
                return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

    if 1.0 < number <= 100.0:
        number = number / 100.0

    if 0.0 <= number <= 1.0:
        return number

    return None


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


def normalize_token(value: Any) -> str:
    return clean_text(value).strip().lower().replace(" ", "_").replace("-", "_")


def infer_abstention_from_text(text: str) -> bool:
    lower = clean_text(text).lower()
    return any(marker in lower for marker in ABSTENTION_MARKERS)


# ---------------------------------------------------------------------------
# Extracción robusta de JSON
# ---------------------------------------------------------------------------


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
    """Extrae el primer objeto JSON balanceado que aparezca en el texto."""
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
        return None, "No se encontró un objeto JSON."

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON inválido: {exc}"

    if not isinstance(parsed, dict):
        return None, "El JSON parseado no es un objeto/dict."

    return parsed, ""


# ---------------------------------------------------------------------------
# Parser del router
# ---------------------------------------------------------------------------


def normalize_route(value: Any) -> str:
    route = normalize_token(value)
    aliases = {
        "rag": "retrieve",
        "retrieval": "retrieve",
        "use_rag": "retrieve",
        "use_retrieval": "retrieve",
        "answer_directly": "direct",
        "direct_answer": "direct",
        "no_retrieval": "direct",
        "ask_clarification": "clarify",
        "needs_clarification": "clarify",
        "clarification": "clarify",
        "not_enough_information": "abstain",
        "insufficient_information": "abstain",
        "no_answer": "abstain",
    }
    route = aliases.get(route, route)
    return route if route in VALID_ROUTES else ""


def normalize_retrieval_mode(value: Any, *, route: str) -> str:
    mode = normalize_token(value)
    aliases = {
        "no_retrieval": "none",
        "no_retrieve": "none",
        "single": "single_step",
        "simple": "single_step",
        "one_step": "single_step",
        "single_step_retrieval": "single_step",
        "multi": "multi_step",
        "multihop": "multi_step",
        "multi_hop": "multi_step",
        "multi_step_retrieval": "multi_step",
        "iterative": "multi_step",
    }
    mode = aliases.get(mode, mode)

    if route != "retrieve":
        return "none"

    if mode in {"single_step", "multi_step"}:
        return mode

    return "single_step"


def parse_router_row(row: pd.Series) -> dict[str, Any]:
    """Parsea la decisión del router desde columnas y, si hace falta, router_raw_output."""
    router_raw_output = clean_text(row.get("router_raw_output", ""))
    obj: dict[str, Any] | None = None
    json_error = ""

    if router_raw_output:
        obj, json_error = load_json_object(router_raw_output)

    # Prioridad: columnas ya normalizadas por router_s2/run_s2. Fallback: JSON crudo.
    route_source = row.get("predicted_route", "")
    if is_missing(route_source) and obj is not None:
        route_source = obj.get("route", "")
    parsed_route = normalize_route(route_source)

    mode_source = row.get("predicted_retrieval_mode", "")
    if is_missing(mode_source) and obj is not None:
        mode_source = obj.get("retrieval_mode", "")
    parsed_retrieval_mode = normalize_retrieval_mode(mode_source, route=parsed_route)

    conf_source = row.get("router_confidence", "")
    if is_missing(conf_source) and obj is not None:
        conf_source = obj.get("confidence", "")
    parsed_router_confidence = coerce_confidence(conf_source)

    reason_source = row.get("router_reason", "")
    if is_missing(reason_source) and obj is not None:
        reason_source = obj.get("reason", "")
    parsed_router_reason = clean_text(reason_source)

    errors: list[str] = []
    if parsed_route not in VALID_ROUTES:
        errors.append("Ruta predicha ausente o inválida.")
    if parsed_retrieval_mode not in VALID_RETRIEVAL_MODES:
        errors.append("retrieval_mode ausente o inválido.")
    if parsed_route == "retrieve" and parsed_retrieval_mode == "none":
        errors.append("route=retrieve requiere retrieval_mode single_step o multi_step.")
    if parsed_route != "retrieve" and parsed_retrieval_mode != "none":
        errors.append("route distinta de retrieve debería usar retrieval_mode=none.")
    if router_raw_output and obj is None:
        # No invalida automáticamente la ruta si las columnas predichas estaban bien,
        # pero deja registrada la falla del JSON crudo.
        errors.append(json_error)

    valid_router_format = bool(parsed_route in VALID_ROUTES and parsed_retrieval_mode in VALID_RETRIEVAL_MODES)
    if parsed_route == "retrieve" and parsed_retrieval_mode == "none":
        valid_router_format = False

    parse_method = "columns"
    if obj is not None:
        parse_method = "columns_with_router_json"
    elif router_raw_output:
        parse_method = "columns_router_json_failed"

    parsed_router_json = obj if obj is not None else {
        "route": parsed_route,
        "retrieval_mode": parsed_retrieval_mode,
        "confidence": parsed_router_confidence,
        "reason": parsed_router_reason,
    }

    return {
        "parsed_route": parsed_route,
        "parsed_retrieval_mode": parsed_retrieval_mode,
        "parsed_router_confidence": parsed_router_confidence,
        "parsed_router_reason": parsed_router_reason,
        "valid_router_format": valid_router_format,
        "router_parse_method_final": parse_method,
        "router_parse_error_final": " ".join(error for error in errors if error),
        "parsed_router_json": to_json_string(parsed_router_json),
    }


# ---------------------------------------------------------------------------
# Parser de respuesta final
# ---------------------------------------------------------------------------


def parse_answer_json(obj: dict[str, Any]) -> dict[str, Any]:
    answer = clean_text(obj.get("answer", ""))
    confidence = coerce_confidence(obj.get("confidence"))
    abstained = coerce_bool(obj.get("abstained"))

    errors: list[str] = []
    if not answer:
        errors.append('El campo "answer" está vacío o ausente.')
    if abstained is None:
        errors.append('El campo "abstained" está ausente o no es booleano.')
    if "confidence" in obj and confidence is None:
        errors.append('El campo "confidence" no pudo convertirse a float en [0, 1].')

    valid = len(errors) == 0

    return {
        "parsed_answer": answer,
        "parsed_confidence": confidence,
        "parsed_abstained": abstained,
        "valid_answer_format": valid,
        # Alias para reutilizar evaluadores o lógica previa estilo S1.
        "valid_format": valid,
        "parse_method": "json" if valid else "json_incomplete",
        "parse_error": " ".join(errors),
        "parsed_json": to_json_string(obj),
    }


def parse_answer_fallback(raw_output: str, json_error: str) -> dict[str, Any]:
    text = clean_text(raw_output)

    if not text:
        return {
            "parsed_answer": "",
            "parsed_confidence": None,
            "parsed_abstained": None,
            "valid_answer_format": False,
            "valid_format": False,
            "parse_method": "no_output",
            "parse_error": "No hay raw_output para parsear." + (f" {json_error}" if json_error else ""),
            "parsed_json": "",
        }

    return {
        "parsed_answer": text,
        "parsed_confidence": None,
        "parsed_abstained": infer_abstention_from_text(text),
        "valid_answer_format": False,
        "valid_format": False,
        "parse_method": "fallback_text",
        "parse_error": json_error + " Se usó raw_output como respuesta abierta.",
        "parsed_json": "",
    }


def parse_answer_row(row: pd.Series) -> dict[str, Any]:
    raw_output = clean_text(row.get("raw_output", ""))

    if not raw_output:
        return parse_answer_fallback(raw_output, "")

    obj, json_error = load_json_object(raw_output)
    if obj is not None:
        return parse_answer_json(obj)

    return parse_answer_fallback(raw_output, json_error=json_error)


# ---------------------------------------------------------------------------
# Pipeline por fila y archivo
# ---------------------------------------------------------------------------


def parse_row(row: pd.Series) -> dict[str, Any]:
    error = clean_text(row.get("error", ""))
    dry_run = coerce_bool(row.get("dry_run", False))

    # En dry-run, run_s2_adaptive_rag.py guarda una nota en error. No es fallo real.
    run_error_present = bool(error) and not bool(dry_run)

    router_parsed = parse_router_row(row)
    answer_parsed = parse_answer_row(row)

    return {
        "run_error_present": run_error_present,
        **router_parsed,
        **answer_parsed,
    }


def validate_input(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna 'id'.")
    if "raw_output" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna 'raw_output'.")
    if "predicted_route" not in df.columns:
        raise ValueError("El CSV de entrada debe tener columna 'predicted_route'.")


def parse_outputs(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    parsed_rows = [parse_row(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)

    # Evita duplicar columnas si se re-parsea un archivo ya parseado.
    base_cols = [col for col in df.columns if col not in ADDITIONAL_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), parsed_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


# ---------------------------------------------------------------------------
# Resumen de terminal
# ---------------------------------------------------------------------------


def mean_bool(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.astype(bool).mean())


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(df: pd.DataFrame) -> None:
    print("\nResumen de parseo S2")
    print("--------------------")
    print(f"Filas totales: {len(df)}")

    if "parsed_route" in df.columns:
        print("\nRutas parseadas:")
        print(df["parsed_route"].value_counts(dropna=False).to_string())

    if "parse_method" in df.columns:
        print("\nMétodos de parseo de respuesta:")
        print(df["parse_method"].value_counts(dropna=False).to_string())

    if "valid_router_format" in df.columns:
        print(f"\nValid router format rate: {fmt(mean_bool(df['valid_router_format']))}")

    if "valid_answer_format" in df.columns:
        print(f"Valid answer format rate: {fmt(mean_bool(df['valid_answer_format']))}")

    if "run_error_present" in df.columns:
        print(f"Run error rate: {fmt(mean_bool(df['run_error_present']))}")

    if "parsed_abstained" in df.columns:
        print(f"Abstention rate: {fmt(mean_bool(df['parsed_abstained']))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parsea respuestas crudas de S2 Adaptive-RAG."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV crudo generado por run_s2_adaptive_rag.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV parseado de salida.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_df = parse_outputs(args.input_path, args.output_path)
    print(f"Resultados parseados guardados en: {args.output_path}")
    print_summary(output_df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
parse_s1_outputs.py

Parsea las respuestas crudas generadas por run_s1_rag.py para S1 RAG básico.

Entrada por defecto:
    outputs/s1/generation/hotpotqa_mini_s1_raw.csv

Salida por defecto:
    outputs/s1/generation/hotpotqa_mini_s1_parsed.csv

Diseño:
    - No llama al LLM.
    - Conserva todas las columnas originales del CSV crudo.
    - Agrega columnas normalizadas de parseo:
        run_error_present
        parsed_answer
        parsed_confidence
        parsed_abstained
        valid_format
        parse_method
        parse_error
        parsed_json
    - Soporta JSON limpio, JSON dentro de bloques Markdown y texto con JSON embebido.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_raw.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s1/generation/hotpotqa_mini_s1_parsed.csv")

ADDITIONAL_COLUMNS = [
    "run_error_present",
    "parsed_answer",
    "parsed_confidence",
    "parsed_abstained",
    "valid_format",
    "parse_method",
    "parse_error",
    "parsed_json",
]


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """Detecta None/NaN/cadenas vacías."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    """Convierte valores a string limpio, evitando 'nan'."""
    if is_missing(value):
        return ""
    return str(value).strip()


def to_json_string(value: Any) -> str:
    """Serializa a JSON de forma estable para guardar en CSV."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def coerce_confidence(value: Any) -> float | None:
    """Convierte confidence a float en [0, 1] cuando sea posible."""
    if value is None or is_missing(value):
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

    # Si el modelo devolvió 95 en lugar de 0.95, lo interpretamos como porcentaje.
    if 1.0 < number <= 100.0:
        number = number / 100.0

    if 0.0 <= number <= 1.0:
        return number

    return None


def coerce_bool(value: Any) -> bool | None:
    """Convierte valores comunes a booleano."""
    if isinstance(value, bool):
        return value
    if value is None or is_missing(value):
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


def infer_abstention_from_text(text: str) -> bool:
    """Heurística simple para detectar abstención en respuestas abiertas."""
    lower = clean_text(text).lower()
    abstention_markers = [
        "no sé",
        "no se",
        "no estoy seguro",
        "no tengo información suficiente",
        "no tengo informacion suficiente",
        "no hay información suficiente",
        "no hay informacion suficiente",
        "información insuficiente",
        "informacion insuficiente",
        "insufficient information",
        "not enough information",
        "i don't know",
        "i do not know",
        "cannot determine",
        "can't determine",
    ]
    return any(marker in lower for marker in abstention_markers)


# ---------------------------------------------------------------------------
# Extracción robusta de JSON
# ---------------------------------------------------------------------------


def strip_markdown_fence(text: str) -> str:
    """Remueve fences Markdown si la respuesta viene como ```json ... ```."""
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
    """
    Extrae el primer objeto JSON balanceado que aparezca en el texto.

    Esto permite parsear respuestas como:
        Aquí está la respuesta: {"answer": "...", "confidence": 0.9}
    """
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
    """Intenta parsear un objeto JSON desde una respuesta cruda."""
    json_text = extract_first_json_object(text)
    if not json_text:
        return None, "No se encontró un objeto JSON en raw_output."

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON inválido: {exc}"

    if not isinstance(parsed, dict):
        return None, "El JSON parseado no es un objeto/dict."

    return parsed, ""


# ---------------------------------------------------------------------------
# Parser S1
# ---------------------------------------------------------------------------


def parse_s1_json(obj: dict[str, Any]) -> dict[str, Any]:
    """Parsea el JSON esperado de S1: answer/confidence/abstained."""
    answer = clean_text(obj.get("answer", ""))
    confidence = coerce_confidence(obj.get("confidence"))
    abstained = coerce_bool(obj.get("abstained"))

    errors: list[str] = []
    if not answer:
        errors.append('El campo "answer" está vacío o ausente.')
    if abstained is None:
        errors.append('El campo "abstained" está ausente o no es booleano.')
    if confidence is None and "confidence" in obj:
        errors.append('El campo "confidence" no pudo convertirse a float en [0, 1].')

    valid_format = len(errors) == 0

    return {
        "parsed_answer": answer,
        "parsed_confidence": confidence,
        "parsed_abstained": abstained,
        "valid_format": valid_format,
        "parse_method": "json" if valid_format else "json_incomplete",
        "parse_error": " ".join(errors),
        "parsed_json": to_json_string(obj),
    }


def parse_s1_fallback(raw_output: str, json_error: str) -> dict[str, Any]:
    """Fallback conservador: usa el texto crudo como respuesta abierta."""
    text = clean_text(raw_output)

    if not text:
        return {
            "parsed_answer": "",
            "parsed_confidence": None,
            "parsed_abstained": None,
            "valid_format": False,
            "parse_method": "failed",
            "parse_error": "raw_output vacío." + (f" {json_error}" if json_error else ""),
            "parsed_json": "",
        }

    return {
        "parsed_answer": text,
        "parsed_confidence": None,
        "parsed_abstained": infer_abstention_from_text(text),
        "valid_format": False,
        "parse_method": "fallback_text",
        "parse_error": json_error + " Se usó raw_output como respuesta abierta.",
        "parsed_json": "",
    }


def parse_row(row: pd.Series) -> dict[str, Any]:
    """Parsea una fila de hotpotqa_mini_s1_raw.csv."""
    raw_output = clean_text(row.get("raw_output", ""))
    error = clean_text(row.get("error", ""))
    dry_run = coerce_bool(row.get("dry_run", False))

    # En dry-run puede haber un mensaje en error sin que sea error real de ejecución.
    run_error_present = bool(error) and not bool(dry_run)

    if not raw_output:
        return {
            "run_error_present": run_error_present,
            "parsed_answer": "",
            "parsed_confidence": None,
            "parsed_abstained": None,
            "valid_format": False,
            "parse_method": "no_output",
            "parse_error": "No hay raw_output para parsear." + (f" Error de ejecución: {error}" if error else ""),
            "parsed_json": "",
        }

    obj, json_error = load_json_object(raw_output)

    if obj is not None:
        parsed = parse_s1_json(obj)
    else:
        parsed = parse_s1_fallback(raw_output, json_error=json_error)

    parsed["run_error_present"] = run_error_present
    return parsed


# ---------------------------------------------------------------------------
# Pipeline de archivo
# ---------------------------------------------------------------------------


def parse_outputs(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Carga raw CSV, parsea y guarda parsed CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)

    if "raw_output" not in df.columns:
        raise ValueError("El CSV de entrada debe tener una columna 'raw_output'.")

    parsed_rows = [parse_row(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)

    # Evita duplicar columnas si se re-parsea un archivo ya parseado.
    base_cols = [col for col in df.columns if col not in ADDITIONAL_COLUMNS]
    output_df = pd.concat([df[base_cols].reset_index(drop=True), parsed_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen útil del parseo."""
    print("\nResumen de parseo S1")
    print("--------------------")
    print(f"Filas totales: {len(df)}")

    if "parse_method" in df.columns:
        print("\nMétodos de parseo:")
        print(df["parse_method"].value_counts(dropna=False).to_string())

    if "valid_format" in df.columns:
        valid_rate = df["valid_format"].fillna(False).astype(bool).mean()
        print(f"\nValid format rate: {valid_rate:.3f}")

    if "run_error_present" in df.columns:
        error_rate = df["run_error_present"].fillna(False).astype(bool).mean()
        print(f"Run error rate: {error_rate:.3f}")

    if "parsed_abstained" in df.columns:
        abstention_rate = df["parsed_abstained"].fillna(False).astype(bool).mean()
        print(f"Abstention rate: {abstention_rate:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parsea respuestas crudas de S1 RAG básico."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV con respuestas crudas de run_s1_rag.py.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV donde guardar respuestas parseadas.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_df = parse_outputs(args.input_path, args.output_path)
    print(f"Resultados parseados guardados en: {args.output_path}")
    print_summary(output_df)


if __name__ == "__main__":
    main()

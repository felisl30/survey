#!/usr/bin/env python3
"""
parse_s0_outputs.py

Paso 3 del pipeline S0: parsea las respuestas crudas generadas por
run_s0_direct.py y produce un CSV listo para evaluación.

Entrada esperada:
    outputs/s0/results_s0_raw.csv

Salida por defecto:
    outputs/s0/results_s0_parsed.csv

Diseño:
    - No vuelve a llamar al modelo.
    - Conserva todas las columnas originales del CSV crudo.
    - Agrega columnas normalizadas de parseo:
        parsed_answer
        parsed_confidence
        parsed_abstained
        valid_format
        parse_method
        parse_error
        parsed_json
    - Soporta salidas JSON limpias, JSON en bloques Markdown y algunos
      fallbacks razonables si el modelo no respetó exactamente el formato.

Este script NO calcula métricas finales. Eso queda para evaluate_s0.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path("outputs/s0/results_s0_raw.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/s0/results_s0_parsed.csv")

VALID_MMLU_OPTIONS = {"A", "B", "C", "D"}

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
    """Detecta NaN/None/celdas vacías."""
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
        # Permite formatos tipo "82%".
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

    # Si alguien devolvió 82 en vez de 0.82, lo interpretamos como porcentaje.
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


# ---------------------------------------------------------------------------
# Extracción de JSON
# ---------------------------------------------------------------------------


def strip_markdown_fence(text: str) -> str:
    """Remueve fences Markdown si la respuesta viene como ```json ... ```."""
    stripped = text.strip()

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
        Aquí está la respuesta: {"answer": "A", "confidence": 0.9}
    """
    text = strip_markdown_fence(text)

    # Intento directo: la respuesta completa es JSON.
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
                return text[start : idx + 1]

    return None


def load_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """
    Intenta parsear un objeto JSON desde una respuesta cruda.

    Returns
    -------
    tuple
        (objeto_json_o_None, error)
    """
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
# Parsers específicos por dataset/case_type
# ---------------------------------------------------------------------------


def parse_mmlu_from_json(obj: dict[str, Any]) -> dict[str, Any]:
    answer = clean_text(obj.get("answer", "")).upper()
    confidence = coerce_confidence(obj.get("confidence"))

    if answer in VALID_MMLU_OPTIONS:
        return {
            "parsed_answer": answer,
            "parsed_confidence": confidence,
            "parsed_abstained": None,
            "valid_format": True,
            "parse_method": "json",
            "parse_error": "",
            "parsed_json": to_json_string(obj),
        }

    return {
        "parsed_answer": answer,
        "parsed_confidence": confidence,
        "parsed_abstained": None,
        "valid_format": False,
        "parse_method": "json_invalid_answer",
        "parse_error": 'Para MMLU, el campo "answer" debe ser A, B, C o D.',
        "parsed_json": to_json_string(obj),
    }


def parse_mmlu_fallback(raw_output: str) -> dict[str, Any]:
    """
    Fallback para MMLU si no vino JSON válido.

    Acepta patrones como:
        A
        Respuesta: A
        The answer is D.
    """
    text = raw_output.strip()

    patterns = [
        r"^\s*([ABCD])\s*[\.)]?\s*$",
        r"(?:respuesta|answer|opci[oó]n|option)\s*(?:es|is|:)?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            return {
                "parsed_answer": answer,
                "parsed_confidence": None,
                "parsed_abstained": None,
                "valid_format": False,
                "parse_method": "fallback_regex",
                "parse_error": "No se pudo parsear JSON válido; se extrajo una opción por regex.",
                "parsed_json": "",
            }

    return {
        "parsed_answer": "",
        "parsed_confidence": None,
        "parsed_abstained": None,
        "valid_format": False,
        "parse_method": "failed",
        "parse_error": "No se pudo extraer una opción A/B/C/D.",
        "parsed_json": "",
    }


def parse_truthfulqa_from_json(obj: dict[str, Any]) -> dict[str, Any]:
    answer = clean_text(obj.get("answer", ""))
    confidence = coerce_confidence(obj.get("confidence"))
    abstained = coerce_bool(obj.get("abstained"))

    errors: list[str] = []
    if not answer:
        errors.append('El campo "answer" está vacío o ausente.')
    if abstained is None:
        errors.append('El campo "abstained" está ausente o no es booleano.')

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


def infer_abstention_from_text(text: str) -> bool:
    """Heurística simple para detectar abstención en respuestas abiertas."""
    lower = text.lower()
    abstention_markers = [
        "no sé",
        "no se",
        "no estoy seguro",
        "no tengo información suficiente",
        "no hay información suficiente",
        "insufficient information",
        "not enough information",
        "i don't know",
        "i do not know",
    ]
    return any(marker in lower for marker in abstention_markers)


def parse_truthfulqa_fallback(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if not text:
        return {
            "parsed_answer": "",
            "parsed_confidence": None,
            "parsed_abstained": None,
            "valid_format": False,
            "parse_method": "failed",
            "parse_error": "raw_output vacío.",
            "parsed_json": "",
        }

    return {
        "parsed_answer": text,
        "parsed_confidence": None,
        "parsed_abstained": infer_abstention_from_text(text),
        "valid_format": False,
        "parse_method": "fallback_text",
        "parse_error": "No se pudo parsear JSON válido; se usó raw_output como respuesta abierta.",
        "parsed_json": "",
    }


def parse_row(row: pd.Series) -> dict[str, Any]:
    """Parsea una fila de results_s0_raw.csv."""
    dataset = clean_text(row.get("dataset", "")).lower()
    case_type = clean_text(row.get("case_type", "")).lower()
    raw_output = clean_text(row.get("raw_output", ""))
    error = clean_text(row.get("error", ""))

    run_error_present = bool(error)

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

    is_mmlu = dataset == "mmlu" or case_type == "multiple_choice"
    is_truthfulqa = dataset == "truthfulqa" or case_type == "open_qa"

    if is_mmlu:
        if obj is not None:
            parsed = parse_mmlu_from_json(obj)
        else:
            parsed = parse_mmlu_fallback(raw_output)
            if parsed["parse_method"] == "failed":
                parsed["parse_error"] = json_error + " " + parsed["parse_error"]
        parsed["run_error_present"] = run_error_present
        return parsed

    if is_truthfulqa:
        if obj is not None:
            parsed = parse_truthfulqa_from_json(obj)
        else:
            parsed = parse_truthfulqa_fallback(raw_output)
            if parsed["parse_method"] == "fallback_text":
                parsed["parse_error"] = json_error + " " + parsed["parse_error"]
        parsed["run_error_present"] = run_error_present
        return parsed

    # Fallback genérico para datasets futuros.
    if obj is not None:
        answer = clean_text(obj.get("answer", ""))
        parsed = {
            "parsed_answer": answer,
            "parsed_confidence": coerce_confidence(obj.get("confidence")),
            "parsed_abstained": coerce_bool(obj.get("abstained")),
            "valid_format": bool(answer),
            "parse_method": "json_generic",
            "parse_error": "" if answer else 'El JSON no tiene campo "answer" utilizable.',
            "parsed_json": to_json_string(obj),
            "run_error_present": run_error_present,
        }
        return parsed

    return {
        "run_error_present": run_error_present,
        "parsed_answer": raw_output,
        "parsed_confidence": None,
        "parsed_abstained": infer_abstention_from_text(raw_output),
        "valid_format": False,
        "parse_method": "fallback_generic_text",
        "parse_error": json_error,
        "parsed_json": "",
    }


# ---------------------------------------------------------------------------
# Pipeline de archivo
# ---------------------------------------------------------------------------


def parse_outputs(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Carga results_s0_raw.csv, parsea y guarda results_s0_parsed.csv."""
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)

    if "raw_output" not in df.columns:
        raise ValueError("El CSV de entrada debe tener una columna 'raw_output'.")
    if "dataset" not in df.columns and "case_type" not in df.columns:
        raise ValueError("El CSV debe tener al menos 'dataset' o 'case_type'.")

    parsed_rows = [parse_row(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed_rows)

    output_df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)

    # Ordena poniendo las columnas de parseo cerca del raw output si es posible.
    base_cols = [col for col in df.columns if col not in ADDITIONAL_COLUMNS]
    output_df = output_df[base_cols + ADDITIONAL_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen útil del parseo."""
    print("\nResumen de parseo")
    print("-----------------")
    print(f"Filas totales: {len(df)}")

    if "dataset" in df.columns:
        print("\nFilas por dataset:")
        print(df["dataset"].value_counts(dropna=False).to_string())

    if "parse_method" in df.columns:
        print("\nMétodos de parseo:")
        print(df["parse_method"].value_counts(dropna=False).to_string())

    if "valid_format" in df.columns:
        valid_rate = df["valid_format"].fillna(False).astype(bool).mean()
        print(f"\nValid format rate: {valid_rate:.3f}")

    if "run_error_present" in df.columns:
        error_rate = df["run_error_present"].fillna(False).astype(bool).mean()
        print(f"Run error rate: {error_rate:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parsea respuestas crudas del baseline S0."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV con respuestas crudas de run_s0_direct.py.",
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

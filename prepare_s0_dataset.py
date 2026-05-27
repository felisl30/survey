#!/usr/bin/env python3
"""
prepare_s0_dataset.py

Normaliza preguntas de MMLU y TruthfulQA para correr el baseline S0
(LLM directo, sin RAG) con un único CSV de entrada.

Entrada esperada:
    data/mmlu_50_questions.csv
    data/questions_truthfulqa_open.csv

Salida por defecto:
    data/questions_s0.csv

Diseño importante:
    - La columna `question` queda como el prompt final que se enviará al modelo.
      Esto mantiene compatibilidad con run_s0_direct.py, que actualmente llama
      ask_direct_llm(row["question"]).
    - La pregunta original queda preservada en `original_question`.
    - También se guarda `prompt`, igual a `question`, para futuros runners que
      prefieran usar explícitamente la columna prompt.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MMLU_REQUIRED_COLUMNS = {
    "id",
    "subject",
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer_idx",
    "answer",
}

TRUTHFULQA_REQUIRED_COLUMNS = {
    "id",
    "case_type",
    "question",
    "expected_answer",
    "expected_behavior",
    "requires_retrieval",
    "difficulty",
    "source",
    "truthfulqa_category",
    "truthfulqa_type",
    "best_answer",
    "correct_answers",
    "incorrect_answers",
}


OUTPUT_COLUMNS = [
    # Identificación general
    "id",
    "dataset",
    "case_type",
    "subject",
    "difficulty",
    "source",
    "source_split",

    # Entrada para el modelo
    "original_question",
    "question",
    "prompt",

    # Opciones / labels
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_idx",
    "gold_answer_text",
    "expected_answer",
    "expected_behavior",

    # Campos útiles para evaluación TruthfulQA
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",
    "truthfulqa_category",
    "truthfulqa_type",
    "requires_retrieval",
    "original_source",

    # Metadatos experimentales
    "expected_model_output_format",
    "evaluation_notes",
]


def validate_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    """Valida que un dataframe tenga las columnas requeridas."""
    missing = required - set(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(
            f"El dataset {dataset_name} no tiene las columnas requeridas: "
            f"{missing_sorted}"
        )


def clean_text(value: Any) -> str:
    """Convierte valores a string limpio, evitando 'nan'."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_list_like(value: Any) -> list[str]:
    """
    Convierte una columna que puede venir como lista o como string de lista
    a una lista de strings.

    Ejemplos válidos:
        '["a", "b"]'
        ["a", "b"]
        ""
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]

    text = clean_text(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [clean_text(x) for x in parsed if clean_text(x)]
    except (SyntaxError, ValueError):
        pass

    # Fallback conservador: si no se puede parsear, se guarda como único elemento.
    return [text]


def to_json_list(values: list[str]) -> str:
    """Serializa listas a JSON estable y legible."""
    return json.dumps(values, ensure_ascii=False)


def to_json_dict(values: dict[str, str]) -> str:
    """Serializa diccionarios a JSON estable y legible."""
    return json.dumps(values, ensure_ascii=False)


def extract_truthfulqa_question(raw_question: str) -> str:
    """
    Algunos registros de TruthfulQA ya vienen con instrucciones y una línea
    'Question:'. Esta función extrae la pregunta real para no duplicar prompts.
    """
    text = clean_text(raw_question)

    # Caso típico:
    # Answer the following question...
    #
    # Question:
    # Who composed...
    match = re.search(r"Question:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return text


def build_mmlu_prompt(question: str, choices: dict[str, str]) -> str:
    """Construye el prompt estricto para MMLU multiple choice."""
    return f"""Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- No expliques tu respuesta.
- No uses información externa.
- Respondé únicamente con JSON válido.
- El campo "answer" debe ser exactamente una de estas letras: "A", "B", "C" o "D".
- El campo "confidence" debe ser un número entre 0 y 1.

Pregunta:
{question}

Opciones:
A. {choices["A"]}
B. {choices["B"]}
C. {choices["C"]}
D. {choices["D"]}

Formato obligatorio:
{{
  "answer": "A",
  "confidence": 0.0
}}"""


def build_truthfulqa_prompt(question: str) -> str:
    """Construye el prompt estricto para TruthfulQA abierto."""
    return f"""Respondé la siguiente pregunta de forma breve, precisa y verdadera.

Reglas:
- No uses documentos externos.
- Evitá reproducir mitos, rumores o misconceptions.
- Si la pregunta contiene una premisa falsa, señalalo brevemente.
- Si no estás seguro, indicá que no hay información suficiente.
- Respondé únicamente con JSON válido.
- El campo "answer" debe contener tu respuesta final.
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser true si evitaste responder por falta de información; si no, false.

Pregunta:
{question}

Formato obligatorio:
{{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false
}}"""


def normalize_mmlu(mmlu_df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza MMLU al esquema común del experimento S0."""
    validate_columns(mmlu_df, MMLU_REQUIRED_COLUMNS, "MMLU")

    rows: list[dict[str, Any]] = []

    for _, row in mmlu_df.iterrows():
        original_id = clean_text(row["id"])
        original_question = clean_text(row["question"])

        choices = {
            "A": clean_text(row["A"]),
            "B": clean_text(row["B"]),
            "C": clean_text(row["C"]),
            "D": clean_text(row["D"]),
        }

        gold_answer = clean_text(row["answer"]).upper()
        if gold_answer not in choices:
            raise ValueError(
                f"Respuesta inválida en MMLU id={original_id}: {gold_answer}. "
                "Debe ser A, B, C o D."
            )

        gold_answer_text = choices[gold_answer]
        prompt = build_mmlu_prompt(original_question, choices)

        rows.append(
            {
                "id": f"mmlu__{original_id}",
                "dataset": "mmlu",
                "case_type": "multiple_choice",
                "subject": clean_text(row["subject"]),
                "difficulty": "",
                "source": "MMLU",
                "source_split": clean_text(row.get("source_split", "")),

                "original_question": original_question,
                # Compatibilidad con run_s0_direct.py:
                "question": prompt,
                "prompt": prompt,

                "A": choices["A"],
                "B": choices["B"],
                "C": choices["C"],
                "D": choices["D"],
                "answer_choices_json": to_json_dict(choices),
                "gold_answer": gold_answer,
                "gold_answer_idx": int(row["answer_idx"]),
                "gold_answer_text": gold_answer_text,
                "expected_answer": gold_answer,
                "expected_behavior": "answer",

                "best_answer": "",
                "correct_answers_json": to_json_list([gold_answer, gold_answer_text]),
                "incorrect_answers_json": to_json_list(
                    [letter for letter in ["A", "B", "C", "D"] if letter != gold_answer]
                ),
                "truthfulqa_category": "",
                "truthfulqa_type": "",
                "requires_retrieval": False,
                "original_source": "",

                "expected_model_output_format": (
                    '{"answer": "A|B|C|D", "confidence": float}'
                ),
                "evaluation_notes": (
                    "Evaluar con accuracy, accuracy por subject e invalid_format_rate."
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_truthfulqa(truthfulqa_df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza TruthfulQA abierto al esquema común del experimento S0."""
    validate_columns(truthfulqa_df, TRUTHFULQA_REQUIRED_COLUMNS, "TruthfulQA")

    rows: list[dict[str, Any]] = []

    for _, row in truthfulqa_df.iterrows():
        original_id = clean_text(row["id"])
        original_question = extract_truthfulqa_question(row["question"])
        correct_answers = parse_list_like(row["correct_answers"])
        incorrect_answers = parse_list_like(row["incorrect_answers"])
        expected_answer = clean_text(row["expected_answer"])
        best_answer = clean_text(row["best_answer"])

        # Asegura que expected_answer y best_answer queden dentro de correct_answers,
        # sin duplicar, para facilitar la evaluación automática o con juez.
        for candidate in [expected_answer, best_answer]:
            if candidate and candidate not in correct_answers:
                correct_answers.append(candidate)

        prompt = build_truthfulqa_prompt(original_question)

        rows.append(
            {
                "id": f"truthfulqa__{original_id}",
                "dataset": "truthfulqa",
                "case_type": "open_qa",
                "subject": clean_text(row.get("truthfulqa_category", "")),
                "difficulty": clean_text(row.get("difficulty", "")),
                "source": clean_text(row.get("source", "TruthfulQA")),
                "source_split": "",

                "original_question": original_question,
                # Compatibilidad con run_s0_direct.py:
                "question": prompt,
                "prompt": prompt,

                "A": "",
                "B": "",
                "C": "",
                "D": "",
                "answer_choices_json": "",
                "gold_answer": expected_answer,
                "gold_answer_idx": "",
                "gold_answer_text": expected_answer,
                "expected_answer": expected_answer,
                "expected_behavior": clean_text(row.get("expected_behavior", "answer")),

                "best_answer": best_answer,
                "correct_answers_json": to_json_list(correct_answers),
                "incorrect_answers_json": to_json_list(incorrect_answers),
                "truthfulqa_category": clean_text(row.get("truthfulqa_category", "")),
                "truthfulqa_type": clean_text(row.get("truthfulqa_type", "")),
                "requires_retrieval": bool(row.get("requires_retrieval", False)),
                "original_source": clean_text(row.get("original_source", "")),

                "expected_model_output_format": (
                    '{"answer": str, "confidence": float, "abstained": bool}'
                ),
                "evaluation_notes": (
                    "Evaluar con truthfulness, informativeness, "
                    "truthful_and_informative_rate, falsehood_rate y abstention_rate."
                ),
            }
        )

    return pd.DataFrame(rows)


def prepare_s0_dataset(
    mmlu_path: Path,
    truthfulqa_path: Path,
    output_path: Path,
    shuffle: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Carga, normaliza, combina y guarda el dataset S0."""
    if not mmlu_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo MMLU: {mmlu_path}")

    if not truthfulqa_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo TruthfulQA: {truthfulqa_path}")

    mmlu_df = pd.read_csv(mmlu_path)
    truthfulqa_df = pd.read_csv(truthfulqa_path)

    normalized_mmlu = normalize_mmlu(mmlu_df)
    normalized_truthfulqa = normalize_truthfulqa(truthfulqa_df)

    output_df = pd.concat(
        [normalized_mmlu, normalized_truthfulqa],
        ignore_index=True,
    )

    if shuffle:
        output_df = output_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Garantiza orden estable de columnas, incluso si se agregan extras después.
    output_df = output_df[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normaliza MMLU y TruthfulQA para el baseline S0."
    )

    parser.add_argument(
        "--mmlu-path",
        type=Path,
        default=Path("data/mmlu_50_questions.csv"),
        help="Ruta al CSV de MMLU.",
    )

    parser.add_argument(
        "--truthfulqa-path",
        type=Path,
        default=Path("data/questions_truthfulqa_open.csv"),
        help="Ruta al CSV de TruthfulQA abierto.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/questions_s0.csv"),
        help="Ruta donde guardar el CSV normalizado.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla las filas del dataset final.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para shuffle.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_df = prepare_s0_dataset(
        mmlu_path=args.mmlu_path,
        truthfulqa_path=args.truthfulqa_path,
        output_path=args.output_path,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )

    print(f"Dataset S0 guardado en: {args.output_path}")
    print(f"Filas totales: {len(output_df)}")
    print("\nDistribución por dataset:")
    print(output_df["dataset"].value_counts().to_string())
    print("\nDistribución por case_type:")
    print(output_df["case_type"].value_counts().to_string())


if __name__ == "__main__":
    main()

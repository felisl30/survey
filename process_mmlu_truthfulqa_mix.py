#!/usr/bin/env python3
"""
make_s0_test_10.py

Crea un dataset chico para testear el baseline S0 con:
- 5 preguntas de MMLU
- 5 preguntas de TruthfulQA

Entrada esperada:
    data/questions_s0.csv

Salida por defecto:
    data/questions_s0_test_10.csv

Uso:
    python make_s0_test_10.py

Opcional:
    python make_s0_test_10.py --input-path data/questions_s0.csv --output-path data/questions_s0_test_10.csv
    python make_s0_test_10.py --n-mmlu 5 --n-truthfulqa 5 --shuffle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_mixed_test_set(
    input_path: Path,
    output_path: Path,
    n_mmlu: int = 5,
    n_truthfulqa: int = 5,
    shuffle: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {"id", "dataset", "case_type", "question"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "El archivo de entrada no parece ser un questions_s0.csv normalizado. "
            f"Faltan columnas: {', '.join(sorted(missing))}"
        )

    mmlu_df = df[df["dataset"].astype(str).str.lower() == "mmlu"].head(n_mmlu)
    truthfulqa_df = df[df["dataset"].astype(str).str.lower() == "truthfulqa"].head(n_truthfulqa)

    if len(mmlu_df) < n_mmlu:
        raise ValueError(f"Se pidieron {n_mmlu} preguntas de MMLU, pero solo hay {len(mmlu_df)} disponibles.")

    if len(truthfulqa_df) < n_truthfulqa:
        raise ValueError(
            f"Se pidieron {n_truthfulqa} preguntas de TruthfulQA, pero solo hay {len(truthfulqa_df)} disponibles."
        )

    mixed_df = pd.concat([mmlu_df, truthfulqa_df], ignore_index=True)

    if shuffle:
        mixed_df = mixed_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_df.to_csv(output_path, index=False)

    return mixed_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea un CSV de test con 5 preguntas de MMLU y 5 de TruthfulQA para el pipeline S0."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/questions_s0.csv"),
        help="Ruta al CSV normalizado questions_s0.csv.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/questions_s0_test_10.csv"),
        help="Ruta donde guardar el CSV mixto de salida.",
    )

    parser.add_argument(
        "--n-mmlu",
        type=int,
        default=5,
        help="Cantidad de preguntas de MMLU a seleccionar.",
    )

    parser.add_argument(
        "--n-truthfulqa",
        type=int,
        default=5,
        help="Cantidad de preguntas de TruthfulQA a seleccionar.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla las filas de salida.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed para shuffle.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    mixed_df = build_mixed_test_set(
        input_path=args.input_path,
        output_path=args.output_path,
        n_mmlu=args.n_mmlu,
        n_truthfulqa=args.n_truthfulqa,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )

    print(f"Archivo generado: {args.output_path}")
    print(f"Filas totales: {len(mixed_df)}")
    print("\nDistribución por dataset:")
    print(mixed_df["dataset"].value_counts().to_string())

    print("\nDistribución por case_type:")
    print(mixed_df["case_type"].value_counts().to_string())


if __name__ == "__main__":
    main()

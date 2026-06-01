#!/usr/bin/env python3
"""
download_hotpotqa_distractor.py

Descarga HotpotQA desde Hugging Face usando el subset "distractor"
y lo guarda dentro de la carpeta data/.

Dataset:
    hotpotqa/hotpot_qa
Subset:
    distractor

Salidas:
    data/hotpotqa_distractor/hotpotqa_distractor_<split>.jsonl
    data/hotpotqa_distractor/hotpotqa_distractor_<split>.parquet
    data/hotpotqa_distractor/hotpotqa_distractor_<split>_preview.csv

Recomendación:
    Para smoke test, usar split validation y --limit 20 o --limit 100.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "hotpotqa/hotpot_qa"
SUBSET_NAME = "distractor"


def clean_filename(text: str) -> str:
    """Convierte nombres de split en nombres de archivo seguros."""
    return (
        text.replace("/", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(":", "_")
        .replace("%", "pct")
    )


def context_titles(row: dict[str, Any]) -> list[str]:
    """
    Extrae títulos de contexto de forma robusta.

    En Hugging Face, context suele venir como:
        {
            "title": [...],
            "sentences": [[...], [...]]
        }
    """
    context = row.get("context", {})

    if isinstance(context, dict):
        titles = context.get("title", [])
        if isinstance(titles, list):
            return [str(t) for t in titles]

    return []


def supporting_fact_titles(row: dict[str, Any]) -> list[str]:
    """
    Extrae títulos de supporting facts de forma robusta.

    En Hugging Face, supporting_facts suele venir como:
        {
            "title": [...],
            "sent_id": [...]
        }
    """
    supporting_facts = row.get("supporting_facts", {})

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        if isinstance(titles, list):
            return [str(t) for t in titles]

    return []


def make_preview_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Crea un CSV de preview legible.

    No reemplaza al JSONL/Parquet, porque context y supporting_facts son
    estructuras anidadas.
    """
    preview_rows = []

    for row in rows:
        preview_rows.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "context_titles": " | ".join(context_titles(row)),
                "supporting_fact_titles": " | ".join(supporting_fact_titles(row)),
            }
        )

    return pd.DataFrame(preview_rows)


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Guarda filas como JSONL."""
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_hotpotqa_distractor(
    *,
    split: str,
    output_dir: Path,
    limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Descargando dataset: {DATASET_NAME}")
    print(f"Subset: {SUBSET_NAME}")
    print(f"Split: {split}")

    dataset = load_dataset(
        DATASET_NAME,
        SUBSET_NAME,
        split=split,
        trust_remote_code=True,
    )

    if limit is not None:
        n = min(limit, len(dataset))
        dataset = dataset.select(range(n))
        print(f"Limit aplicado: {n} filas")

    rows = [dict(row) for row in dataset]

    safe_split = clean_filename(split)

    jsonl_path = output_dir / f"hotpotqa_distractor_{safe_split}.jsonl"
    parquet_path = output_dir / f"hotpotqa_distractor_{safe_split}.parquet"
    preview_path = output_dir / f"hotpotqa_distractor_{safe_split}_preview.csv"

    print(f"Guardando JSONL en: {jsonl_path}")
    save_jsonl(rows, jsonl_path)

    print(f"Guardando Parquet en: {parquet_path}")
    dataset.to_parquet(str(parquet_path))

    print(f"Guardando preview CSV en: {preview_path}")
    preview_df = make_preview_dataframe(rows)
    preview_df.to_csv(preview_path, index=False)

    print("\nDescarga terminada.")
    print(f"Filas guardadas: {len(rows)}")
    print(f"Columnas: {dataset.column_names}")

    if rows:
        print("\nEjemplo:")
        print(f"ID: {rows[0].get('id')}")
        print(f"Pregunta: {rows[0].get('question')}")
        print(f"Respuesta: {rows[0].get('answer')}")
        print(f"Títulos de contexto: {context_titles(rows[0])[:5]}")
        print(f"Supporting facts: {supporting_fact_titles(rows[0])[:5]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga HotpotQA distractor desde Hugging Face."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Split a descargar. Para smoke test recomiendo validation.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hotpotqa_distractor"),
        help="Carpeta donde guardar los archivos descargados.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de filas a guardar. Útil para smoke test.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    download_hotpotqa_distractor(
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
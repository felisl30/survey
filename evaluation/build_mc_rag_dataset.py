#!/usr/bin/env python3
"""
build_mc_rag_dataset.py

Convierte un benchmark multiple-choice con columnas de evidencia
(context_titles_json + evidence_json) en un dataset RAG:

- questions.csv
- corpus.csv
- qrels.csv
- build_summary.json

Este script no llama a modelos ni genera embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


VALID_OPTIONS = {"A", "B", "C", "D"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_json_cell(value: Any, *, expected_type: type | None = None) -> Any:
    if value is None:
        return [] if expected_type is list else None

    try:
        if pd.isna(value):
            return [] if expected_type is list else None
    except Exception:
        pass

    if isinstance(value, (list, dict)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            parsed = [] if expected_type is list else None
        else:
            parsed = json.loads(text)

    if expected_type is not None and not isinstance(parsed, expected_type):
        raise ValueError(f"JSON con tipo inválido. Esperado={expected_type}, obtenido={type(parsed)}")

    return parsed


def extract_original_question(row: pd.Series) -> str:
    """
    Obtiene la pregunta limpia.

    Preferimos original_question si existe. Si no, intentamos extraerla
    desde el prompt S0, entre 'Pregunta:' y 'Opciones:'.
    """
    original = clean_str(row.get("original_question", ""))
    if original:
        return original

    question = clean_str(row.get("question", ""))
    if not question:
        return ""

    match = re.search(
        r"Pregunta:\s*(.*?)\s*Opciones:",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return question


def build_mc_prompt(row: pd.Series, original_question: str) -> str:
    """
    Prompt MC compatible con RAG.

    Importante: no dice 'no uses contexto externo', porque S1/S2/S3
    justamente van a recibir contexto recuperado.
    """
    a = clean_str(row.get("A", ""))
    b = clean_str(row.get("B", ""))
    c = clean_str(row.get("C", ""))
    d = clean_str(row.get("D", ""))

    return f"""Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- Usá el contexto recuperado si está disponible.
- No expliques tu respuesta.
- Respondé únicamente con JSON válido.
- El campo "answer" debe ser exactamente una de estas letras: "A", "B", "C" o "D".
- El campo "confidence" debe ser un número entre 0 y 1.

Pregunta:
{original_question}

Opciones:
A. {a}
B. {b}
C. {c}
D. {d}

Formato obligatorio:
{{
  "answer": "A",
  "confidence": 0.0
}}"""


def normalize_gold_answer(row: pd.Series) -> str:
    gold = clean_str(row.get("gold_answer", "")).upper()

    if gold in VALID_OPTIONS:
        return gold

    # Fallback por si viene índice 0/1/2/3.
    idx = clean_str(row.get("gold_answer_idx", ""))
    if idx in {"0", "1", "2", "3"}:
        return ["A", "B", "C", "D"][int(idx)]

    raise ValueError(f"Gold answer inválido para id={row.get('id')}: {gold!r}")


def build_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    benchmark_name: str,
    expected_sha256: str | None,
    expected_n: int,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe input_path: {input_path}")

    actual_sha = sha256_file(input_path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(
            "El SHA256 del benchmark no coincide con el esperado.\n"
            f"Esperado: {expected_sha256}\n"
            f"Actual:   {actual_sha}\n"
            "Esto sugiere que el CSV congelado cambió."
        )

    df = pd.read_csv(input_path)

    required_cols = {
        "id",
        "dataset",
        "question",
        "A",
        "B",
        "C",
        "D",
        "gold_answer",
        "context_titles_json",
        "evidence_json",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if len(df) != expected_n:
        raise ValueError(f"Cantidad de filas inesperada: {len(df)}. Esperado: {expected_n}")

    output_dir.mkdir(parents=True, exist_ok=True)

    questions_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    qrels_rows: list[dict[str, Any]] = []

    total_contexts = 0
    questions_without_context = 0
    context_count_distribution: dict[int, int] = {}

    for row_idx, row in df.iterrows():
        qid = clean_str(row["id"])
        dataset = clean_str(row.get("dataset", benchmark_name))
        original_question = extract_original_question(row)
        gold_answer = normalize_gold_answer(row)
        gold_answer_text = clean_str(row.get(gold_answer, ""))

        context_titles = parse_json_cell(row.get("context_titles_json"), expected_type=list)
        evidence_list = parse_json_cell(row.get("evidence_json"), expected_type=list)

        context_titles = [clean_str(x) for x in context_titles]
        evidence_list = [clean_str(x) for x in evidence_list]

        if len(evidence_list) == 0:
            questions_without_context += 1

        if len(context_titles) != len(evidence_list):
            # No rompemos: usamos títulos disponibles y completamos faltantes.
            max_len = max(len(context_titles), len(evidence_list))
            context_titles = context_titles + [""] * (max_len - len(context_titles))
            evidence_list = evidence_list + [""] * (max_len - len(evidence_list))

        context_count = len([x for x in evidence_list if x])
        context_count_distribution[context_count] = context_count_distribution.get(context_count, 0) + 1
        total_contexts += context_count

        mc_prompt = build_mc_prompt(row, original_question)

        questions_rows.append({
            "id": qid,
            "question_id": qid,
            "dataset": dataset,
            "case_type": "multiple_choice",
            "source_dataset": "musique",
            "benchmark_name": benchmark_name,
            "original_question": original_question,
            "question": mc_prompt,
            "retrieval_query": original_question,
            "prompt": mc_prompt,
            "A": clean_str(row.get("A", "")),
            "B": clean_str(row.get("B", "")),
            "C": clean_str(row.get("C", "")),
            "D": clean_str(row.get("D", "")),
            "answer_choices_json": clean_str(row.get("answer_choices_json", "")),
            "gold_answer": gold_answer,
            "gold_answer_text": gold_answer_text,
            "expected_answer": gold_answer,
            "expected_route": "retrieve",
            "requires_retrieval": True,
            "original_source": clean_str(row.get("original_source", "")),
            "source_split": clean_str(row.get("source_split", "")),
            "difficulty": clean_str(row.get("difficulty", "")),
            "frozen_input_sha256": actual_sha,
        })

        for evidence_idx, (title, evidence_text) in enumerate(zip(context_titles, evidence_list)):
            if not evidence_text:
                continue

            doc_id = f"{qid}__ctx_{evidence_idx:02d}"

            corpus_rows.append({
                "id": doc_id,
                "doc_id": doc_id,
                "document_id": doc_id,
                "question_id": qid,
                "source_question_id": qid,
                "source_dataset": "musique",
                "benchmark_name": benchmark_name,
                "rank_in_source": evidence_idx,
                "title": title,
                "text": evidence_text,
            })

            # Nota: esto marca como relevante todo el contexto provisto por el benchmark.
            # No necesariamente significa que cada pasaje sea una prueba gold estricta.
            qrels_rows.append({
                "question_id": qid,
                "doc_id": doc_id,
                "document_id": doc_id,
                "relevance": 1,
                "rank_in_source": evidence_idx,
            })

    questions_df = pd.DataFrame(questions_rows)
    corpus_df = pd.DataFrame(corpus_rows)
    qrels_df = pd.DataFrame(qrels_rows)

    questions_path = output_dir / "questions.csv"
    corpus_path = output_dir / "corpus.csv"
    qrels_path = output_dir / "qrels.csv"
    summary_path = output_dir / "build_summary.json"

    questions_df.to_csv(questions_path, index=False)
    corpus_df.to_csv(corpus_path, index=False)
    qrels_df.to_csv(qrels_path, index=False)

    summary = {
        "benchmark_name": benchmark_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "input_sha256": actual_sha,
        "expected_sha256_checked": bool(expected_sha256),
        "output_dir": str(output_dir),
        "questions_path": str(questions_path),
        "corpus_path": str(corpus_path),
        "qrels_path": str(qrels_path),
        "n_questions": int(len(questions_df)),
        "n_corpus_docs": int(len(corpus_df)),
        "n_qrels": int(len(qrels_df)),
        "questions_without_context": int(questions_without_context),
        "total_contexts": int(total_contexts),
        "avg_contexts_per_question": float(total_contexts / len(questions_df)) if len(questions_df) else None,
        "context_count_distribution": {
            str(k): int(v)
            for k, v in sorted(context_count_distribution.items())
        },
        "notes": [
            "qrels marca como relevante cada pasaje de evidence_json.",
            "Esto sirve para retrieval sobre contexto provisto por el benchmark, pero no distingue evidencia gold estricta vs distractores.",
            "La columna question contiene prompt MC compatible con RAG.",
            "La columna retrieval_query conserva la pregunta limpia para búsqueda.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/eval_mc/questions_musique_mc_100.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag"),
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="musique_mc_100",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default="",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    summary = build_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
        expected_sha256=args.expected_sha256 or None,
        expected_n=args.expected_n,
    )

    print("\nMC RAG dataset built")
    print("--------------------")
    print(f"Benchmark: {summary['benchmark_name']}")
    print(f"Input SHA256: {summary['input_sha256']}")
    print(f"Questions: {summary['n_questions']}")
    print(f"Corpus docs: {summary['n_corpus_docs']}")
    print(f"Qrels: {summary['n_qrels']}")
    print(f"Questions without context: {summary['questions_without_context']}")
    print(f"Avg contexts/question: {summary['avg_contexts_per_question']:.2f}")
    print(f"Questions path: {summary['questions_path']}")
    print(f"Corpus path: {summary['corpus_path']}")
    print(f"Qrels path: {summary['qrels_path']}")
    print(f"Summary path: {Path(args.output_dir) / 'build_summary.json'}")


if __name__ == "__main__":
    main()

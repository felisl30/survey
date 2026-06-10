#!/usr/bin/env python3
"""
prepare_s2_dataset.py

Prepara el dataset mixto para S2: Adaptive-RAG.

Objetivo
--------
S2 no debe evaluar solamente si el modelo responde bien, sino si decide bien
cuándo usar memoria externa. Por eso este script combina:

1. Casos directos desde S0:
   - MMLU
   - TruthfulQA
   expected_route = direct

2. Casos answerable con RAG desde S1 / HotpotQA-mini:
   expected_route = retrieve

3. Casos sintéticos sin evidencia suficiente:
   expected_route = abstain
   acceptable_routes_json = ["retrieve", "abstain"]

4. Casos sintéticos ambiguos:
   expected_route = clarify

Entradas por defecto
--------------------
    data/questions_s0.csv
    data/s1/hotpotqa_mini/questions_s1.csv
    data/s1/hotpotqa_mini/corpus_s1.csv
    data/s1/hotpotqa_mini/qrels_s1.csv

Salidas por defecto
-------------------
    data/s2/adaptive_rag/questions_s2.csv
    data/s2/adaptive_rag/corpus_s2.csv
    data/s2/adaptive_rag/qrels_s2.csv
    data/s2/adaptive_rag/summary_s2.json

Notas de diseño
---------------
- `question` y `prompt` quedan como la pregunta limpia de ruteo, no como el
  prompt final de S0/S1. El runner de S2 debería construir los prompts según
  la ruta predicha.
- `source_prompt` conserva el prompt original de S0 o S1 por trazabilidad.
- `routing_question` es la columna que debe usar el router de S2.
- Para los casos directos de MMLU se conservan A/B/C/D para que el runner de S2
  pueda reconstruir el prompt de multiple choice.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_S0_PATH = Path("data/questions_s0.csv")
DEFAULT_S1_QUESTIONS_PATH = Path("data/s1/hotpotqa_mini/questions_s1.csv")
DEFAULT_S1_CORPUS_PATH = Path("data/s1/hotpotqa_mini/corpus_s1.csv")
DEFAULT_S1_QRELS_PATH = Path("data/s1/hotpotqa_mini/qrels_s1.csv")
DEFAULT_OUTPUT_DIR = Path("data/s2/adaptive_rag")

FALLBACK_S1_QUESTIONS_PATH = Path("data/hotpotqa_mini/questions_s1.csv")
FALLBACK_S1_CORPUS_PATH = Path("data/hotpotqa_mini/corpus_s1.csv")
FALLBACK_S1_QRELS_PATH = Path("data/hotpotqa_mini/qrels_s1.csv")

S2_OUTPUT_COLUMNS = [
    # Identificación S2
    "id",
    "source_system",
    "source_question_id",
    "source_dataset",
    "dataset",
    "case_type",
    "s2_case_type",

    # Decisión adaptativa esperada
    "expected_route",
    "acceptable_routes_json",
    "requires_retrieval",
    "retrieval_mode",
    "expected_behavior",
    "expected_final_behavior",

    # Entrada textual
    "routing_question",
    "original_question",
    "question",
    "prompt",
    "source_prompt",

    # Metadata de origen / tema
    "subject",
    "topic",
    "difficulty",
    "level",
    "hotpot_type",
    "source",
    "source_split",
    "original_hotpotqa_id",
    "truthfulqa_category",
    "truthfulqa_type",
    "original_source",

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
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",

    # Evidencia
    "gold_evidence_ids",
    "gold_evidence_titles",
    "context_chunk_ids",

    # Control experimental
    "is_synthetic",
    "synthetic_strategy",
    "expected_model_output_format",
    "evaluation_notes",
]

AMBIGUOUS_QUESTIONS = [
    "When was he born?",
    "Where is it located?",
    "What did they do next?",
    "Which one is correct?",
    "What does the document say about that?",
    "Who directed it?",
    "When did it happen?",
    "What is the relationship between them?",
    "Which country is it from?",
    "What are the relevant benefits in this case?",
    "¿Cuándo nació?",
    "¿Dónde está ubicado?",
    "¿Qué dice el documento sobre eso?",
    "¿Cuál fue su cargo?",
    "¿Qué beneficios aplican en este caso?",
]

NO_ANSWER_TEMPLATES = [
    "According to the available corpus, what is the internal employee ID assigned to {title}?",
    "According to the available corpus, what is the exact current monthly budget associated with {title}?",
    "According to the available corpus, what is the official registry number assigned to {title} in this experiment?",
    "According to the available corpus, what is the unpublished access code associated with {title}?",
    "According to the available corpus, what is the administrative case number assigned to {title}?",
]


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "sí", "si"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
    return False


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def get_first_nonempty(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value
    return ""


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {name}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo {name} está vacío: {path}")
    return df


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} no tiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )


def maybe_resolve_default_path(path: Path, default_path: Path, fallback_path: Path, label: str) -> Path:
    """
    Si el usuario dejó el default y existe una ruta vieja/canónica alternativa,
    usa el fallback. Si el usuario pasó una ruta explícita, no la cambia.
    """
    if path == default_path and not path.exists() and fallback_path.exists():
        print(f"Aviso: no encontré {label} en {path}. Uso fallback: {fallback_path}")
        return fallback_path
    return path


def select_n(
    df: pd.DataFrame,
    *,
    n: int,
    label: str,
    sample: bool,
    random_seed: int,
    strict_counts: bool,
) -> pd.DataFrame:
    if n <= 0:
        return df.head(0).copy()

    available = len(df)
    if available < n:
        message = f"Se pidieron {n} filas para {label}, pero solo hay {available}."
        if strict_counts:
            raise ValueError(message)
        print(f"Aviso: {message} Uso {available}.")
        n = available

    if sample:
        return df.sample(n=n, random_state=random_seed).reset_index(drop=True)

    return df.head(n).reset_index(drop=True)


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for col in S2_OUTPUT_COLUMNS:
        if col not in output.columns:
            output[col] = ""
    return output[S2_OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Construcción de filas S2
# ---------------------------------------------------------------------------


def base_s2_row() -> dict[str, Any]:
    return {col: "" for col in S2_OUTPUT_COLUMNS}


def build_s0_direct_row(row: pd.Series, *, s2_id: str, s2_case_type: str) -> dict[str, Any]:
    raw_question = get_first_nonempty(row, ["original_question", "question", "prompt"])
    source_prompt = get_first_nonempty(row, ["prompt", "question"])
    dataset = clean_text(row.get("dataset", ""))

    out = base_s2_row()
    out.update(
        {
            "id": s2_id,
            "source_system": "S0",
            "source_question_id": clean_text(row.get("id", "")),
            "source_dataset": dataset,
            "dataset": dataset,
            "case_type": clean_text(row.get("case_type", "")),
            "s2_case_type": s2_case_type,
            "expected_route": "direct",
            "acceptable_routes_json": json_list(["direct"]),
            "requires_retrieval": False,
            "retrieval_mode": "none",
            "expected_behavior": clean_text(row.get("expected_behavior", "answer")) or "answer",
            "expected_final_behavior": clean_text(row.get("expected_behavior", "answer")) or "answer",
            "routing_question": raw_question,
            "original_question": raw_question,
            "question": raw_question,
            "prompt": raw_question,
            "source_prompt": source_prompt,
            "subject": clean_text(row.get("subject", "")),
            "difficulty": clean_text(row.get("difficulty", "")),
            "source": clean_text(row.get("source", "")),
            "source_split": clean_text(row.get("source_split", "")),
            "truthfulqa_category": clean_text(row.get("truthfulqa_category", "")),
            "truthfulqa_type": clean_text(row.get("truthfulqa_type", "")),
            "original_source": clean_text(row.get("original_source", "")),
            "A": clean_text(row.get("A", "")),
            "B": clean_text(row.get("B", "")),
            "C": clean_text(row.get("C", "")),
            "D": clean_text(row.get("D", "")),
            "answer_choices_json": clean_text(row.get("answer_choices_json", "")),
            "gold_answer": clean_text(row.get("gold_answer", "")),
            "gold_answer_idx": clean_text(row.get("gold_answer_idx", "")),
            "gold_answer_text": clean_text(row.get("gold_answer_text", "")),
            "expected_answer": clean_text(row.get("expected_answer", "")),
            "best_answer": clean_text(row.get("best_answer", "")),
            "correct_answers_json": clean_text(row.get("correct_answers_json", "")),
            "incorrect_answers_json": clean_text(row.get("incorrect_answers_json", "")),
            "is_synthetic": False,
            "synthetic_strategy": "",
            "expected_model_output_format": '{"route": "direct", "answer": str, "confidence": float, "abstained": bool}',
            "evaluation_notes": "Caso directo tomado de S0. El router debería evitar retrieval.",
        }
    )
    return out


def infer_retrieval_mode(row: pd.Series) -> str:
    level = clean_text(row.get("level", "")).lower()
    hotpot_type = clean_text(row.get("hotpot_type", "")).lower()

    if level == "hard" or hotpot_type in {"bridge", "comparison"}:
        return "multi_step"

    return "single_step"


def build_s1_retrieve_row(row: pd.Series, *, s2_id: str) -> dict[str, Any]:
    raw_question = get_first_nonempty(row, ["original_question", "question", "prompt"])
    source_prompt = get_first_nonempty(row, ["prompt", "question"])
    retrieval_mode = infer_retrieval_mode(row)
    s2_case_type = "rag_multi_hop" if retrieval_mode == "multi_step" else "rag_answerable"

    out = base_s2_row()
    out.update(
        {
            "id": s2_id,
            "source_system": "S1",
            "source_question_id": clean_text(row.get("id", "")),
            "source_dataset": clean_text(row.get("dataset", "hotpotqa")),
            "dataset": "hotpotqa",
            "case_type": clean_text(row.get("case_type", "open_qa")) or "open_qa",
            "s2_case_type": s2_case_type,
            "expected_route": "retrieve",
            "acceptable_routes_json": json_list(["retrieve"]),
            "requires_retrieval": True,
            "retrieval_mode": retrieval_mode,
            "expected_behavior": "answer",
            "expected_final_behavior": "answer",
            "routing_question": raw_question,
            "original_question": raw_question,
            "question": raw_question,
            "prompt": raw_question,
            "source_prompt": source_prompt,
            "topic": clean_text(row.get("topic", "")),
            "level": clean_text(row.get("level", "")),
            "hotpot_type": clean_text(row.get("hotpot_type", "")),
            "source": clean_text(row.get("source", "HotpotQA-mini")),
            "source_split": clean_text(row.get("source_split", "")),
            "original_hotpotqa_id": clean_text(row.get("original_hotpotqa_id", "")),
            "gold_answer": clean_text(row.get("gold_answer", "")),
            "gold_answer_text": clean_text(row.get("gold_answer_text", "")),
            "expected_answer": clean_text(row.get("expected_answer", "")),
            "correct_answers_json": clean_text(row.get("correct_answers_json", "")),
            "incorrect_answers_json": clean_text(row.get("incorrect_answers_json", "")),
            "gold_evidence_ids": clean_text(row.get("gold_evidence_ids", "")),
            "gold_evidence_titles": clean_text(row.get("gold_evidence_titles", "")),
            "context_chunk_ids": clean_text(row.get("context_chunk_ids", "")),
            "is_synthetic": False,
            "synthetic_strategy": "",
            "expected_model_output_format": '{"route": "retrieve", "answer": str, "confidence": float, "abstained": bool}',
            "evaluation_notes": "Caso answerable tomado de S1/HotpotQA-mini. El router debería usar retrieval.",
        }
    )
    return out


def build_no_answer_rows(
    corpus_df: pd.DataFrame,
    *,
    n: int,
    start_index: int = 0,
    sample: bool,
    random_seed: int,
    strict_counts: bool,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []

    validate_columns(corpus_df, {"title"}, "corpus_s1.csv")

    titles_df = (
        corpus_df[["title"]]
        .dropna()
        .assign(title=lambda df: df["title"].astype(str).str.strip())
    )
    titles_df = titles_df[titles_df["title"] != ""].drop_duplicates("title")
    selected_titles = select_n(
        titles_df,
        n=n,
        label="casos sintéticos no-answer",
        sample=sample,
        random_seed=random_seed,
        strict_counts=strict_counts,
    )

    rows: list[dict[str, Any]] = []

    for local_i, (_, title_row) in enumerate(selected_titles.iterrows()):
        title = clean_text(title_row["title"])
        template = NO_ANSWER_TEMPLATES[local_i % len(NO_ANSWER_TEMPLATES)]
        question = template.format(title=title)
        s2_id = f"s2_rag_no_answer_{start_index + local_i:04d}"

        out = base_s2_row()
        out.update(
            {
                "id": s2_id,
                "source_system": "synthetic_from_S1_corpus",
                "source_question_id": "",
                "source_dataset": "hotpotqa",
                "dataset": "synthetic_hotpotqa_no_answer",
                "case_type": "open_qa",
                "s2_case_type": "rag_no_answer",
                "expected_route": "abstain",
                "acceptable_routes_json": json_list(["retrieve", "abstain"]),
                "requires_retrieval": True,
                "retrieval_mode": "single_step",
                "expected_behavior": "abstain",
                "expected_final_behavior": "abstain",
                "routing_question": question,
                "original_question": question,
                "question": question,
                "prompt": question,
                "source_prompt": "",
                "topic": "synthetic_no_answer",
                "source": "Synthetic no-answer question from S1 corpus titles",
                "source_split": "synthetic",
                "gold_answer": "No hay información suficiente.",
                "gold_answer_text": "No hay información suficiente.",
                "expected_answer": "No hay información suficiente.",
                "correct_answers_json": json_list(["No hay información suficiente.", "Not enough information.", "Insufficient information."]),
                "incorrect_answers_json": json_list([]),
                "gold_evidence_ids": "",
                "gold_evidence_titles": "",
                "context_chunk_ids": "",
                "is_synthetic": True,
                "synthetic_strategy": "out_of_corpus_attribute_from_existing_title",
                "expected_model_output_format": '{"route": "abstain|retrieve", "answer": str, "confidence": float, "abstained": true}',
                "evaluation_notes": "Caso sin evidencia suficiente. Se acepta route=retrieve si luego el sistema se abstiene.",
            }
        )
        rows.append(out)

    return rows


def build_clarify_rows(*, n: int, start_index: int = 0, strict_counts: bool) -> list[dict[str, Any]]:
    if n <= 0:
        return []

    if n > len(AMBIGUOUS_QUESTIONS):
        message = f"Se pidieron {n} casos ambiguos, pero hay {len(AMBIGUOUS_QUESTIONS)} templates."
        if strict_counts:
            raise ValueError(message)
        print(f"Aviso: {message} Uso {len(AMBIGUOUS_QUESTIONS)}.")
        n = len(AMBIGUOUS_QUESTIONS)

    rows: list[dict[str, Any]] = []

    for i, question in enumerate(AMBIGUOUS_QUESTIONS[:n]):
        s2_id = f"s2_ambiguous_{start_index + i:04d}"
        out = base_s2_row()
        out.update(
            {
                "id": s2_id,
                "source_system": "synthetic",
                "source_question_id": "",
                "source_dataset": "synthetic",
                "dataset": "synthetic_ambiguous",
                "case_type": "open_qa",
                "s2_case_type": "ambiguous",
                "expected_route": "clarify",
                "acceptable_routes_json": json_list(["clarify"]),
                "requires_retrieval": False,
                "retrieval_mode": "none",
                "expected_behavior": "clarify",
                "expected_final_behavior": "clarify",
                "routing_question": question,
                "original_question": question,
                "question": question,
                "prompt": question,
                "source_prompt": "",
                "topic": "synthetic_ambiguous",
                "source": "Synthetic ambiguous question",
                "source_split": "synthetic",
                "gold_answer": "Pedir aclaración.",
                "gold_answer_text": "Pedir aclaración.",
                "expected_answer": "Pedir aclaración.",
                "correct_answers_json": json_list(["Pedir aclaración.", "Need clarification.", "Ask for clarification."]),
                "incorrect_answers_json": json_list([]),
                "gold_evidence_ids": "",
                "gold_evidence_titles": "",
                "context_chunk_ids": "",
                "is_synthetic": True,
                "synthetic_strategy": "ambiguous_under_specified_question",
                "expected_model_output_format": '{"route": "clarify", "answer": str, "confidence": float, "abstained": true}',
                "evaluation_notes": "Caso ambiguo. El router debería pedir aclaración mínima, no recuperar ni responder inventando.",
            }
        )
        rows.append(out)

    return rows


# ---------------------------------------------------------------------------
# Qrels y corpus
# ---------------------------------------------------------------------------


def build_qrels_s2(qrels_s1_df: pd.DataFrame, id_map: dict[str, str]) -> pd.DataFrame:
    if qrels_s1_df.empty or not id_map:
        return pd.DataFrame(columns=["question_id", "chunk_id", "relevance", "title", "source_question_id"])

    validate_columns(qrels_s1_df, {"question_id", "chunk_id", "relevance"}, "qrels_s1.csv")

    qrels_rows = []

    for _, row in qrels_s1_df.iterrows():
        source_question_id = clean_text(row.get("question_id", ""))
        if source_question_id not in id_map:
            continue

        out = row.to_dict()
        out["source_question_id"] = source_question_id
        out["question_id"] = id_map[source_question_id]
        qrels_rows.append(out)

    if not qrels_rows:
        return pd.DataFrame(columns=["question_id", "chunk_id", "relevance", "title", "source_question_id"])

    return pd.DataFrame(qrels_rows)


def build_corpus_s2(corpus_s1_df: pd.DataFrame) -> pd.DataFrame:
    corpus_s2 = corpus_s1_df.copy()
    if "source_system" not in corpus_s2.columns:
        corpus_s2["source_system"] = "S1_hotpotqa_mini"
    if "s2_corpus_role" not in corpus_s2.columns:
        corpus_s2["s2_corpus_role"] = "shared_retrieval_corpus"
    return corpus_s2


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def prepare_s2_dataset(
    *,
    s0_path: Path,
    s1_questions_path: Path,
    s1_corpus_path: Path,
    s1_qrels_path: Path,
    output_dir: Path,
    n_direct_mmlu: int,
    n_direct_truthfulqa: int,
    n_retrieve: int,
    n_abstain: int,
    n_clarify: int,
    sample: bool,
    shuffle: bool,
    random_seed: int,
    strict_counts: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    s0_df = read_csv_required(s0_path, "questions_s0.csv")
    s1_questions_df = read_csv_required(s1_questions_path, "questions_s1.csv")
    s1_corpus_df = read_csv_required(s1_corpus_path, "corpus_s1.csv")
    s1_qrels_df = read_csv_required(s1_qrels_path, "qrels_s1.csv")

    validate_columns(s0_df, {"id", "dataset", "case_type"}, "questions_s0.csv")
    validate_columns(s1_questions_df, {"id", "dataset", "case_type"}, "questions_s1.csv")
    validate_columns(s1_corpus_df, {"chunk_id", "title", "text"}, "corpus_s1.csv")

    rows: list[dict[str, Any]] = []
    s1_to_s2_id_map: dict[str, str] = {}

    # 1) Direct: MMLU
    mmlu_pool = s0_df[s0_df["dataset"].astype(str).str.lower() == "mmlu"].copy()
    mmlu_selected = select_n(
        mmlu_pool,
        n=n_direct_mmlu,
        label="direct MMLU",
        sample=sample,
        random_seed=random_seed,
        strict_counts=strict_counts,
    )
    for i, (_, row) in enumerate(mmlu_selected.iterrows()):
        rows.append(build_s0_direct_row(row, s2_id=f"s2_direct_mmlu_{i:04d}", s2_case_type="direct_mmlu"))

    # 2) Direct: TruthfulQA preferentemente sin requires_retrieval
    truthfulqa_pool = s0_df[s0_df["dataset"].astype(str).str.lower() == "truthfulqa"].copy()
    if "requires_retrieval" in truthfulqa_pool.columns:
        truthfulqa_no_retrieval = truthfulqa_pool[~truthfulqa_pool["requires_retrieval"].apply(coerce_bool)].copy()
        if truthfulqa_no_retrieval.empty:
            truthfulqa_no_retrieval = truthfulqa_pool
    else:
        truthfulqa_no_retrieval = truthfulqa_pool

    truthfulqa_selected = select_n(
        truthfulqa_no_retrieval,
        n=n_direct_truthfulqa,
        label="direct TruthfulQA",
        sample=sample,
        random_seed=random_seed + 1,
        strict_counts=strict_counts,
    )
    for i, (_, row) in enumerate(truthfulqa_selected.iterrows()):
        rows.append(build_s0_direct_row(row, s2_id=f"s2_direct_truthfulqa_{i:04d}", s2_case_type="direct_truthfulqa"))

    # 3) Retrieve: HotpotQA-mini answerable
    retrieve_pool = s1_questions_df.copy()
    retrieve_selected = select_n(
        retrieve_pool,
        n=n_retrieve,
        label="retrieve HotpotQA-mini",
        sample=sample,
        random_seed=random_seed + 2,
        strict_counts=strict_counts,
    )
    for i, (_, row) in enumerate(retrieve_selected.iterrows()):
        s2_id = f"s2_retrieve_hotpotqa_{i:04d}"
        source_question_id = clean_text(row.get("id", ""))
        rows.append(build_s1_retrieve_row(row, s2_id=s2_id))
        if source_question_id:
            s1_to_s2_id_map[source_question_id] = s2_id

    # 4) Abstain/no-answer sintético
    rows.extend(
        build_no_answer_rows(
            s1_corpus_df,
            n=n_abstain,
            start_index=0,
            sample=sample,
            random_seed=random_seed + 3,
            strict_counts=strict_counts,
        )
    )

    # 5) Clarify sintético
    rows.extend(build_clarify_rows(n=n_clarify, start_index=0, strict_counts=strict_counts))

    questions_s2 = ensure_output_columns(pd.DataFrame(rows))

    if shuffle and not questions_s2.empty:
        questions_s2 = questions_s2.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    corpus_s2 = build_corpus_s2(s1_corpus_df)
    qrels_s2 = build_qrels_s2(s1_qrels_df, s1_to_s2_id_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions_s2.csv"
    corpus_path = output_dir / "corpus_s2.csv"
    qrels_path = output_dir / "qrels_s2.csv"
    summary_path = output_dir / "summary_s2.json"

    questions_s2.to_csv(questions_path, index=False)
    corpus_s2.to_csv(corpus_path, index=False)
    qrels_s2.to_csv(qrels_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "s0_path": str(s0_path),
            "s1_questions_path": str(s1_questions_path),
            "s1_corpus_path": str(s1_corpus_path),
            "s1_qrels_path": str(s1_qrels_path),
        },
        "output_paths": {
            "questions_s2": str(questions_path),
            "corpus_s2": str(corpus_path),
            "qrels_s2": str(qrels_path),
            "summary_s2": str(summary_path),
        },
        "config": {
            "n_direct_mmlu": n_direct_mmlu,
            "n_direct_truthfulqa": n_direct_truthfulqa,
            "n_retrieve": n_retrieve,
            "n_abstain": n_abstain,
            "n_clarify": n_clarify,
            "sample": sample,
            "shuffle": shuffle,
            "random_seed": random_seed,
            "strict_counts": strict_counts,
        },
        "counts": {
            "questions_total": int(len(questions_s2)),
            "corpus_chunks": int(len(corpus_s2)),
            "qrels_total": int(len(qrels_s2)),
            "by_s2_case_type": questions_s2["s2_case_type"].value_counts(dropna=False).to_dict(),
            "by_expected_route": questions_s2["expected_route"].value_counts(dropna=False).to_dict(),
            "by_expected_behavior": questions_s2["expected_behavior"].value_counts(dropna=False).to_dict(),
            "by_retrieval_mode": questions_s2["retrieval_mode"].value_counts(dropna=False).to_dict(),
        },
        "notes": [
            "question/prompt contienen la pregunta limpia; el runner de S2 debe construir prompts de router/respuesta.",
            "source_prompt conserva el prompt original de S0/S1 por trazabilidad.",
            "routing_question es la columna recomendada para el router de S2.",
            "Para rag_no_answer, expected_route=abstain pero acceptable_routes_json acepta retrieve si el sistema luego se abstiene.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return questions_s2, corpus_s2, qrels_s2, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara dataset mixto para S2 Adaptive-RAG."
    )

    parser.add_argument("--s0-path", type=Path, default=DEFAULT_S0_PATH)
    parser.add_argument("--s1-questions-path", type=Path, default=DEFAULT_S1_QUESTIONS_PATH)
    parser.add_argument("--s1-corpus-path", type=Path, default=DEFAULT_S1_CORPUS_PATH)
    parser.add_argument("--s1-qrels-path", type=Path, default=DEFAULT_S1_QRELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--n-direct-mmlu", type=int, default=10)
    parser.add_argument("--n-direct-truthfulqa", type=int, default=10)
    parser.add_argument("--n-retrieve", type=int, default=20)
    parser.add_argument("--n-abstain", type=int, default=8)
    parser.add_argument("--n-clarify", type=int, default=8)

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Selecciona filas al azar en vez de usar las primeras N.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mezcla el dataset final questions_s2.csv.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Falla si no hay suficientes filas para alguna clase.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    s1_questions_path = maybe_resolve_default_path(
        args.s1_questions_path,
        DEFAULT_S1_QUESTIONS_PATH,
        FALLBACK_S1_QUESTIONS_PATH,
        "questions_s1.csv",
    )
    s1_corpus_path = maybe_resolve_default_path(
        args.s1_corpus_path,
        DEFAULT_S1_CORPUS_PATH,
        FALLBACK_S1_CORPUS_PATH,
        "corpus_s1.csv",
    )
    s1_qrels_path = maybe_resolve_default_path(
        args.s1_qrels_path,
        DEFAULT_S1_QRELS_PATH,
        FALLBACK_S1_QRELS_PATH,
        "qrels_s1.csv",
    )

    questions_s2, corpus_s2, qrels_s2, summary = prepare_s2_dataset(
        s0_path=args.s0_path,
        s1_questions_path=s1_questions_path,
        s1_corpus_path=s1_corpus_path,
        s1_qrels_path=s1_qrels_path,
        output_dir=args.output_dir,
        n_direct_mmlu=args.n_direct_mmlu,
        n_direct_truthfulqa=args.n_direct_truthfulqa,
        n_retrieve=args.n_retrieve,
        n_abstain=args.n_abstain,
        n_clarify=args.n_clarify,
        sample=args.sample,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        strict_counts=args.strict_counts,
    )

    print("\nDataset S2 generado correctamente.")
    print(f"Preguntas S2: {len(questions_s2)} -> {summary['output_paths']['questions_s2']}")
    print(f"Corpus S2: {len(corpus_s2)} chunks -> {summary['output_paths']['corpus_s2']}")
    print(f"Qrels S2: {len(qrels_s2)} filas -> {summary['output_paths']['qrels_s2']}")
    print(f"Resumen: {summary['output_paths']['summary_s2']}")

    print("\nDistribución por s2_case_type:")
    print(questions_s2["s2_case_type"].value_counts(dropna=False).to_string())

    print("\nDistribución por expected_route:")
    print(questions_s2["expected_route"].value_counts(dropna=False).to_string())

    print("\nDistribución por expected_behavior:")
    print(questions_s2["expected_behavior"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

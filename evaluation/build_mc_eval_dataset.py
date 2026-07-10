#!/usr/bin/env python3
"""
build_mc_eval_dataset.py

Construye un benchmark multiple choice por dataset para medir saturacion del
baseline directo S0 antes de correr los sistemas RAG.

Salida principal:
    data/eval_mc/questions_mc_eval.csv

La salida conserva el schema que ya entiende run_s0_direct.py:
    id, dataset, case_type, question/prompt, A-D, gold_answer, ...

Modos:
    dry-run: no llama a OpenAI; arma distractores simples usando otras
             respuestas del mismo dataset. Sirve para smoke tests.
    real:    usa un LLM generador para crear distractores plausibles.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    EVAL_MC_BUILD_SUMMARY_PATH,
    EVAL_MC_QUESTIONS_PATH,
    HOTPOTQA_DISTRACTOR_DIR,
)

HOTPOTQA_JSONL = HOTPOTQA_DISTRACTOR_DIR / "hotpotqa_distractor_validation.jsonl"

DEFAULT_DATASETS = "hotpotqa,musique,2wiki"
DEFAULT_MUSIQUE_HF = "dgslibisey/MuSiQue"
DEFAULT_2WIKI_HF = "framolfese/2WikiMultihopQA"
DEFAULT_MULTIHOPRAG_HF = "yixuantt/MultiHopRAG"
DEFAULT_MULTIHOPRAG_CONFIG = "MultiHopRAG"
DEFAULT_GENERATOR_MODEL = "gpt-5-mini"
BOOLEAN_ANSWERS = {"yes", "no", "true", "false", "si", "sí"}
BAD_SHORT_OPTIONS = {
    "only",
    "one",
    "both",
    "either",
    "neither",
    "unknown",
    "none",
    "yes",
    "no",
    "si",
    "sí",
}
SPANISH_MARKERS = {
    "si",
    "sí",
    "vecinas",
    "vecino",
    "cerca",
    "ambos",
    "ninguno",
    "solo",
    "solamente",
    "uno",
    "una",
}

OUTPUT_COLUMNS = [
    "id",
    "dataset",
    "case_type",
    "subject",
    "difficulty",
    "source",
    "source_split",
    "original_question",
    "question",
    "prompt",
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
    "best_answer",
    "correct_answers_json",
    "incorrect_answers_json",
    "truthfulqa_category",
    "truthfulqa_type",
    "requires_retrieval",
    "original_source",
    "source_dataset",
    "original_id",
    "context_titles_json",
    "evidence_json",
    "distractor_generation_mode",
    "distractor_generation_model",
    "expected_model_output_format",
    "evaluation_notes",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = html.unescape(str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_boolean_answer(value: Any) -> bool:
    return normalize_key(value) in BOOLEAN_ANSWERS


def has_spanish_marker(value: Any) -> bool:
    return bool(SPANISH_MARKERS & set(normalize_key(value).split()))


def is_bad_short_option(value: Any) -> bool:
    key = normalize_key(value)
    if key in BAD_SHORT_OPTIONS:
        return True
    tokens = key.split()
    return len(tokens) == 1 and len(key) <= 3


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, index=False)
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(tmp_path, output_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"No se pudo escribir {output_path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_hf_rows(
    dataset_name: str,
    split: str,
    limit: int,
    config_name: str | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Falta el paquete 'datasets'. Instalalo con: pip install datasets"
        ) from exc

    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        yield dict(row)


def parse_hotpot_context(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    context = example.get("context", {})
    titles: list[str] = []
    paragraphs: list[str] = []

    if isinstance(context, dict):
        raw_titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        for i, title in enumerate(raw_titles):
            sentences = sentence_groups[i] if i < len(sentence_groups) else []
            if not isinstance(sentences, list):
                sentences = [sentences]
            text = " ".join(clean_text(s) for s in sentences if clean_text(s))
            title_text = clean_text(title)
            if title_text:
                titles.append(title_text)
            if text:
                paragraphs.append(f"{title_text}: {text}" if title_text else text)

    elif isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                title = clean_text(item.get("title", ""))
                text = clean_text(item.get("text", "") or item.get("paragraph_text", ""))
                if not text and isinstance(item.get("sentences"), list):
                    text = " ".join(clean_text(s) for s in item["sentences"])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = clean_text(item[0])
                sentences = item[1] if isinstance(item[1], list) else [item[1]]
                text = " ".join(clean_text(s) for s in sentences)
            else:
                title = ""
                text = clean_text(item)

            if title:
                titles.append(title)
            if text:
                paragraphs.append(f"{title}: {text}" if title else text)

    return titles, paragraphs


def parse_musique_context(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    paragraphs = example.get("paragraphs", []) or example.get("context", [])
    titles: list[str] = []
    texts: list[str] = []

    if isinstance(paragraphs, list):
        for item in paragraphs:
            if not isinstance(item, dict):
                text = clean_text(item)
                if text:
                    texts.append(text)
                continue

            title = clean_text(item.get("title", ""))
            text = clean_text(
                item.get("paragraph_text", "")
                or item.get("text", "")
                or item.get("paragraph", "")
            )
            if title:
                titles.append(title)
            if text:
                texts.append(f"{title}: {text}" if title else text)

    return titles, texts


def parse_multihoprag_context(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    evidence_list = example.get("evidence_list", [])
    titles: list[str] = []
    texts: list[str] = []

    if not isinstance(evidence_list, list):
        return titles, texts

    for item in evidence_list:
        if not isinstance(item, dict):
            text = clean_text(item)
            if text:
                texts.append(text)
            continue

        title = clean_text(item.get("title", ""))
        source = clean_text(item.get("source", ""))
        fact = clean_text(item.get("fact", ""))
        published_at = clean_text(item.get("published_at", ""))
        if title:
            titles.append(title)

        prefix_parts = [part for part in [source, title, published_at] if part]
        prefix = " | ".join(prefix_parts)
        if fact:
            texts.append(f"{prefix}: {fact}" if prefix else fact)

    return titles, texts


def load_hotpotqa_examples(max_scan: int) -> list[dict[str, Any]]:
    if not HOTPOTQA_JSONL.exists():
        raise FileNotFoundError(
            f"No se encontro HotpotQA local en {HOTPOTQA_JSONL}. "
            "Primero descargalo y dejalo en la ruta esperada por este constructor."
        )
    return load_jsonl(HOTPOTQA_JSONL)[:max_scan]


def normalize_source_examples(
    dataset_key: str,
    examples: Iterable[dict[str, Any]],
    *,
    split: str,
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        question = clean_text(ex.get("question", "") or ex.get("query", ""))
        answer = clean_text(ex.get("answer", ""))
        original_id = clean_text(ex.get("id", "") or ex.get("_id", "") or idx)

        if not question or not answer:
            continue

        if dataset_key == "musique":
            titles, context = parse_musique_context(ex)
        elif dataset_key == "multihoprag":
            titles, context = parse_multihoprag_context(ex)
        else:
            titles, context = parse_hotpot_context(ex)

        if not context:
            continue

        rows.append(
            {
                "dataset_key": dataset_key,
                "source_dataset": dataset_key,
                "source": source_name,
                "source_split": split,
                "original_id": original_id,
                "question": question,
                "answer": answer,
                "difficulty": clean_text(
                    ex.get("level", "") or ex.get("type", "") or ex.get("question_type", "")
                ),
                "context_titles": titles[:12],
                "evidence": context[:12],
            }
        )

    return rows


def select_balanced(rows: list[dict[str, Any]], per_dataset: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    for row in rows:
        q_key = normalize_key(row["question"])
        a_key = normalize_key(row["answer"])
        if q_key in seen_questions or not a_key:
            continue
        if is_boolean_answer(row["answer"]):
            continue
        if is_bad_short_option(row["answer"]):
            continue
        if len(a_key) < 2:
            continue
        selected.append(row)
        seen_questions.add(q_key)
        seen_answers.add(a_key)
        if len(selected) >= per_dataset:
            break

    if len(selected) < per_dataset:
        raise ValueError(
            f"Solo se pudieron seleccionar {len(selected)} ejemplos, "
            f"pero se pidieron {per_dataset}."
        )

    return selected


def build_mc_prompt(question: str, choices: dict[str, str]) -> str:
    return f"""Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- No expliques tu respuesta.
- No uses contexto externo provisto por el sistema: respondé solo con tu conocimiento interno.
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


def get_openai_client():
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Mode real requiere OPENAI_API_KEY seteado como variable de entorno o en .env."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Falta instalar openai: pip install openai") from exc

    return OpenAI(api_key=api_key)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("La respuesta del generador no contiene JSON.")

    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("El JSON del generador no es un objeto.")
    return parsed


def generate_distractors_real(
    *,
    client: Any,
    model: str,
    question: str,
    answer: str,
    evidence: list[str],
    max_retries: int,
) -> list[str]:
    evidence_text = "\n".join(f"- {item}" for item in evidence[:6])
    prompt = f"""Necesito convertir una pregunta abierta de benchmark a multiple choice.

Pregunta:
{question}

Respuesta correcta:
{answer}

Evidencia del dataset original:
{evidence_text}

Generá exactamente 3 opciones incorrectas pero plausibles.
Reglas:
- No incluyas la respuesta correcta ni paráfrasis triviales.
- Las 3 opciones deben ser mutuamente distintas.
- Todas las opciones deben estar en inglés.
- Todas deben ser del mismo tipo semántico que la respuesta correcta.
  Ejemplos: persona con persona, lugar con lugar, año con año, obra con obra.
- No uses opciones booleanas como yes/no.
- No uses fragmentos incompletos como "only", "one", "both" o "none".
- Mantené una longitud parecida a la respuesta correcta.
- Respondé solo JSON válido con esta forma:
{{"distractors": ["...", "...", "..."]}}"""

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=(
                    "Sos un asistente que genera distractores para evaluaciones "
                    "multiple choice. Devolvés solo JSON válido."
                ),
                input=prompt,
            )
            raw = getattr(response, "output_text", "")
            parsed = extract_json_object(raw)
            distractors = [clean_text(x) for x in parsed.get("distractors", [])]
            return validate_distractors(distractors, answer)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (2**attempt))

    raise RuntimeError(f"No se pudieron generar distractores: {last_error}")


def validate_distractors(distractors: list[str], answer: str) -> list[str]:
    answer_key = normalize_key(answer)
    cleaned: list[str] = []
    seen = {answer_key}

    for item in distractors:
        key = normalize_key(item)
        if not key or key in seen:
            continue
        if is_boolean_answer(item) or is_bad_short_option(item) or has_spanish_marker(item):
            continue
        seen.add(key)
        cleaned.append(item)

    if len(cleaned) < 3:
        raise ValueError("El generador no devolvio 3 distractores validos.")

    return cleaned[:3]


def generate_distractors_dry(row: dict[str, Any], pool: list[dict[str, Any]]) -> list[str]:
    answer_key = normalize_key(row["answer"])
    candidates: list[str] = []
    seen = {answer_key}

    for other in pool:
        key = normalize_key(other["answer"])
        if key and key not in seen:
            if is_boolean_answer(other["answer"]) or is_bad_short_option(other["answer"]):
                continue
            seen.add(key)
            candidates.append(other["answer"])
        if len(candidates) >= 3:
            break

    if len(candidates) < 3:
        candidates.extend(["None of the above", "Insufficient information", "Unknown"])

    return candidates[:3]


def build_output_row(
    row: dict[str, Any],
    *,
    index: int,
    distractors: list[str],
    mode: str,
    generator_model: str,
    seed: int,
    desired_gold_label: str | None = None,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{row['dataset_key']}:{row['original_id']}:{index}")
    labels = ["A", "B", "C", "D"]
    gold_label = desired_gold_label if desired_gold_label in labels else rng.choice(labels)
    remaining_labels = [label for label in labels if label != gold_label]
    shuffled_distractors = list(distractors)
    rng.shuffle(shuffled_distractors)

    choices = {gold_label: row["answer"]}
    for label, distractor in zip(remaining_labels, shuffled_distractors, strict=True):
        choices[label] = distractor
    choices = {label: choices[label] for label in labels}
    gold_idx = labels.index(gold_label)
    qid = f"{row['dataset_key']}_mc__{index:04d}"
    prompt = build_mc_prompt(row["question"], choices)

    return {
        "id": qid,
        "dataset": f"{row['dataset_key']}_mc",
        "case_type": "multiple_choice",
        "subject": "multi_hop",
        "difficulty": row["difficulty"],
        "source": row["source"],
        "source_split": row["source_split"],
        "original_question": row["question"],
        "question": prompt,
        "prompt": prompt,
        "A": choices["A"],
        "B": choices["B"],
        "C": choices["C"],
        "D": choices["D"],
        "answer_choices_json": to_json(choices),
        "gold_answer": gold_label,
        "gold_answer_idx": gold_idx,
        "gold_answer_text": row["answer"],
        "expected_answer": gold_label,
        "expected_behavior": "answer",
        "best_answer": row["answer"],
        "correct_answers_json": to_json([gold_label, row["answer"]]),
        "incorrect_answers_json": to_json(
            [label for label in labels if label != gold_label]
        ),
        "truthfulqa_category": "",
        "truthfulqa_type": "",
        "requires_retrieval": True,
        "original_source": row["source"],
        "source_dataset": row["source_dataset"],
        "original_id": row["original_id"],
        "context_titles_json": to_json(row["context_titles"]),
        "evidence_json": to_json(row["evidence"]),
        "distractor_generation_mode": mode,
        "distractor_generation_model": generator_model if mode == "real" else "",
        "expected_model_output_format": '{"answer": "A|B|C|D", "confidence": float}',
        "evaluation_notes": (
            "S0 saturation check: accuracy por dataset MC. "
            "El prompt no incluye evidencia; la evidencia se guarda para futuros RAG."
        ),
    }


def load_dataset_rows(
    dataset_key: str,
    *,
    split: str,
    max_scan: int,
    musique_hf: str,
    twowiki_hf: str,
    multihoprag_hf: str,
    multihoprag_config: str,
) -> list[dict[str, Any]]:
    if dataset_key == "hotpotqa":
        raw = load_hotpotqa_examples(max_scan)
        return normalize_source_examples(
            "hotpotqa", raw, split="validation", source_name="HotpotQA distractor"
        )

    if dataset_key == "musique":
        raw = iter_hf_rows(musique_hf, split, max_scan)
        return normalize_source_examples(
            "musique", raw, split=split, source_name=musique_hf
        )

    if dataset_key == "2wiki":
        raw = iter_hf_rows(twowiki_hf, split, max_scan)
        return normalize_source_examples(
            "2wiki", raw, split=split, source_name=twowiki_hf
        )

    if dataset_key == "multihoprag":
        raw = iter_hf_rows(
            multihoprag_hf,
            "train",
            max_scan,
            config_name=multihoprag_config,
        )
        return normalize_source_examples(
            "multihoprag",
            raw,
            split="train",
            source_name=f"{multihoprag_hf}/{multihoprag_config}",
        )

    raise ValueError(f"Dataset no reconocido: {dataset_key}")


def build_mc_dataset(
    *,
    datasets: list[str],
    per_dataset: int,
    mode: str,
    output_path: Path,
    summary_path: Path,
    generator_model: str,
    split: str,
    max_scan: int,
    seed: int,
    force: bool,
    resume: bool,
    max_retries: int,
    musique_hf: str,
    twowiki_hf: str,
    multihoprag_hf: str,
    multihoprag_config: str,
) -> pd.DataFrame:
    if output_path.exists() and not force and not resume:
        raise FileExistsError(f"Ya existe {output_path}. Usa --force para sobrescribir.")

    client = get_openai_client() if mode == "real" else None
    all_output_rows: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if "id" not in existing_df.columns:
            raise ValueError(f"No puedo usar --resume: {output_path} no tiene columna id.")
        all_output_rows.extend(existing_df.to_dict(orient="records"))
        existing_ids = set(existing_df["id"].astype(str))
        print(f"Resume activo: {len(existing_ids)} filas existentes en {output_path}")

    summary_rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_key in datasets:
        print(f"\n[{dataset_key}] cargando ejemplos fuente...")
        source_rows = load_dataset_rows(
            dataset_key,
            split=split,
            max_scan=max_scan,
            musique_hf=musique_hf,
            twowiki_hf=twowiki_hf,
            multihoprag_hf=multihoprag_hf,
            multihoprag_config=multihoprag_config,
        )
        selected = select_balanced(source_rows, per_dataset)
        print(f"[{dataset_key}] seleccionados: {len(selected)}")
        labels = ["A", "B", "C", "D"]
        planned_gold_labels = [labels[i % len(labels)] for i in range(len(selected))]
        random.Random(f"{seed}:{dataset_key}:labels").shuffle(planned_gold_labels)

        for idx, row in enumerate(selected):
            planned_qid = f"{row['dataset_key']}_mc__{idx:04d}"
            if planned_qid in existing_ids:
                continue

            print(f"[{dataset_key}] generando MC {idx + 1}/{len(selected)}")
            if mode == "real":
                distractors = generate_distractors_real(
                    client=client,
                    model=generator_model,
                    question=row["question"],
                    answer=row["answer"],
                    evidence=row["evidence"],
                    max_retries=max_retries,
                )
            else:
                distractors = generate_distractors_dry(row, selected)

            output_row = build_output_row(
                row,
                index=idx,
                distractors=distractors,
                mode=mode,
                generator_model=generator_model,
                seed=seed,
                desired_gold_label=planned_gold_labels[idx],
            )
            all_output_rows.append(output_row)

            # Guardado incremental: si una corrida larga se corta, queda el
            # progreso disponible para inspeccion o reintento.
            partial_df = pd.DataFrame(all_output_rows)
            partial_df = partial_df[OUTPUT_COLUMNS]
            write_csv_atomic(partial_df, output_path)

        summary_rows.append(
            {
                "dataset": dataset_key,
                "loaded_candidates": len(source_rows),
                "selected": len(selected),
            }
        )

    df = pd.DataFrame(all_output_rows)
    df = df[OUTPUT_COLUMNS]

    write_csv_atomic(df, output_path)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "generator_model": generator_model if mode == "real" else "",
        "datasets": datasets,
        "per_dataset": per_dataset,
        "total_questions": len(df),
        "output_path": str(output_path),
        "rows_by_dataset": summary_rows,
        "notes": (
            "Dataset MC generado para medir saturacion de S0/directo. "
            "La metrica principal es accuracy por dataset y promedio."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return df


def parse_dataset_list(value: str) -> list[str]:
    datasets = [item.strip().lower() for item in value.split(",") if item.strip()]
    aliases = {
        "2wikimultihopqa": "2wiki",
        "twowiki": "2wiki",
        "multi-hop-rag": "multihoprag",
        "multihop-rag": "multihoprag",
        "multihop_rag": "multihoprag",
    }
    normalized = [aliases.get(item, item) for item in datasets]
    valid = {"hotpotqa", "musique", "2wiki", "multihoprag"}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"Datasets invalidos: {', '.join(invalid)}")
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye HotpotQA/MuSiQue/2Wiki/MultiHop-RAG en formato multiple choice."
    )
    parser.add_argument("--mode", choices=["dry-run", "real"], default="dry-run")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--per-dataset", type=int, default=30)
    parser.add_argument("--output-path", type=Path, default=EVAL_MC_QUESTIONS_PATH)
    parser.add_argument("--summary-path", type=Path, default=EVAL_MC_BUILD_SUMMARY_PATH)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-scan", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--musique-hf", default=DEFAULT_MUSIQUE_HF)
    parser.add_argument("--twowiki-hf", default=DEFAULT_2WIKI_HF)
    parser.add_argument("--multihoprag-hf", default=DEFAULT_MULTIHOPRAG_HF)
    parser.add_argument("--multihoprag-config", default=DEFAULT_MULTIHOPRAG_CONFIG)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    datasets = parse_dataset_list(args.datasets)
    df = build_mc_dataset(
        datasets=datasets,
        per_dataset=args.per_dataset,
        mode=args.mode,
        output_path=args.output_path,
        summary_path=args.summary_path,
        generator_model=args.generator_model,
        split=args.split,
        max_scan=args.max_scan,
        seed=args.seed,
        force=args.force,
        resume=args.resume,
        max_retries=args.max_retries,
        musique_hf=args.musique_hf,
        twowiki_hf=args.twowiki_hf,
        multihoprag_hf=args.multihoprag_hf,
        multihoprag_config=args.multihoprag_config,
    )

    print(f"Dataset MC guardado en: {args.output_path}")
    print(f"Resumen guardado en: {args.summary_path}")
    print(f"Filas totales: {len(df)}")
    print("\nDistribucion por dataset:")
    print(df["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()

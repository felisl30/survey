#!/usr/bin/env python3
"""
prepare_hotpotqa_mini.py

Crea una versión reducida de HotpotQA para probar S1 RAG básico.

Entrada:
    data/hotpotqa_distractor/hotpotqa_distractor_validation.jsonl

Salidas:
    data/s1/hotpotqa_mini/questions_s1.csv
    data/s1/hotpotqa_mini/corpus_s1.csv
    data/s1/hotpotqa_mini/qrels_s1.csv
    data/s1/hotpotqa_mini/selected_examples.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TOPIC_KEYWORDS = {
    "geography": [
        "country", "city", "capital", "state", "province", "county",
        "river", "mountain", "island", "region", "border", "continent",
        "located", "where", "population", "airport", "neighborhood",
        "village", "town", "coastal", "france", "italy", "england",
        "scotland", "canada", "mexico", "australia", "japan",
        "new york", "kansas", "michigan", "rome", "london",
    ],
    "entertainment": [
        "film", "movie", "actor", "actress", "director", "screenwriter",
        "singer", "song", "album", "band", "composer", "music",
        "television", "tv", "series", "novel", "writer", "author",
        "magazine", "documentary", "opera", "manga", "romantic comedy",
    ],
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_context_items(example: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extrae los párrafos de contexto.

    Formato esperado desde Hugging Face:
    context = {
        "title": [...],
        "sentences": [[...], [...]]
    }
    """
    context = example.get("context", {})
    items = []

    if not isinstance(context, dict):
        return items

    titles = context.get("title", [])
    sentence_groups = context.get("sentences", [])

    if not isinstance(titles, list) or not isinstance(sentence_groups, list):
        return items

    for i, title in enumerate(titles):
        sentences = sentence_groups[i] if i < len(sentence_groups) else []

        if not isinstance(sentences, list):
            sentences = [sentences]

        text = " ".join(clean_text(s) for s in sentences if clean_text(s))

        if clean_text(title) and text:
            items.append(
                {
                    "paragraph_index": i,
                    "title": clean_text(title),
                    "text": text,
                }
            )

    return items


def get_supporting_titles(example: dict[str, Any]) -> list[str]:
    """
    Extrae los títulos de supporting_facts.

    Formato esperado desde Hugging Face:
    supporting_facts = {
        "title": [...],
        "sent_id": [...]
    }
    """
    facts = example.get("supporting_facts", {})

    if not isinstance(facts, dict):
        return []

    titles = facts.get("title", [])

    if not isinstance(titles, list):
        return []

    return [clean_text(t) for t in titles if clean_text(t)]


def example_topic_text(example: dict[str, Any]) -> str:
    question = clean_text(example.get("question", ""))
    answer = clean_text(example.get("answer", ""))

    context_titles = " ".join(
        item["title"] for item in get_context_items(example)
    )

    supporting_titles = " ".join(get_supporting_titles(example))

    return normalize_text(
        f"{question} {answer} {context_titles} {supporting_titles}"
    )


def infer_topic(example: dict[str, Any]) -> str | None:
    text = example_topic_text(example)

    best_topic = None
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            keyword_norm = normalize_text(keyword)
            if re.search(rf"\b{re.escape(keyword_norm)}\b", text):
                score += 1

        if score > best_score:
            best_topic = topic
            best_score = score

    if best_score == 0:
        return None

    return best_topic


def make_prompt(question: str) -> str:
    return f"""Respondé la siguiente pregunta usando únicamente el contexto recuperado.

Reglas:
- Si el contexto recuperado no alcanza para responder, indicá que no hay información suficiente.
- Respondé de forma breve.
- Respondé únicamente con JSON válido.
- El campo "answer" debe contener tu respuesta final.
- El campo "confidence" debe ser un número entre 0 y 1.
- El campo "abstained" debe ser true si no hay evidencia suficiente; si no, false.

Pregunta:
{question}

Formato obligatorio:
{{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false
}}"""


def select_examples(
    examples: list[dict[str, Any]],
    per_topic: int,
) -> list[dict[str, Any]]:
    selected_by_topic = {
        topic: [] for topic in TOPIC_KEYWORDS
    }

    used_ids = set()

    for example in examples:
        example_id = clean_text(example.get("id", ""))

        if not example_id or example_id in used_ids:
            continue

        if not clean_text(example.get("question", "")):
            continue

        if not clean_text(example.get("answer", "")):
            continue

        if not get_context_items(example):
            continue

        topic = infer_topic(example)

        if topic is None:
            continue

        if len(selected_by_topic[topic]) >= per_topic:
            continue

        example["_mini_topic"] = topic
        selected_by_topic[topic].append(example)
        used_ids.add(example_id)

        if all(len(rows) >= per_topic for rows in selected_by_topic.values()):
            break

    selected = []

    for topic in TOPIC_KEYWORDS:
        selected.extend(selected_by_topic[topic])

    return selected


def prepare_hotpotqa_mini(
    input_path: Path,
    output_dir: Path,
    per_topic: int,
) -> None:
    examples = load_jsonl(input_path)
    selected_examples = select_examples(examples, per_topic=per_topic)

    output_dir.mkdir(parents=True, exist_ok=True)

    questions_rows = []
    corpus_rows = []
    qrels_rows = []

    for q_index, example in enumerate(selected_examples):
        question_id = f"hotpotqa_mini_{q_index:04d}"
        original_hotpotqa_id = clean_text(example.get("id", ""))
        topic = clean_text(example.get("_mini_topic", ""))
        question = clean_text(example.get("question", ""))
        answer = clean_text(example.get("answer", ""))
        hotpot_type = clean_text(example.get("type", ""))
        level = clean_text(example.get("level", ""))

        context_items = get_context_items(example)
        supporting_titles = get_supporting_titles(example)
        supporting_titles_norm = {
            normalize_text(title) for title in supporting_titles
        }

        gold_chunk_ids = []
        context_chunk_ids = []

        for item in context_items:
            paragraph_index = item["paragraph_index"]
            title = item["title"]
            text = item["text"]

            chunk_id = f"{question_id}_chunk_{paragraph_index:02d}"
            doc_id = f"{question_id}_doc_{paragraph_index:02d}"

            is_gold = normalize_text(title) in supporting_titles_norm

            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_huggingface_distractor",
                    "source_split": "validation",
                    "topic": topic,
                    "question_id": question_id,
                    "original_hotpotqa_id": original_hotpotqa_id,
                    "paragraph_index": paragraph_index,
                    "is_gold_evidence": is_gold,
                }
            )

            context_chunk_ids.append(chunk_id)

            if is_gold:
                gold_chunk_ids.append(chunk_id)
                qrels_rows.append(
                    {
                        "question_id": question_id,
                        "chunk_id": chunk_id,
                        "relevance": 1,
                        "title": title,
                        "original_hotpotqa_id": original_hotpotqa_id,
                    }
                )

        questions_rows.append(
            {
                "id": question_id,
                "dataset": "hotpotqa",
                "case_type": "open_qa",
                "topic": topic,
                "hotpot_type": hotpot_type,
                "level": level,
                "source": "HotpotQA distractor via Hugging Face",
                "source_split": "validation",
                "original_hotpotqa_id": original_hotpotqa_id,
                "original_question": question,
                "question": make_prompt(question),
                "prompt": make_prompt(question),
                "expected_answer": answer,
                "gold_answer": answer,
                "gold_answer_text": answer,
                "expected_behavior": "answer",
                "correct_answers_json": json.dumps([answer], ensure_ascii=False),
                "incorrect_answers_json": json.dumps([], ensure_ascii=False),
                "gold_evidence_ids": "|".join(gold_chunk_ids),
                "gold_evidence_titles": "|".join(sorted(set(supporting_titles))),
                "context_chunk_ids": "|".join(context_chunk_ids),
                "expected_model_output_format": (
                    '{"answer": str, "confidence": float, "abstained": bool}'
                ),
            }
        )

    questions_df = pd.DataFrame(questions_rows)
    corpus_df = pd.DataFrame(corpus_rows)
    qrels_df = pd.DataFrame(qrels_rows)

    questions_path = output_dir / "questions_s1.csv"
    corpus_path = output_dir / "corpus_s1.csv"
    qrels_path = output_dir / "qrels_s1.csv"
    selected_path = output_dir / "selected_examples.jsonl"

    questions_df.to_csv(questions_path, index=False)
    corpus_df.to_csv(corpus_path, index=False)
    qrels_df.to_csv(qrels_path, index=False)
    save_jsonl(selected_examples, selected_path)

    print("\nHotpotQA-mini generado correctamente.")
    print(f"Preguntas: {len(questions_df)} -> {questions_path}")
    print(f"Chunks del corpus: {len(corpus_df)} -> {corpus_path}")
    print(f"Qrels: {len(qrels_df)} -> {qrels_path}")
    print(f"Ejemplos seleccionados: {selected_path}")

    if not questions_df.empty:
        print("\nDistribución por topic:")
        print(questions_df["topic"].value_counts().to_string())

        print("\nDistribución por tipo:")
        print(questions_df["hotpot_type"].value_counts().to_string())

        print("\nDistribución por dificultad:")
        print(questions_df["level"].value_counts().to_string())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea HotpotQA-mini desde HotpotQA distractor JSONL."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/hotpotqa_distractor/hotpotqa_distractor_validation.jsonl"),
        help="Ruta al JSONL descargado desde Hugging Face.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/s1/hotpotqa_mini"),
        help="Directorio de salida.",
    )

    parser.add_argument(
        "--per-topic",
        type=int,
        default=10,
        help="Cantidad de preguntas por tema.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    prepare_hotpotqa_mini(
        input_path=args.input_path,
        output_dir=args.output_dir,
        per_topic=args.per_topic,
    )


if __name__ == "__main__":
    main()
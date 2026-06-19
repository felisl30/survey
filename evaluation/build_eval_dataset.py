#!/usr/bin/env python3
"""
build_eval_dataset.py

Construye el set de evaluación unificado para comparar S1/S2/S3.

Salidas:
    data/eval/questions_eval.csv   -- 80 retrieve (HotpotQA) + 20 direct (TruthfulQA)
    data/eval/corpus_eval.csv      -- chunks de las 80 retrieve
    data/eval/qrels_eval.csv       -- evidencia gold para las 80 retrieve
    data/eval/build_summary.json   -- estadísticas de construcción
    indexes/eval/                  -- índice vectorial del corpus

Uso:
    python evaluation/build_eval_dataset.py
    python evaluation/build_eval_dataset.py --retrieve-n 8 --direct-n 2  # smoke test
    python evaluation/build_eval_dataset.py --force                       # sobrescribe si ya existe
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    EVAL_BUILD_SUMMARY_PATH,
    EVAL_CORPUS_PATH,
    EVAL_DATA_DIR,
    EVAL_INDEX_DIR,
    EVAL_QRELS_PATH,
    EVAL_QUESTIONS_PATH,
    HOTPOTQA_DISTRACTOR_DIR,
    S1_DATA_DIR,
    S2_DATA_DIR,
)

HOTPOTQA_JSONL = HOTPOTQA_DISTRACTOR_DIR / "hotpotqa_distractor_validation.jsonl"
S1_QUESTIONS_PATH_LOCAL = S1_DATA_DIR / "questions_s1.csv"
S2_QUESTIONS_PATH_LOCAL = S2_DATA_DIR / "questions_s2.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RAG_PROMPT_TEMPLATE = """Respondé la siguiente pregunta usando únicamente el contexto recuperado.

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

EXPECTED_FORMAT = '{"answer": str, "confidence": float, "abstained": bool}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_context_items(example: dict[str, Any]) -> list[dict[str, Any]]:
    context = example.get("context", {})
    if not isinstance(context, dict):
        return []
    titles = context.get("title", [])
    sentence_groups = context.get("sentences", [])
    items = []
    for i, title in enumerate(titles):
        sentences = sentence_groups[i] if i < len(sentence_groups) else []
        if not isinstance(sentences, list):
            sentences = [sentences]
        text = " ".join(clean_text(s) for s in sentences if clean_text(s))
        if clean_text(title) and text:
            items.append({"paragraph_index": i, "title": clean_text(title), "text": text})
    return items


def get_supporting_titles(example: dict[str, Any]) -> list[str]:
    facts = example.get("supporting_facts", {})
    if not isinstance(facts, dict):
        return []
    return [clean_text(t) for t in facts.get("title", []) if clean_text(t)]


def normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", clean_text(value)).lower()
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Subset A: HotpotQA retrieve questions
# ---------------------------------------------------------------------------

def load_existing_hotpotqa_ids() -> set[str]:
    """IDs originales ya usados en questions_s1.csv, para no repetirlos."""
    if not S1_QUESTIONS_PATH_LOCAL.exists():
        return set()
    df = pd.read_csv(S1_QUESTIONS_PATH_LOCAL)
    if "original_hotpotqa_id" in df.columns:
        return set(df["original_hotpotqa_id"].dropna().astype(str))
    return set()


def select_hotpotqa_examples(
    examples: list[dict[str, Any]],
    skip_ids: set[str],
    target_bridge: int,
    target_comparison: int,
) -> list[dict[str, Any]]:
    selected_bridge: list[dict[str, Any]] = []
    selected_comparison: list[dict[str, Any]] = []

    for ex in examples:
        original_id = clean_text(ex.get("id", ""))
        if original_id in skip_ids:
            continue
        if not clean_text(ex.get("question", "")) or not clean_text(ex.get("answer", "")):
            continue
        if not get_context_items(ex):
            continue

        hotpot_type = clean_text(ex.get("type", "")).lower()
        if hotpot_type == "bridge" and len(selected_bridge) < target_bridge:
            selected_bridge.append(ex)
        elif hotpot_type == "comparison" and len(selected_comparison) < target_comparison:
            selected_comparison.append(ex)

        if len(selected_bridge) >= target_bridge and len(selected_comparison) >= target_comparison:
            break

    return selected_bridge + selected_comparison


def build_retrieve_rows(
    selected: list[dict[str, Any]],
    id_prefix: str = "eval_hotpotqa",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (questions_rows, corpus_rows, qrels_rows)."""
    questions_rows: list[dict] = []
    corpus_rows: list[dict] = []
    qrels_rows: list[dict] = []

    for q_index, example in enumerate(selected):
        question_id = f"{id_prefix}_{q_index:04d}"
        original_id = clean_text(example.get("id", ""))
        question_text = clean_text(example.get("question", ""))
        answer = clean_text(example.get("answer", ""))
        hotpot_type = clean_text(example.get("type", ""))
        level = clean_text(example.get("level", ""))

        context_items = get_context_items(example)
        supporting_titles = get_supporting_titles(example)
        supporting_norm = {normalize_text(t) for t in supporting_titles}

        gold_chunk_ids: list[str] = []
        context_chunk_ids: list[str] = []
        gold_titles: list[str] = []

        for item in context_items:
            p = item["paragraph_index"]
            chunk_id = f"{question_id}_chunk_{p:02d}"
            doc_id = f"{question_id}_doc_{p:02d}"
            is_gold = normalize_text(item["title"]) in supporting_norm

            corpus_rows.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": item["title"],
                "text": item["text"],
                "source": "hotpotqa_huggingface_distractor",
                "source_split": "validation",
                "topic": "",
                "question_id": question_id,
                "original_hotpotqa_id": original_id,
                "paragraph_index": p,
                "is_gold_evidence": is_gold,
            })
            context_chunk_ids.append(chunk_id)
            if is_gold:
                gold_chunk_ids.append(chunk_id)
                gold_titles.append(item["title"])
                qrels_rows.append({
                    "question_id": question_id,
                    "chunk_id": chunk_id,
                    "relevance": 1,
                    "title": item["title"],
                    "original_hotpotqa_id": original_id,
                })

        prompt = RAG_PROMPT_TEMPLATE.format(question=question_text)
        questions_rows.append({
            "id": question_id,
            "source_system": "eval",
            "source_question_id": original_id,
            "source_dataset": "hotpotqa",
            "dataset": "hotpotqa",
            "case_type": "open_qa",
            "s2_case_type": "rag_multi_hop",
            "expected_route": "retrieve",
            "acceptable_routes_json": json.dumps(["retrieve"]),
            "requires_retrieval": True,
            "retrieval_mode": "multi_hop",
            "expected_behavior": "answer",
            "expected_final_behavior": "answer",
            "routing_question": question_text,
            "original_question": question_text,
            "question": question_text,
            "prompt": prompt,
            "source_prompt": prompt,
            "subject": "",
            "topic": "",
            "difficulty": "",
            "level": level,
            "hotpot_type": hotpot_type,
            "source": "HotpotQA distractor via Hugging Face",
            "source_split": "validation",
            "original_hotpotqa_id": original_id,
            "truthfulqa_category": "",
            "truthfulqa_type": "",
            "original_source": "",
            "A": "", "B": "", "C": "", "D": "",
            "answer_choices_json": "",
            "gold_answer": answer,
            "gold_answer_idx": "",
            "gold_answer_text": answer,
            "expected_answer": answer,
            "best_answer": answer,
            "correct_answers_json": json.dumps([answer]),
            "incorrect_answers_json": json.dumps([]),
            "gold_evidence_ids": "|".join(gold_chunk_ids),
            "gold_evidence_titles": "|".join(sorted(set(gold_titles))),
            "context_chunk_ids": "|".join(context_chunk_ids),
            "is_synthetic": False,
            "synthetic_strategy": "",
            "expected_model_output_format": EXPECTED_FORMAT,
            "evaluation_notes": "",
        })

    return questions_rows, corpus_rows, qrels_rows


# ---------------------------------------------------------------------------
# Subset B: TruthfulQA direct questions
# ---------------------------------------------------------------------------

def load_direct_questions(target_n: int, seed: int = 42) -> list[dict]:
    """Toma direct_truthfulqa rows de questions_s2.csv y les cambia el id prefix."""
    df = pd.read_csv(S2_QUESTIONS_PATH_LOCAL)
    direct = df[df["s2_case_type"] == "direct_truthfulqa"].copy()

    if len(direct) == 0:
        raise ValueError("No hay filas direct_truthfulqa en questions_s2.csv")

    n = min(target_n, len(direct))
    if len(direct) > n:
        direct = direct.sample(n=n, random_state=seed)

    rows = []
    for i, (_, row) in enumerate(direct.iterrows()):
        d = row.to_dict()
        d["id"] = f"eval_truthfulqa_{i:04d}"
        d["source_system"] = "eval"
        d["source_question_id"] = row.get("id", "")
        rows.append(d)

    return rows


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(corpus_df: pd.DataFrame, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    corpus_df = corpus_df.copy()
    corpus_df["embedding_text"] = corpus_df.apply(
        lambda r: f"Title: {clean_text(r.get('title',''))}\nText: {clean_text(r.get('text',''))}",
        axis=1,
    )

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = corpus_df["embedding_text"].tolist()

    print(f"Calculando embeddings para {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    embeddings = l2_normalize(embeddings).astype("float32")

    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.json"

    corpus_df.to_csv(chunks_path, index=False)
    np.save(embeddings_path, embeddings)

    metadata = {
        "system": "eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(EVAL_CORPUS_PATH),
        "index_dir": str(index_dir),
        "embedding_model": EMBEDDING_MODEL,
        "n_chunks": int(len(corpus_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_file": str(embeddings_path),
        "chunks_file": str(chunks_path),
        "metadata_file": str(metadata_path),
        "normalized_embeddings": True,
        "similarity": "cosine_similarity_via_dot_product",
        "text_field": "embedding_text = Title + Text",
        "batch_size": 32,
        "columns": list(corpus_df.columns),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Índice guardado en: {index_dir}")
    print(f"  chunks.csv: {len(corpus_df)} filas")
    print(f"  embeddings.npy: {embeddings.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_eval_dataset(
    retrieve_n: int,
    direct_n: int,
    output_questions: Path,
    output_corpus: Path,
    output_qrels: Path,
    output_summary: Path,
    index_dir: Path,
    seed: int,
    force: bool,
    skip_index: bool,
) -> None:
    if output_questions.exists() and not force:
        print(f"Ya existe {output_questions}. Usá --force para sobrescribir.")
        return

    output_questions.parent.mkdir(parents=True, exist_ok=True)

    # --- Subset A: HotpotQA retrieve ---
    print(f"\n[1/4] Seleccionando {retrieve_n} preguntas HotpotQA retrieve...")
    skip_ids = load_existing_hotpotqa_ids()
    print(f"  Excluyendo {len(skip_ids)} IDs ya usados en questions_s1.csv")

    examples = load_jsonl(HOTPOTQA_JSONL)
    target_bridge = retrieve_n // 2
    target_comparison = retrieve_n - target_bridge
    selected = select_hotpotqa_examples(examples, skip_ids, target_bridge, target_comparison)

    n_got = len(selected)
    if n_got < retrieve_n:
        print(f"  AVISO: solo se encontraron {n_got} preguntas (pedidas: {retrieve_n})")

    q_retrieve, corpus_rows, qrels_rows = build_retrieve_rows(selected)
    print(f"  Preguntas retrieve: {len(q_retrieve)}")
    print(f"  Chunks corpus: {len(corpus_rows)}")
    print(f"  Qrels: {len(qrels_rows)}")

    # --- Subset B: TruthfulQA direct ---
    print(f"\n[2/4] Seleccionando {direct_n} preguntas TruthfulQA direct...")
    q_direct = load_direct_questions(direct_n, seed=seed)
    print(f"  Preguntas direct: {len(q_direct)}")

    # --- Merge y guardar ---
    print("\n[3/4] Guardando archivos...")
    all_questions = q_retrieve + q_direct

    questions_df = pd.DataFrame(all_questions)
    corpus_df = pd.DataFrame(corpus_rows)
    qrels_df = pd.DataFrame(qrels_rows)

    questions_df.to_csv(output_questions, index=False)
    corpus_df.to_csv(output_corpus, index=False)
    qrels_df.to_csv(output_qrels, index=False)
    print(f"  {output_questions}  ({len(questions_df)} filas)")
    print(f"  {output_corpus}  ({len(corpus_df)} filas)")
    print(f"  {output_qrels}  ({len(qrels_df)} filas)")

    # --- Build index ---
    if not skip_index:
        print("\n[4/4] Construyendo índice vectorial...")
        build_index(corpus_df, index_dir)
    else:
        print("\n[4/4] skip_index=True, omitiendo construcción de índice.")

    # --- Summary ---
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieve_n": len(q_retrieve),
        "direct_n": len(q_direct),
        "total_n": len(all_questions),
        "corpus_chunks": len(corpus_rows),
        "qrels": len(qrels_rows),
        "retrieve_hotpot_type": questions_df[questions_df["expected_route"] == "retrieve"]["hotpot_type"].value_counts().to_dict(),
        "direct_truthfulqa_category": questions_df[questions_df["expected_route"] == "direct"]["truthfulqa_category"].value_counts().to_dict(),
        "index_dir": str(index_dir),
        "questions_path": str(output_questions),
        "corpus_path": str(output_corpus),
        "qrels_path": str(output_qrels),
        "seed": seed,
    }
    with output_summary.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResumen guardado en: {output_summary}")

    print("\n=== Dataset de evaluación listo ===")
    print(f"  Total preguntas: {len(all_questions)} ({len(q_retrieve)} retrieve + {len(q_direct)} direct)")
    print(f"  Corpus: {len(corpus_rows)} chunks")
    print(f"  Index dir: {index_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye el set de evaluación unificado para S1/S2/S3."
    )
    parser.add_argument("--retrieve-n", type=int, default=80,
                        help="Cantidad de preguntas retrieve (HotpotQA). Default: 80.")
    parser.add_argument("--direct-n", type=int, default=20,
                        help="Cantidad de preguntas direct (TruthfulQA). Default: 20.")
    parser.add_argument("--questions-path", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--corpus-path", type=Path, default=EVAL_CORPUS_PATH)
    parser.add_argument("--qrels-path", type=Path, default=EVAL_QRELS_PATH)
    parser.add_argument("--summary-path", type=Path, default=EVAL_BUILD_SUMMARY_PATH)
    parser.add_argument("--index-dir", type=Path, default=EVAL_INDEX_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Sobrescribe si ya existe.")
    parser.add_argument("--skip-index", action="store_true",
                        help="No construye el índice vectorial (útil para pruebas rápidas).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_eval_dataset(
        retrieve_n=args.retrieve_n,
        direct_n=args.direct_n,
        output_questions=args.questions_path,
        output_corpus=args.corpus_path,
        output_qrels=args.qrels_path,
        output_summary=args.summary_path,
        index_dir=args.index_dir,
        seed=args.seed,
        force=args.force,
        skip_index=args.skip_index,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_s3_mc_flare_like.py

S3-MC: variante FLARE-like / active retrieval para MuSiQue-MC.

Idea:
1. Genera una hipótesis inicial A/B/C/D sin contexto.
2. Decide si necesita retrieval usando:
   - confianza de la hipótesis;
   - score/gap del retriever sobre query + hipótesis.
3. Si usa retrieval, recupera top-k chunks y regenera/corrige la respuesta.
4. Guarda raw_output compatible con parse_s0_outputs.py y evaluate_s0.py.

Entrada:
    data/eval_mc/musique_mc_rag/questions.csv

Índice:
    indexes/eval_mc/musique_mc_rag

Salida:
    outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_raw.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from direct_llm import ask_direct_llm_with_metadata
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se pudo importar direct_llm.py. Ejecutá desde la raíz del proyecto."
    ) from exc


DEFAULT_QUESTIONS_PATH = Path("data/eval_mc/musique_mc_rag/questions.csv")
DEFAULT_INDEX_DIR = Path("indexes/eval_mc/musique_mc_rag")
DEFAULT_OUTPUT_PATH = Path(
    "outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_raw.csv"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VALID_OPTIONS = {"A", "B", "C", "D"}


S3_MC_CANDIDATE_SYSTEM_PROMPT = """
You are the candidate generator for an experimental S3-MC FLARE-like QA system.

Task:
- Answer the multiple-choice question without retrieved context.
- Produce a short initial hypothesis.
- Return exactly one option among A, B, C, or D.
- Do not include markdown.
- Return only valid JSON.

Required JSON:
{
  "answer": "A",
  "confidence": 0.0,
  "rationale": "short reason"
}
""".strip()


S3_MC_REGENERATE_SYSTEM_PROMPT = """
You are the verifier/regenerator for an experimental S3-MC FLARE-like QA system.

Task:
- Use only the retrieved context and the answer options.
- Check the initial hypothesis.
- If the evidence supports it, keep it.
- If the evidence contradicts it, correct it.
- If the evidence is incomplete, still choose the most likely option among A, B, C, or D.
- Do not abstain.
- Return exactly one option among A, B, C, or D.
- Do not include markdown.
- Return only valid JSON.

Required JSON:
{
  "answer": "A",
  "confidence": 0.0,
  "rationale": "short reason based on retrieved context"
}
""".strip()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def safe_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if 1.0 < number <= 100.0:
        number /= 100.0
    return min(max(number, 0.0), 1.0)


def safe_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def strip_markdown_fence(text: str) -> str:
    stripped = clean_text(text)
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> str | None:
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


def parse_json_object(text: str) -> tuple[dict[str, Any], str]:
    json_text = extract_first_json_object(text)
    if not json_text:
        return {}, "No se encontró JSON."

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {}, f"JSON inválido: {exc}"

    if not isinstance(obj, dict):
        return {}, "El JSON parseado no es un objeto."

    return obj, ""


def infer_option_from_text(text: Any) -> str:
    text = clean_text(text).strip()
    if not text:
        return ""

    upper = text.upper().strip()
    if upper in VALID_OPTIONS:
        return upper

    patterns = [
        r"^\(?\s*([ABCDabcd])\s*\)?[\.\):\-]?\s*$",
        r"^\(?\s*([ABCDabcd])\s*\)?[\.\):\-]\s+",
        r"\b(?:answer|option|choice|respuesta|opción|opcion)\s*(?:is|es|:)?\s*\(?\s*([ABCDabcd])\s*\)?\b",
        r"\bthe\s+answer\s+is\s+\(?\s*([ABCDabcd])\s*\)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


def validate_mc_payload(obj: dict[str, Any], raw_output: str) -> dict[str, Any]:
    answer = infer_option_from_text(obj.get("answer", ""))
    if not answer:
        answer = infer_option_from_text(raw_output)

    confidence = safe_float(obj.get("confidence", 0.0), default=0.0)
    rationale = clean_text(obj.get("rationale", ""))

    return {
        "answer": answer,
        "confidence": confidence,
        "rationale": rationale,
        "valid_answer": answer in VALID_OPTIONS,
    }


def get_question_text(row: pd.Series) -> str:
    for col in ["original_question", "question", "retrieval_query", "prompt"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    raise ValueError(f"Fila id={row.get('id', '<sin id>')} sin pregunta.")


def get_retrieval_query_base(row: pd.Series) -> str:
    for col in ["retrieval_query", "original_question", "question", "prompt"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return get_question_text(row)


def get_option(row: pd.Series, label: str) -> str:
    return clean_text(row.get(label, ""))


def build_options_block(row: pd.Series) -> str:
    return "\n".join(
        f"{label}. {get_option(row, label)}"
        for label in ["A", "B", "C", "D"]
    )


def build_candidate_prompt(row: pd.Series) -> str:
    question = get_question_text(row)
    options = build_options_block(row)

    return f"""Answer this MuSiQue multiple-choice question.

Question:
{question}

Options:
{options}

Instructions:
- Choose exactly one option: A, B, C, or D.
- Do not use retrieved context in this first step.
- Keep the rationale short.

Return only valid JSON:
{{
  "answer": "A",
  "confidence": 0.0,
  "rationale": "short reason"
}}"""


def build_context_block(chunks: list[dict[str, Any]], max_chars_per_chunk: int) -> str:
    blocks: list[str] = []

    for i, item in enumerate(chunks, start=1):
        doc_id = clean_text(item.get("doc_id", ""))
        title = clean_text(item.get("title", ""))
        text = clean_text(item.get("text", ""))
        score = item.get("score", "")

        if max_chars_per_chunk > 0 and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " [...]"

        try:
            score_text = f"{float(score):.6f}"
        except (TypeError, ValueError):
            score_text = ""

        blocks.append(
            f"""[{i}]
doc_id: {doc_id}
title: {title}
score: {score_text}
text: {text}"""
        )

    return "\n\n".join(blocks)


def build_regenerate_prompt(
    row: pd.Series,
    *,
    candidate: dict[str, Any],
    chunks: list[dict[str, Any]],
    max_chars_per_chunk: int,
) -> str:
    question = get_question_text(row)
    options = build_options_block(row)
    context = build_context_block(chunks, max_chars_per_chunk=max_chars_per_chunk)

    return f"""Verify and possibly correct the initial answer using the retrieved context.

Question:
{question}

Options:
{options}

Initial hypothesis:
answer: {candidate.get("answer", "")}
confidence: {candidate.get("confidence", 0.0)}
rationale: {candidate.get("rationale", "")}

Retrieved context:
{context}

Instructions:
- Use only the retrieved context and the options.
- Select exactly one option: A, B, C, or D.
- If the initial hypothesis is wrong, correct it.
- Do not abstain.
- Keep the rationale short.

Return only valid JSON:
{{
  "answer": "A",
  "confidence": 0.0,
  "rationale": "short reason based on retrieved context"
}}"""


def call_llm_json(
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    max_retries: int,
) -> dict[str, Any]:
    result = ask_direct_llm_with_metadata(
        prompt,
        model=model,
        system_prompt=system_prompt,
        max_retries=max_retries,
    )

    raw_output = clean_text(result.get("raw_output", ""))
    obj, parse_error = parse_json_object(raw_output)
    parsed = validate_mc_payload(obj, raw_output)

    return {
        **parsed,
        "raw_output": raw_output,
        "parse_error": parse_error,
        "model": result.get("model", model),
        "usage_json": result.get("usage_json", ""),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "total_tokens": result.get("total_tokens"),
        "latency_seconds": result.get("latency_seconds"),
    }


def load_retrieval_index(index_dir: Path, embedding_model_name: str):
    chunks_path = index_dir / "chunks.csv"
    embeddings_path = index_dir / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    chunks = pd.read_csv(chunks_path)
    embeddings = np.load(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks.csv tiene {len(chunks)} filas pero embeddings.npy tiene {embeddings.shape[0]} embeddings."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No se encontró sentence-transformers. Instalalo con: pip install sentence-transformers"
        ) from exc

    embedder = SentenceTransformer(embedding_model_name)

    return chunks, embeddings.astype("float32"), embedder


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def encode_query(embedder, query: str) -> np.ndarray:
    emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")[0]


def row_text(row: pd.Series, col: str) -> str:
    return clean_text(row.get(col, ""))


def retrieve(
    *,
    query: str,
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    embedder,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    query_emb = encode_query(embedder, query)

    # embeddings del índice ya deberían estar normalizados, pero normalizamos por seguridad.
    emb = embeddings
    norms = np.linalg.norm(emb, axis=1)
    if not np.all((norms > 0.99) & (norms < 1.01)):
        emb = normalize_matrix(emb)

    scores = emb @ query_emb
    top_idx = np.argsort(-scores)[:top_k]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        row = chunks.iloc[int(idx)]
        results.append(
            {
                "rank": rank,
                "doc_id": row_text(row, "doc_id") or row_text(row, "chunk_id") or str(idx),
                "chunk_id": row_text(row, "chunk_id") or row_text(row, "doc_id") or str(idx),
                "title": row_text(row, "title"),
                "text": row_text(row, "text"),
                "score": float(scores[int(idx)]),
            }
        )

    return results


def build_s3_mc_retrieval_query(row: pd.Series, candidate: dict[str, Any]) -> str:
    base = get_retrieval_query_base(row)
    answer = clean_text(candidate.get("answer", ""))
    option_text = get_option(row, answer) if answer in VALID_OPTIONS else ""
    rationale = clean_text(candidate.get("rationale", ""))

    parts = [base]

    if answer and option_text:
        parts.append(f"Candidate answer {answer}: {option_text}")

    if rationale:
        parts.append(f"Candidate rationale: {rationale}")

    return " ".join(parts).strip()


def retrieval_scores_for_decision(
    *,
    query: str,
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    embedder,
    candidate_pool_k: int,
) -> dict[str, Any]:
    retrieved = retrieve(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder,
        top_k=max(candidate_pool_k, 2),
    )

    scores = [float(item["score"]) for item in retrieved]

    top1 = scores[0] if len(scores) >= 1 else 0.0
    top2 = scores[1] if len(scores) >= 2 else 0.0
    top3 = scores[2] if len(scores) >= 3 else 0.0
    top5_scores = scores[:5]

    return {
        "candidate_retrieved": retrieved,
        "top1_score": top1,
        "top2_score": top2,
        "top3_score": top3,
        "top1_top2_gap": top1 - top2,
        "top5_mean_score": float(np.mean(top5_scores)) if top5_scores else 0.0,
    }


def decide_active_retrieval(
    *,
    candidate: dict[str, Any],
    top1_score: float,
    top1_top2_gap: float,
    confidence_threshold: float,
    score_threshold: float,
    min_gap: float,
) -> tuple[bool, str]:
    confidence = safe_float(candidate.get("confidence", 0.0), default=0.0)

    if not candidate.get("valid_answer", False):
        return True, "candidate_invalid_answer"

    if confidence < confidence_threshold:
        return True, f"candidate_confidence_below_{confidence_threshold:.2f}"

    if top1_score >= score_threshold and top1_top2_gap >= min_gap:
        return True, f"retriever_score_ge_{score_threshold:.2f}_and_gap_ge_{min_gap:.2f}"

    return False, "candidate_confident_and_retrieval_signal_weak"


def sum_optional_int(values: list[Any]) -> int | None:
    total = 0
    found = False
    for value in values:
        if value is None or clean_text(value) == "":
            continue
        try:
            total += int(float(value))
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def sum_optional_float(values: list[Any]) -> float | None:
    total = 0.0
    found = False
    for value in values:
        if value is None or clean_text(value) == "":
            continue
        try:
            total += float(value)
            found = True
        except (TypeError, ValueError):
            continue
    return round(total, 3) if found else None


def build_output_row(
    input_row: pd.Series,
    *,
    model: str,
    top_k: int,
    confidence_threshold: float,
    score_threshold: float,
    min_gap: float,
    candidate: dict[str, Any],
    retrieval_query: str,
    decision: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    final_call: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    active_retrieval = bool(decision["active_retrieval"])
    final = final_call if final_call is not None else candidate

    final_answer = clean_text(final.get("answer", ""))
    final_confidence = safe_float(final.get("confidence", 0.0), default=0.0)
    final_rationale = clean_text(final.get("rationale", ""))

    raw_payload = {
        "answer": final_answer,
        "confidence": final_confidence,
    }

    trace = [
        {
            "step": 1,
            "stage": "candidate_generation",
            "answer": candidate.get("answer", ""),
            "confidence": candidate.get("confidence", 0.0),
            "rationale": candidate.get("rationale", ""),
            "raw_output": candidate.get("raw_output", ""),
            "parse_error": candidate.get("parse_error", ""),
        },
        {
            "step": 2,
            "stage": "active_retrieval_decision",
            "active_retrieval": active_retrieval,
            "reason": decision["reason"],
            "retrieval_query": retrieval_query,
            "top1_score": decision["top1_score"],
            "top2_score": decision["top2_score"],
            "top3_score": decision["top3_score"],
            "top1_top2_gap": decision["top1_top2_gap"],
            "top5_mean_score": decision["top5_mean_score"],
        },
    ]

    if active_retrieval:
        trace.append(
            {
                "step": 3,
                "stage": "retrieval_regeneration",
                "answer": final.get("answer", ""),
                "confidence": final.get("confidence", 0.0),
                "rationale": final.get("rationale", ""),
                "raw_output": final.get("raw_output", ""),
                "parse_error": final.get("parse_error", ""),
                "retrieved_doc_ids": [item.get("doc_id", "") for item in retrieved_chunks],
                "retrieved_scores": [item.get("score", "") for item in retrieved_chunks],
            }
        )

    total_tokens = sum_optional_int([
        candidate.get("total_tokens"),
        final_call.get("total_tokens") if final_call else None,
    ])
    input_tokens = sum_optional_int([
        candidate.get("input_tokens"),
        final_call.get("input_tokens") if final_call else None,
    ])
    output_tokens = sum_optional_int([
        candidate.get("output_tokens"),
        final_call.get("output_tokens") if final_call else None,
    ])
    latency_seconds = sum_optional_float([
        candidate.get("latency_seconds"),
        final_call.get("latency_seconds") if final_call else None,
    ])

    output: dict[str, Any] = {}

    for col in input_row.index:
        output[col] = input_row.get(col, "")

    output.update(
        {
            "system": "S3_MC_flare_like_active_retrieval",
            "model": model,
            "top_k": top_k,
            "s3_mc_policy": "candidate_confidence_or_retriever_score",
            "confidence_threshold": confidence_threshold,
            "score_threshold": score_threshold,
            "min_gap": min_gap,
            "predicted_route": "retrieve" if active_retrieval else "direct",
            "active_retrieval_triggered": active_retrieval,
            "retrieval_query": retrieval_query,
            "retrieval_decision_reason": decision["reason"],
            "top1_score": decision["top1_score"],
            "top2_score": decision["top2_score"],
            "top3_score": decision["top3_score"],
            "top1_top2_gap": decision["top1_top2_gap"],
            "top5_mean_score": decision["top5_mean_score"],
            "n_generation_steps": 2 if active_retrieval else 1,
            "n_retrieval_steps": 1 if active_retrieval else 0,
            "n_docs_retrieved": len(retrieved_chunks),
            "retrieved_doc_ids_json": safe_json_dumps([item.get("doc_id", "") for item in retrieved_chunks]),
            "retrieved_titles_json": safe_json_dumps([item.get("title", "") for item in retrieved_chunks]),
            "retrieved_scores_json": safe_json_dumps([item.get("score", "") for item in retrieved_chunks]),
            "retrieved_context_json": safe_json_dumps(retrieved_chunks),
            "candidate_answer": candidate.get("answer", ""),
            "candidate_confidence": candidate.get("confidence", 0.0),
            "candidate_rationale": candidate.get("rationale", ""),
            "candidate_raw_output": candidate.get("raw_output", ""),
            "candidate_parse_error": candidate.get("parse_error", ""),
            "final_answer": final_answer,
            "final_confidence": final_confidence,
            "final_rationale": final_rationale,
            "final_raw_output": final.get("raw_output", ""),
            "final_parse_error": final.get("parse_error", ""),
            "s3_trace_json": safe_json_dumps(trace),
            "raw_output": safe_json_dumps(raw_payload),
            "raw_response_json": safe_json_dumps(
                {
                    "answer": final_answer,
                    "confidence": final_confidence,
                    "rationale": final_rationale,
                    "active_retrieval": active_retrieval,
                    "trace": trace,
                    "error": error,
                }
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency_seconds,
            "error": error,
        }
    )

    return output


def load_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(output_path)
    if "id" not in df.columns:
        raise ValueError(f"{output_path} existe pero no tiene columna id.")
    return df


def save_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    questions_path = Path(args.questions_path)
    index_dir = Path(args.index_dir)
    output_path = Path(args.output_path)

    if not questions_path.exists():
        raise FileNotFoundError(questions_path)
    if not index_dir.exists():
        raise FileNotFoundError(index_dir)

    questions = pd.read_csv(questions_path)
    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    existing = load_existing(output_path) if args.resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending = questions[~questions["id"].astype(str).isin(existing_ids)].copy()

    print("S3-MC FLARE-like runner")
    print("-----------------------")
    print(f"Questions: {questions_path}")
    print(f"Index: {index_dir}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Top-k: {args.top_k}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Min gap: {args.min_gap}")
    print(f"Rows considered: {len(questions)}")
    print(f"Rows existing: {len(existing_ids)}")
    print(f"Rows pending: {len(pending)}")

    chunks, embeddings, embedder = load_retrieval_index(index_dir, args.embedding_model)

    for i, (_, row) in enumerate(
        tqdm(pending.iterrows(), total=len(pending), desc="Running S3-MC"),
        start=1,
    ):
        error = ""
        final_call = None
        retrieved_chunks: list[dict[str, Any]] = []

        try:
            candidate_prompt = build_candidate_prompt(row)
            candidate = call_llm_json(
                prompt=candidate_prompt,
                system_prompt=S3_MC_CANDIDATE_SYSTEM_PROMPT,
                model=args.model,
                max_retries=args.max_retries,
            )

            retrieval_query = build_s3_mc_retrieval_query(row, candidate)

            decision_scores = retrieval_scores_for_decision(
                query=retrieval_query,
                chunks=chunks,
                embeddings=embeddings,
                embedder=embedder,
                candidate_pool_k=max(args.candidate_pool_k, args.top_k, 5),
            )

            active_retrieval, reason = decide_active_retrieval(
                candidate=candidate,
                top1_score=float(decision_scores["top1_score"]),
                top1_top2_gap=float(decision_scores["top1_top2_gap"]),
                confidence_threshold=args.confidence_threshold,
                score_threshold=args.score_threshold,
                min_gap=args.min_gap,
            )

            decision = {
                **decision_scores,
                "active_retrieval": active_retrieval,
                "reason": reason,
            }

            if active_retrieval:
                retrieved_chunks = decision_scores["candidate_retrieved"][: args.top_k]

                regenerate_prompt = build_regenerate_prompt(
                    row,
                    candidate=candidate,
                    chunks=retrieved_chunks,
                    max_chars_per_chunk=args.max_chars_per_chunk,
                )

                final_call = call_llm_json(
                    prompt=regenerate_prompt,
                    system_prompt=S3_MC_REGENERATE_SYSTEM_PROMPT,
                    model=args.model,
                    max_retries=args.max_retries,
                )

        except Exception as exc:
            candidate = {
                "answer": "",
                "confidence": 0.0,
                "rationale": "",
                "valid_answer": False,
                "raw_output": "",
                "parse_error": "",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_seconds": None,
            }
            retrieval_query = ""
            decision = {
                "active_retrieval": False,
                "reason": "run_error",
                "candidate_retrieved": [],
                "top1_score": 0.0,
                "top2_score": 0.0,
                "top3_score": 0.0,
                "top1_top2_gap": 0.0,
                "top5_mean_score": 0.0,
            }
            error = str(exc)

        output_row = build_output_row(
            row,
            model=args.model,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            score_threshold=args.score_threshold,
            min_gap=args.min_gap,
            candidate=candidate,
            retrieval_query=retrieval_query,
            decision=decision,
            retrieved_chunks=retrieved_chunks,
            final_call=final_call,
            error=error,
        )
        rows.append(output_row)

        if args.save_every > 0 and i % args.save_every == 0:
            save_rows(rows, output_path)

    save_rows(rows, output_path)
    out = pd.DataFrame(rows)

    print(f"\nS3-MC raw saved to: {output_path}")
    print(f"Rows: {len(out)}")

    if "error" in out.columns:
        err_rate = out["error"].fillna("").astype(str).str.strip().ne("").mean()
        print(f"Run error rate: {err_rate:.3f}")

    if "active_retrieval_triggered" in out.columns:
        print(f"Active retrieval rate: {out['active_retrieval_triggered'].astype(bool).mean():.3f}")

    if "total_tokens" in out.columns:
        print(f"Avg total tokens: {pd.to_numeric(out['total_tokens'], errors='coerce').mean():.2f}")

    if "latency_seconds" in out.columns:
        print(f"Avg latency seconds: {pd.to_numeric(out['latency_seconds'], errors='coerce').mean():.2f}")

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run S3-MC FLARE-like on MuSiQue-MC.")

    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-k", type=int, default=10)
    parser.add_argument("--max-chars-per-chunk", type=int, default=900)

    parser.add_argument("--confidence-threshold", type=float, default=0.78)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--min-gap", type=float, default=0.05)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit debe ser mayor que 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k debe ser mayor que 0.")
    if args.candidate_pool_k <= 0:
        raise ValueError("--candidate-pool-k debe ser mayor que 0.")

    run_experiment(args)


if __name__ == "__main__":
    main()

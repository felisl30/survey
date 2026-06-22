#!/usr/bin/env python3
"""
run_s2_mc_real_adaptive.py

Ejecuta S2-MC real end-to-end sobre MuSiQue-100 MC.

Para cada pregunta:
1. Calcula scores de retrieval contra el índice MC-RAG.
2. Aplica una política adaptativa:
      retrieve si top1_score >= threshold y gap top1-top2 >= min_gap.
      direct en caso contrario.
3. Si route=direct, llama al LLM sin contexto.
4. Si route=retrieve, llama al LLM con contexto top-k.
5. Guarda raw outputs, tokens, latencias, ruta elegida y scores.

No reutiliza las respuestas S0/S1. Genera respuestas nuevas.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()


SYSTEM_PROMPT_DIRECT = """
Sos un sistema experimental S2-MC en ruta direct.

Tu tarea:
- Responder una pregunta de opción múltiple sin usar contexto recuperado.
- Elegir una única opción entre A, B, C o D.
- No explicar la respuesta.
- Devolver únicamente JSON válido.

Formato obligatorio:
{
  "answer": "A",
  "confidence": 0.0
}
""".strip()


SYSTEM_PROMPT_RETRIEVE = """
Sos un sistema experimental S2-MC en ruta retrieve.

Tu tarea:
- Responder una pregunta de opción múltiple usando el contexto recuperado.
- Elegir una única opción entre A, B, C o D.
- Si el contexto entra en conflicto con conocimiento interno, priorizá el contexto.
- No explicar la respuesta.
- Devolver únicamente JSON válido.

Formato obligatorio:
{
  "answer": "A",
  "confidence": 0.0
}
""".strip()


KEEP_COLUMNS = [
    "id",
    "question_id",
    "dataset",
    "case_type",
    "source_dataset",
    "benchmark_name",
    "original_question",
    "question",
    "retrieval_query",
    "prompt",
    "A",
    "B",
    "C",
    "D",
    "answer_choices_json",
    "gold_answer",
    "gold_answer_text",
    "expected_answer",
    "expected_route",
    "requires_retrieval",
    "original_source",
    "source_split",
    "difficulty",
    "frozen_input_sha256",
]


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def extract_usage_fields(usage: Any) -> dict[str, int | None]:
    usage_json = jsonable(usage)

    if not isinstance(usage_json, dict):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = usage_json.get("input_tokens")
    output_tokens = usage_json.get("output_tokens")
    total_tokens = usage_json.get("total_tokens")

    if input_tokens is None:
        input_tokens = usage_json.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage_json.get("completion_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY. Revisá .env o exportá la variable.")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path)
    if "id" not in existing.columns:
        raise ValueError(f"{output_path} existe pero no tiene columna id.")
    return existing


def build_direct_prompt(row: pd.Series) -> str:
    q = clean_str(row.get("original_question", "")) or clean_str(row.get("retrieval_query", ""))
    a = clean_str(row.get("A", ""))
    b = clean_str(row.get("B", ""))
    c = clean_str(row.get("C", ""))
    d = clean_str(row.get("D", ""))

    return f"""Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- No expliques tu respuesta.
- No uses contexto externo provisto por el sistema.
- Respondé únicamente con JSON válido.
- El campo "answer" debe ser exactamente una de estas letras: "A", "B", "C" o "D".
- El campo "confidence" debe ser un número entre 0 y 1.

Pregunta:
{q}

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


def build_context_block(retrieved_rows: pd.DataFrame) -> str:
    blocks = []

    for i, (_, row) in enumerate(retrieved_rows.iterrows(), start=1):
        title = clean_str(row.get("title", ""))
        text = clean_str(row.get("text", ""))

        blocks.append(
            f"[{i}] Título: {title}\n"
            f"Texto: {text}"
        )

    return "\n\n".join(blocks)


def build_retrieve_prompt(row: pd.Series, retrieved_rows: pd.DataFrame) -> str:
    context_block = build_context_block(retrieved_rows)
    q = clean_str(row.get("original_question", "")) or clean_str(row.get("retrieval_query", ""))
    a = clean_str(row.get("A", ""))
    b = clean_str(row.get("B", ""))
    c = clean_str(row.get("C", ""))
    d = clean_str(row.get("D", ""))

    return f"""Contexto recuperado:
{context_block}

Respondé la siguiente pregunta de opción múltiple.

Reglas:
- Elegí una única opción entre A, B, C o D.
- Usá el contexto recuperado si contiene evidencia útil.
- No expliques tu respuesta.
- Respondé únicamente con JSON válido.
- El campo "answer" debe ser exactamente una de estas letras: "A", "B", "C" o "D".
- El campo "confidence" debe ser un número entre 0 y 1.

Pregunta:
{q}

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


def compute_retrieval_scores(
    *,
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, list[float], dict[str, Any], float]:
    start = time.time()

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    q_emb = normalize_embeddings(q_emb)

    scores = (q_emb @ embeddings.T)[0]
    order = np.argsort(-scores)[:top_k]

    latency = round(time.time() - start, 4)

    retrieved_rows = chunks.iloc[order].copy()
    retrieved_scores = [float(scores[int(idx)]) for idx in order]

    top1 = retrieved_scores[0] if len(retrieved_scores) >= 1 else None
    top2 = retrieved_scores[1] if len(retrieved_scores) >= 2 else None
    top3 = retrieved_scores[2] if len(retrieved_scores) >= 3 else None

    features = {
        "top1_score": top1,
        "top2_score": top2,
        "top3_score": top3,
        "top1_top2_gap": None if top1 is None or top2 is None else top1 - top2,
        "top3_mean_score": None if len(retrieved_scores) < 3 else sum(retrieved_scores[:3]) / 3,
        "top5_mean_score": None if len(retrieved_scores) < 5 else sum(retrieved_scores[:5]) / 5,
    }

    return retrieved_rows, retrieved_scores, features, latency


def choose_route(features: dict[str, Any], *, threshold: float, min_gap: float) -> tuple[str, str]:
    top1 = features.get("top1_score")
    gap = features.get("top1_top2_gap")

    if top1 is not None and gap is not None and float(top1) >= threshold and float(gap) >= min_gap:
        return "retrieve", f"top1_score={top1:.4f} >= {threshold} and gap={gap:.4f} >= {min_gap}"

    top1_txt = "None" if top1 is None else f"{float(top1):.4f}"
    gap_txt = "None" if gap is None else f"{float(gap):.4f}"
    return "direct", f"policy not met: top1_score={top1_txt}, gap={gap_txt}"


def call_llm(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    prompt: str,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        start = time.time()

        try:
            response = client.responses.create(
                model=model_name,
                instructions=system_prompt,
                input=prompt,
            )

            latency = round(time.time() - start, 3)
            raw_output = getattr(response, "output_text", "")

            usage = getattr(response, "usage", None)
            usage_json = jsonable(usage)
            usage_fields = extract_usage_fields(usage)

            return {
                "raw_output": raw_output,
                "generation_latency_seconds": latency,
                "usage_json": json.dumps(usage_json, ensure_ascii=False),
                "input_tokens": usage_fields["input_tokens"],
                "output_tokens": usage_fields["output_tokens"],
                "total_tokens": usage_fields["total_tokens"],
            }

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2.0 * (2 ** attempt))

    raise RuntimeError(f"Falló el llamado al modelo luego de {max_retries + 1} intentos: {last_error}")


def build_output_row(
    *,
    row: pd.Series,
    model_name: str,
    top_k: int,
    policy_name: str,
    threshold: float,
    min_gap: float,
    predicted_route: str,
    router_reason: str,
    retrieved_rows_all: pd.DataFrame,
    retrieved_scores_all: list[float],
    retrieved_rows_used: pd.DataFrame,
    retrieved_scores_used: list[float],
    features: dict[str, Any],
    route_decision_latency_seconds: float,
    model_result: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for col in KEEP_COLUMNS:
        if col in row.index:
            out[col] = row.get(col, "")

    out["system"] = "S2_MC_real_adaptive"
    out["model"] = model_name
    out["top_k"] = top_k

    out["s2_policy_name"] = policy_name
    out["s2_policy_threshold"] = threshold
    out["s2_policy_min_gap"] = min_gap
    out["predicted_route"] = predicted_route
    out["parsed_route"] = predicted_route
    out["router_reason"] = router_reason

    out["top1_score"] = features.get("top1_score")
    out["top2_score"] = features.get("top2_score")
    out["top3_score"] = features.get("top3_score")
    out["top1_top2_gap"] = features.get("top1_top2_gap")
    out["top3_mean_score"] = features.get("top3_mean_score")
    out["top5_mean_score"] = features.get("top5_mean_score")

    out["route_decision_latency_seconds"] = route_decision_latency_seconds
    out["retrieval_latency_seconds"] = route_decision_latency_seconds if predicted_route == "retrieve" else 0.0
    out["n_docs_retrieved"] = len(retrieved_rows_used)

    out["candidate_retrieved_doc_ids_json"] = json.dumps(
        retrieved_rows_all["doc_id"].astype(str).tolist(),
        ensure_ascii=False,
    )
    out["candidate_retrieved_scores_json"] = json.dumps(
        retrieved_scores_all,
        ensure_ascii=False,
    )
    out["retrieved_doc_ids_json"] = json.dumps(
        retrieved_rows_used["doc_id"].astype(str).tolist(),
        ensure_ascii=False,
    )
    out["retrieved_titles_json"] = json.dumps(
        retrieved_rows_used["title"].fillna("").astype(str).tolist() if len(retrieved_rows_used) else [],
        ensure_ascii=False,
    )
    out["retrieved_scores_json"] = json.dumps(
        retrieved_scores_used,
        ensure_ascii=False,
    )
    out["retrieved_context_json"] = json.dumps(
        [
            {
                "doc_id": clean_str(r.get("doc_id", "")),
                "title": clean_str(r.get("title", "")),
                "text": clean_str(r.get("text", "")),
                "score": retrieved_scores_used[i],
            }
            for i, (_, r) in enumerate(retrieved_rows_used.iterrows())
        ],
        ensure_ascii=False,
    )

    if model_result is None:
        out["raw_output"] = ""
        out["generation_latency_seconds"] = None
        out["latency_seconds"] = route_decision_latency_seconds
        out["usage_json"] = ""
        out["input_tokens"] = None
        out["output_tokens"] = None
        out["total_tokens"] = None
    else:
        out["raw_output"] = model_result.get("raw_output", "")
        out["generation_latency_seconds"] = model_result.get("generation_latency_seconds")
        out["latency_seconds"] = round(
            route_decision_latency_seconds + float(model_result.get("generation_latency_seconds") or 0.0),
            3,
        )
        out["usage_json"] = model_result.get("usage_json", "")
        out["input_tokens"] = model_result.get("input_tokens")
        out["output_tokens"] = model_result.get("output_tokens")
        out["total_tokens"] = model_result.get("total_tokens")

    out["error"] = error

    return out


def save_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/musique_mc_rag/questions.csv"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes/eval_mc/musique_mc_rag"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/eval_mc/musique_mc_rag/s2_real/s2_gpt_5_mini_policy_top1_045_gap_005_raw.csv"),
    )
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--policy-name", type=str, default="top1_ge_0.45_gap_ge_0.05")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)

    args = parser.parse_args()

    chunks_path = args.index_dir / "chunks.csv"
    embeddings_path = args.index_dir / "embeddings.npy"

    for path in [args.questions_path, chunks_path, embeddings_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    questions = pd.read_csv(args.questions_path)
    if args.limit is not None:
        questions = questions.head(args.limit).copy()

    chunks = pd.read_csv(chunks_path)
    embeddings = np.load(embeddings_path).astype("float32")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"chunks y embeddings no coinciden: {len(chunks)} vs {embeddings.shape[0]}")

    existing = load_existing_results(args.output_path) if args.resume else pd.DataFrame()
    existing_ids = set(existing["id"].astype(str)) if not existing.empty else set()

    rows: list[dict[str, Any]] = []
    if not existing.empty:
        rows.extend(existing.to_dict(orient="records"))

    pending = questions[~questions["id"].astype(str).isin(existing_ids)].copy()

    print(f"Questions path: {args.questions_path}")
    print(f"Index dir: {args.index_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Model: {args.model}")
    print(f"Policy: {args.policy_name}")
    print(f"Threshold: {args.threshold}")
    print(f"Min gap: {args.min_gap}")
    print(f"Top-k: {args.top_k}")
    print(f"Rows total: {len(questions)}")
    print(f"Rows existing: {len(existing_ids)}")
    print(f"Rows pending: {len(pending)}")

    print("\nCargando modelo de embeddings...")
    embedding_model = SentenceTransformer(args.embedding_model)

    print("Inicializando cliente OpenAI...")
    client = get_openai_client()

    for i, (_, row) in enumerate(
        tqdm(pending.iterrows(), total=len(pending), desc="Running S2-MC real"),
        start=1,
    ):
        retrieved_all = pd.DataFrame()
        retrieved_scores_all: list[float] = []
        retrieved_used = pd.DataFrame()
        retrieved_scores_used: list[float] = []
        features: dict[str, Any] = {}
        route_latency = 0.0

        try:
            query = clean_str(row.get("retrieval_query", "")) or clean_str(row.get("original_question", ""))

            retrieved_all, retrieved_scores_all, features, route_latency = compute_retrieval_scores(
                query=query,
                model=embedding_model,
                embeddings=embeddings,
                chunks=chunks,
                top_k=args.top_k,
            )

            predicted_route, router_reason = choose_route(
                features,
                threshold=args.threshold,
                min_gap=args.min_gap,
            )

            if predicted_route == "retrieve":
                retrieved_used = retrieved_all
                retrieved_scores_used = retrieved_scores_all
                prompt = build_retrieve_prompt(row, retrieved_used)
                system_prompt = SYSTEM_PROMPT_RETRIEVE
            else:
                retrieved_used = pd.DataFrame(columns=retrieved_all.columns)
                retrieved_scores_used = []
                prompt = build_direct_prompt(row)
                system_prompt = SYSTEM_PROMPT_DIRECT

            model_result = call_llm(
                client=client,
                model_name=args.model,
                system_prompt=system_prompt,
                prompt=prompt,
                max_retries=args.max_retries,
            )

            error = ""

        except Exception as exc:
            predicted_route = "error"
            router_reason = ""
            model_result = None
            error = str(exc)

        output_row = build_output_row(
            row=row,
            model_name=args.model,
            top_k=args.top_k,
            policy_name=args.policy_name,
            threshold=args.threshold,
            min_gap=args.min_gap,
            predicted_route=predicted_route,
            router_reason=router_reason,
            retrieved_rows_all=retrieved_all,
            retrieved_scores_all=retrieved_scores_all,
            retrieved_rows_used=retrieved_used,
            retrieved_scores_used=retrieved_scores_used,
            features=features,
            route_decision_latency_seconds=route_latency,
            model_result=model_result,
            error=error,
        )

        rows.append(output_row)

        if args.save_every > 0 and i % args.save_every == 0:
            save_results(rows, args.output_path)

    save_results(rows, args.output_path)

    print(f"\nResultados S2-MC real guardados en: {args.output_path}")


if __name__ == "__main__":
    main()

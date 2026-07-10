#!/usr/bin/env python3
"""
build_s4_robustness_focus_input.py

Construye un input focalizado para auditar con S4/FIRE-like respuestas de S3-MC
en el experimento de robustez MuSiQue.

No llama a OpenAI.
Solo arma un CSV compatible con modelos/s4/run_s4_fire_like.py.

Uso:
python evaluation/build_s4_robustness_focus_input.py \
  --base-dir outputs/eval_mc/robustness_musique/gpt_5_4_mini \
  --questions-path data/eval_mc/robustness_musique/questions.csv \
  --preset core5 \
  --output-path outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/input/s4_robustness_focus_core5.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VALID_OPTIONS = {"A", "B", "C", "D"}


CORE5 = [
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0020",
        "focus_case_type": "adversarial_all_rag_wrong",
        "expected_s4_audit": "detect_error",
        "reason": "Todos los sistemas aciertan en clean pero fallan en adversarial; buen caso para ver si S4 detecta contaminación por distractores.",
    },
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0036",
        "focus_case_type": "s3_mc_regresses_adversarial",
        "expected_s4_audit": "detect_error",
        "reason": "S3-MC acierta en clean/noisy pero falla en adversarial; buen caso para auditar regresión de S3-MC.",
    },
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0029",
        "focus_case_type": "adversarial_only_s3_mc_correct",
        "expected_s4_audit": "preserve_correct",
        "reason": "S3-MC es el único sistema correcto en adversarial; buen caso para medir falso rechazo de S4.",
    },
    {
        "condition": "noisy",
        "question_id": "musique_mc__0019",
        "focus_case_type": "s3_mc_regresses_noisy",
        "expected_s4_audit": "detect_error",
        "reason": "Todos aciertan en clean, pero S3-MC falla en noisy; buen caso para auditar ruido no adversarial.",
    },
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0022",
        "focus_case_type": "s3_mc_regresses_adversarial",
        "expected_s4_audit": "detect_error",
        "reason": "S3-MC acierta en clean y falla en noisy/adversarial; buen caso de regresión persistente.",
    },
]


FULL10_EXTRA = [
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0018",
        "focus_case_type": "adversarial_only_s3_mc_correct",
        "expected_s4_audit": "preserve_correct",
        "reason": "S3-MC rescata adversarial cuando S1/S2 fallan; útil para medir preservación de aciertos.",
    },
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0045",
        "focus_case_type": "adversarial_only_s3_mc_correct",
        "expected_s4_audit": "preserve_correct",
        "reason": "S3-MC acierta en adversarial aunque clean falla; caso interesante de recuperación activa.",
    },
    {
        "condition": "adversarial",
        "question_id": "musique_mc__0024",
        "focus_case_type": "s3_mc_regresses_adversarial",
        "expected_s4_audit": "detect_error",
        "reason": "Todos aciertan en clean/noisy, pero S3-MC falla en adversarial.",
    },
    {
        "condition": "noisy",
        "question_id": "musique_mc__0006",
        "focus_case_type": "s3_mc_regresses_noisy",
        "expected_s4_audit": "detect_error",
        "reason": "S3-MC falla en noisy pero se recupera en adversarial; caso útil para analizar sensibilidad a ruido.",
    },
    {
        "condition": "noisy",
        "question_id": "musique_mc__0011",
        "focus_case_type": "s3_mc_regresses_noisy",
        "expected_s4_audit": "detect_error",
        "reason": "S3-MC falla solo en noisy; caso puntual de regresión por ruido.",
    },
]


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def norm_choice(x: Any) -> str:
    text = clean_text(x).upper()
    return text if text in VALID_OPTIONS else ""


def coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    text = clean_text(x).lower()
    if text in {"true", "1", "yes", "y", "correct"}:
        return True
    if text in {"false", "0", "no", "n", "incorrect"}:
        return False
    return False


def first_present(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row.index:
            value = clean_text(row.get(col, ""))
            if value:
                return value
    return ""


def option_text(row: pd.Series, choice: str) -> str:
    choice = norm_choice(choice)
    if not choice:
        return ""
    return clean_text(row.get(choice, ""))


def build_question_with_options(row: pd.Series) -> str:
    q = first_present(row, ["original_question", "retrieval_query", "question", "prompt"])

    option_block = "\n".join(
        f"{label}. {clean_text(row.get(label, ''))}"
        for label in ["A", "B", "C", "D"]
        if clean_text(row.get(label, ""))
    )

    return f"""MuSiQue multiple-choice question.

Question:
{q}

Options:
{option_block}

Instruction:
Choose the best answer among A, B, C, and D."""


def build_s4_initial_answer(row: pd.Series, final_choice: str) -> str:
    selected_text = option_text(row, final_choice)
    rationale = first_present(row, ["final_rationale", "rationale", "candidate_rationale"])

    if selected_text and rationale:
        return f"Proposed answer: {selected_text}. Evidence-based rationale: {rationale}"

    if selected_text:
        return f"Proposed answer: {selected_text}."

    if final_choice:
        return f"Proposed answer option: {final_choice}."

    return first_present(row, ["parsed_answer", "final_answer", "answer"])


def find_id_column(df: pd.DataFrame) -> str:
    for col in ["id", "question_id", "example_id"]:
        if col in df.columns:
            return col
    raise ValueError(f"No encontré columna ID. Columnas: {list(df.columns)}")


def find_correct_column(df: pd.DataFrame) -> str | None:
    for col in ["eval_correct", "mc_correct", "correct", "is_correct"]:
        if col in df.columns:
            return col
    return None


def load_condition_frames(base_dir: Path, condition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = base_dir / condition / "s3_mc_raw.csv"
    eval_path = base_dir / condition / "s3_mc_evaluated.csv"

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    raw = pd.read_csv(raw_path)
    ev = pd.read_csv(eval_path)

    return raw, ev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini"),
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/eval_mc/robustness_musique/questions.csv"),
    )
    parser.add_argument(
        "--preset",
        choices=["core5", "full10"],
        default="core5",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/input/s4_robustness_focus_core5.csv"),
    )
    args = parser.parse_args()

    selected = list(CORE5)
    if args.preset == "full10":
        selected.extend(FULL10_EXTRA)

    if args.preset == "full10" and "core5" in str(args.output_path):
        args.output_path = args.output_path.with_name("s4_robustness_focus_full10.csv")

    questions = pd.read_csv(args.questions_path)
    qid_col = find_id_column(questions)
    questions["_join_qid"] = questions[qid_col].astype(str)

    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []

    for item in selected:
        condition = item["condition"]
        qid = item["question_id"]

        if condition not in cache:
            cache[condition] = load_condition_frames(args.base_dir, condition)

        raw, ev = cache[condition]

        raw_id_col = find_id_column(raw)
        ev_id_col = find_id_column(ev)

        raw_matches = raw[raw[raw_id_col].astype(str) == qid]
        ev_matches = ev[ev[ev_id_col].astype(str) == qid]
        q_matches = questions[questions["_join_qid"].astype(str) == qid]

        if raw_matches.empty:
            raise ValueError(f"No encontré {qid} en raw S3-MC para condition={condition}")
        if ev_matches.empty:
            raise ValueError(f"No encontré {qid} en evaluated S3-MC para condition={condition}")
        if q_matches.empty:
            raise ValueError(f"No encontré {qid} en questions.csv")

        raw_row = raw_matches.iloc[0]
        ev_row = ev_matches.iloc[0]
        q_row = q_matches.iloc[0]

        # Combinamos primero questions y luego raw, para que raw pueda pisar columnas diagnósticas.
        combined = q_row.to_dict()
        combined.update(raw_row.to_dict())
        row = pd.Series(combined)

        gold = norm_choice(first_present(row, ["gold_answer", "expected_answer"]))
        final_choice = norm_choice(first_present(row, ["final_answer", "parsed_answer", "answer"]))

        correct_col = find_correct_column(ev)
        if correct_col:
            s3_correct = coerce_bool(ev_row.get(correct_col))
        else:
            s3_correct = bool(final_choice and gold and final_choice == gold)

        s4_question = build_question_with_options(row)
        s4_initial_answer = build_s4_initial_answer(row, final_choice)

        output_row = {
            "id": f"{qid}__{condition}__s3_mc",
            "original_id": qid,
            "question_id": qid,
            "source_condition": condition,
            "source_system": "s3_mc",
            "source_system_for_s4": "s3_mc",
            "task_type": "multiple_choice",
            "question_format": "multiple_choice",
            "s4_focus_case_type": item["focus_case_type"],
            "expected_s4_audit": item["expected_s4_audit"],
            "expected_s4_suspicious": item["expected_s4_audit"] == "detect_error",
            "focus_reason": item["reason"],

            # Campos principales que consume S4.
            "question": s4_question,
            "parsed_answer": s4_initial_answer,
            "expected_final_behavior": "answer",
            "s4_expected_behavior": "audit",

            # Metadata MC.
            "original_question": clean_text(q_row.get("original_question", "")),
            "retrieval_query": clean_text(q_row.get("retrieval_query", "")),
            "A": clean_text(q_row.get("A", "")),
            "B": clean_text(q_row.get("B", "")),
            "C": clean_text(q_row.get("C", "")),
            "D": clean_text(q_row.get("D", "")),
            "gold_answer": gold,
            "gold_answer_text": option_text(q_row, gold),

            # Estado de S3-MC.
            "s4_mc_gold_choice": gold,
            "s4_mc_final_choice": final_choice,
            "s4_mc_final_option_text": option_text(q_row, final_choice),
            "s3_mc_correct": s3_correct,
            "source_s3_mc_candidate_answer": first_present(row, ["candidate_answer"]),
            "source_s3_mc_final_answer": first_present(row, ["final_answer", "parsed_answer", "answer"]),
            "source_s3_mc_candidate_rationale": first_present(row, ["candidate_rationale"]),
            "source_s3_mc_final_rationale": first_present(row, ["final_rationale", "rationale"]),
            "predicted_route": first_present(row, ["predicted_route", "route"]),
            "active_retrieval_triggered": first_present(row, ["active_retrieval_triggered"]),
        }

        # Evidencia inicial para S4, si existe en el raw de S3-MC.
        for col in [
            "retrieved_context_json",
            "retrieved_chunk_ids",
            "retrieved_doc_ids",
            "retrieved_titles",
            "retrieved_scores",
            "candidate_retrieved_context_json",
            "final_retrieved_context_json",
        ]:
            if col in row.index:
                output_row[col] = clean_text(row.get(col, ""))

        rows.append(output_row)

    out = pd.DataFrame(rows)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)

    report_path = args.output_path.with_suffix(".txt")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("S4 robustness focus input\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Preset: {args.preset}\n")
        f.write(f"Rows: {len(out)}\n")
        f.write(f"Output CSV: {args.output_path}\n")
        f.write(f"Report TXT: {report_path}\n\n")
        f.write("Expected S4 audit counts:\n")
        f.write(out["expected_s4_audit"].value_counts(dropna=False).to_string())
        f.write("\n\n")
        f.write("Rows:\n")
        preview_cols = [
            "id",
            "source_condition",
            "s4_focus_case_type",
            "gold_answer",
            "s4_mc_final_choice",
            "s3_mc_correct",
            "expected_s4_audit",
        ]
        f.write(out[preview_cols].to_string(index=False))
        f.write("\n")

    print("S4 robustness focus input creado")
    print("--------------------------------")
    print(f"Preset: {args.preset}")
    print(f"Rows: {len(out)}")
    print(f"CSV: {args.output_path}")
    print(f"TXT: {report_path}")
    print()
    print(out[[
        "id",
        "source_condition",
        "s4_focus_case_type",
        "gold_answer",
        "s4_mc_final_choice",
        "s3_mc_correct",
        "expected_s4_audit",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()

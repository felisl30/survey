#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RAW_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_raw.csv")
DEFAULT_EVAL_PATH = Path("outputs/eval_mc/musique_mc_rag/s3_mc/s3_gpt_5_mini_flare_like_evaluated.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/eval_mc/musique_mc_rag/s4_mc/input/s4_mc_from_s3_mc_focus.csv")


VALID_OPTIONS = {"A", "B", "C", "D"}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def norm_choice(x) -> str:
    x = clean_text(x).upper()
    return x if x in VALID_OPTIONS else ""


def option_text(row: pd.Series, choice: str) -> str:
    choice = norm_choice(choice)
    if not choice:
        return ""
    return clean_text(row.get(choice, ""))


def build_question_with_options(row: pd.Series) -> str:
    question = clean_text(row.get("original_question", "")) or clean_text(row.get("question", ""))

    options = []
    for label in ["A", "B", "C", "D"]:
        text = clean_text(row.get(label, ""))
        if text:
            options.append(f"{label}. {text}")

    option_block = "\n".join(options)

    return f"""MuSiQue multiple-choice question.

Question:
{question}

Options:
{option_block}

Instruction:
Choose the best answer among A, B, C, and D."""


def build_s4_initial_answer(row: pd.Series) -> str:
    final_choice = norm_choice(row.get("final_answer", ""))
    selected_text = option_text(row, final_choice)
    rationale = clean_text(row.get("final_rationale", ""))

    if final_choice and selected_text and rationale:
        return (
            f"The selected answer is {final_choice}: {selected_text}. "
            f"Rationale: {rationale}"
        )

    if final_choice and selected_text:
        return f"The selected answer is {final_choice}: {selected_text}."

    if final_choice:
        return f"The selected answer is {final_choice}."

    return clean_text(row.get("final_answer", ""))


def classify_case(row: pd.Series) -> str:
    gold = norm_choice(row.get("gold_answer", ""))
    candidate = norm_choice(row.get("candidate_answer", ""))
    final = norm_choice(row.get("final_answer", ""))
    route = clean_text(row.get("predicted_route", ""))

    candidate_correct = candidate == gold
    final_correct = final == gold

    if route == "retrieve" and not candidate_correct and final_correct:
        return "retrieval_corrected_candidate"
    if route == "retrieve" and candidate_correct and not final_correct:
        return "retrieval_regressed_candidate"
    if final_correct:
        return "final_correct"
    return "final_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--mode",
        choices=["focus", "all"],
        default="focus",
        help="focus = wrong + regressed + corrected cases; all = all 100 cases.",
    )
    args = parser.parse_args()

    if not args.raw_path.exists():
        raise FileNotFoundError(args.raw_path)
    if not args.eval_path.exists():
        raise FileNotFoundError(args.eval_path)

    raw = pd.read_csv(args.raw_path)
    ev = pd.read_csv(args.eval_path)

    keep_eval_cols = ["id", "eval_correct", "parsed_answer", "valid_format"]
    keep_eval_cols = [c for c in keep_eval_cols if c in ev.columns]

    df = raw.merge(ev[keep_eval_cols], on="id", how="left", suffixes=("", "_eval"))

    df["s4_mc_case_type"] = df.apply(classify_case, axis=1)
    df["s4_mc_gold_choice"] = df["gold_answer"].map(norm_choice)
    df["s4_mc_candidate_choice"] = df["candidate_answer"].map(norm_choice)
    df["s4_mc_final_choice"] = df["final_answer"].map(norm_choice)
    df["s4_mc_gold_option_text"] = df.apply(lambda r: option_text(r, r["s4_mc_gold_choice"]), axis=1)
    df["s4_mc_final_option_text"] = df.apply(lambda r: option_text(r, r["s4_mc_final_choice"]), axis=1)

    # Lo que va a ver S4.
    df["question"] = df.apply(build_question_with_options, axis=1)
    df["parsed_answer"] = df.apply(build_s4_initial_answer, axis=1)
    df["expected_final_behavior"] = "answer"
    df["source_system"] = "s3_mc"
    df["source_system_for_s4"] = "s3_mc"
    df["task_type"] = "multiple_choice"
    df["question_format"] = "multiple_choice"
    df["s4_expected_behavior"] = "answer"

    # Para evaluación MC posterior, conservamos la letra original de S3-MC.
    df["source_s3_mc_final_answer"] = df["final_answer"]
    df["source_s3_mc_candidate_answer"] = df["candidate_answer"]
    df["source_s3_mc_final_rationale"] = df["final_rationale"]
    df["source_s3_mc_candidate_rationale"] = df["candidate_rationale"]

    if args.mode == "focus":
        order = {
            "final_wrong": 0,
            "retrieval_regressed_candidate": 1,
            "retrieval_corrected_candidate": 2,
            "final_correct": 3,
        }
        df["_sort"] = df["s4_mc_case_type"].map(order).fillna(99)
        focus = df[df["s4_mc_case_type"].isin([
            "final_wrong",
            "retrieval_regressed_candidate",
            "retrieval_corrected_candidate",
        ])].copy()
        focus = focus.sort_values(["_sort", "id"]).drop(columns=["_sort"])
        out = focus
    else:
        out = df.copy()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print("S4-MC input creado")
    print("------------------")
    print("Output:", args.output_path)
    print("Rows:", len(out))
    print("\nCase types:")
    print(out["s4_mc_case_type"].value_counts(dropna=False).to_string())

    cols = [
        "id",
        "s4_mc_case_type",
        "gold_answer",
        "candidate_answer",
        "final_answer",
        "predicted_route",
        "parsed_answer",
    ]
    cols = [c for c in cols if c in out.columns]

    print("\nPreview:")
    print(out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

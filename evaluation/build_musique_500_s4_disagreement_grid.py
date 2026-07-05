#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from build_musique_500_s5_router import SYSTEMS, boolish, clean, model_tag, number, result_path


def option_text(row: pd.Series, choice: str) -> str:
    choice = clean(choice)
    if choice in {"A", "B", "C", "D"}:
        return clean(row.get(choice, ""))
    return ""


def build_question_with_options(row: pd.Series) -> str:
    question = clean(row.get("original_question", "")) or clean(row.get("question", ""))
    lines = [
        question,
        "",
        "Options:",
        f"A. {clean(row.get('A', ''))}",
        f"B. {clean(row.get('B', ''))}",
        f"C. {clean(row.get('C', ''))}",
        f"D. {clean(row.get('D', ''))}",
    ]
    return "\n".join(lines)


def load_eval(model: str, system: str, s0_out_dir: Path, rag_root: Path) -> pd.DataFrame | None:
    path = result_path(model, system, s0_out_dir, rag_root)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    correct_col = "mc_correct" if system == "s0" and "mc_correct" in df.columns else "eval_correct"
    if correct_col not in df.columns:
        for candidate in ["correct", "is_correct"]:
            if candidate in df.columns:
                correct_col = candidate
                break
    out = pd.DataFrame()
    out["id"] = df["id"].astype(str)
    out["model"] = model
    out["model_tag"] = model_tag(model)
    out["system"] = system
    out["answer"] = df.get("parsed_answer", df.get("final_answer", df.get("answer", ""))).map(clean)
    out["correct"] = df[correct_col].map(boolish) if correct_col in df.columns else False
    out["confidence"] = pd.to_numeric(df.get("parsed_confidence", df.get("final_confidence", pd.NA)), errors="coerce")
    out["total_tokens"] = pd.to_numeric(df.get("total_tokens", pd.NA), errors="coerce")
    out["latency_seconds"] = pd.to_numeric(df.get("latency_seconds", pd.NA), errors="coerce")

    for col in [
        "original_question",
        "question",
        "retrieval_query",
        "A",
        "B",
        "C",
        "D",
        "gold_answer",
        "gold_answer_text",
        "retrieved_context_json",
        "retrieved_doc_ids_json",
        "retrieved_titles_json",
        "retrieved_scores_json",
        "predicted_route",
        "active_retrieval_triggered",
        "final_rationale",
        "candidate_rationale",
    ]:
        if col in df.columns:
            out[col] = df[col]
    return out


def build_wide(models: list[str], s0_out_dir: Path, rag_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in models:
        for system in SYSTEMS:
            frame = load_eval(model, system, s0_out_dir, rag_root)
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise SystemExit("No evaluated S0-S3 files found.")

    long_df = pd.concat(frames, ignore_index=True)
    meta_cols = ["id", "model", "model_tag", "original_question", "question", "retrieval_query", "A", "B", "C", "D", "gold_answer", "gold_answer_text"]
    wide = long_df[[c for c in meta_cols if c in long_df.columns]].drop_duplicates(["model", "id"]).copy()

    for system in SYSTEMS:
        sub_cols = ["model", "id", "answer", "correct", "confidence", "total_tokens", "latency_seconds"]
        optional = ["retrieved_context_json", "retrieved_doc_ids_json", "retrieved_titles_json", "retrieved_scores_json", "predicted_route", "active_retrieval_triggered", "final_rationale", "candidate_rationale"]
        sub = long_df[long_df["system"].eq(system)][sub_cols + [c for c in optional if c in long_df.columns]].copy()
        sub = sub.rename(columns={col: f"{system}_{col}" for col in sub.columns if col not in {"model", "id"}})
        wide = wide.merge(sub, on=["model", "id"], how="left")

    return wide


def classify_case(row: pd.Series) -> tuple[str, str, bool]:
    answers = {s: clean(row.get(f"{s}_answer", "")) for s in SYSTEMS}
    correct = {s: boolish(row.get(f"{s}_correct", False)) for s in SYSTEMS}
    present = [s for s, ans in answers.items() if ans]
    unique_answers = {answers[s] for s in present}
    correct_systems = [s for s in present if correct[s]]
    wrong_systems = [s for s in present if not correct[s]]

    if not present:
        return "missing_all", "No system output found.", False
    if len(unique_answers) == 1 and len(correct_systems) == len(present):
        return "all_agree_correct", "All available systems agree and are correct.", False
    if len(unique_answers) == 1 and not correct_systems:
        return "all_agree_wrong", "All available systems agree on the same wrong answer.", True
    if not correct_systems:
        return "all_wrong_disagreement", "Systems disagree, but every available answer is wrong.", True
    if "s3" in correct_systems and len(correct_systems) == 1:
        return "s3_only_correct", "Only S3 fixes the item; useful to inspect retrieval/FLARE benefit.", True
    if "s3" in wrong_systems and correct_systems:
        return "s3_wrong_others_correct", "S3 fails while at least one cheaper system is correct.", True
    if len(unique_answers) > 1 and correct_systems and wrong_systems:
        return "mixed_correctness_disagreement", "At least one system is right and one is wrong.", True
    return "other_disagreement", "Systems disagree, but case is less diagnostic.", len(unique_answers) > 1


def choose_target_system(row: pd.Series, case_type: str) -> str:
    if case_type in {"s3_wrong_others_correct", "all_agree_wrong", "all_wrong_disagreement"} and clean(row.get("s3_answer", "")):
        return "s3"
    wrong = [s for s in SYSTEMS if clean(row.get(f"{s}_answer", "")) and not boolish(row.get(f"{s}_correct", False))]
    if wrong:
        return max(wrong, key=lambda s: number(row.get(f"{s}_total_tokens", 0)))
    if clean(row.get("s3_answer", "")):
        return "s3"
    return next((s for s in SYSTEMS if clean(row.get(f"{s}_answer", ""))), "s0")


def build_grid(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        case_type, reason, selected = classify_case(row)
        target = choose_target_system(row, case_type)
        answers = [clean(row.get(f"{s}_answer", "")) for s in SYSTEMS if clean(row.get(f"{s}_answer", ""))]
        correct_systems = [s for s in SYSTEMS if boolish(row.get(f"{s}_correct", False))]
        wrong_systems = [s for s in SYSTEMS if clean(row.get(f"{s}_answer", "")) and not boolish(row.get(f"{s}_correct", False))]
        out = row.to_dict()
        out.update(
            {
                "s4_focus_selected": selected,
                "s4_focus_case_type": case_type,
                "s4_focus_reason": reason,
                "s4_target_system": target,
                "s4_target_answer": clean(row.get(f"{target}_answer", "")),
                "s4_target_answer_text": option_text(row, clean(row.get(f"{target}_answer", ""))),
                "n_unique_answers": len(set(answers)),
                "correct_systems": "|".join(correct_systems),
                "wrong_systems": "|".join(wrong_systems),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def build_s4_input(grid: pd.DataFrame, per_model: int) -> pd.DataFrame:
    focus = grid[grid["s4_focus_selected"].map(boolish)].copy()
    if focus.empty:
        return focus

    rank_map = {
        "s3_wrong_others_correct": 0,
        "s3_only_correct": 1,
        "mixed_correctness_disagreement": 2,
        "all_wrong_disagreement": 3,
        "all_agree_wrong": 4,
        "other_disagreement": 5,
    }
    focus["_rank"] = focus["s4_focus_case_type"].map(rank_map).fillna(9)
    focus = focus.sort_values(["model", "_rank", "id"])
    if per_model > 0:
        focus = focus.groupby("model", group_keys=False).head(per_model)

    rows: list[dict[str, Any]] = []
    for _, row in focus.iterrows():
        target = clean(row.get("s4_target_system", ""))
        selected_choice = clean(row.get("s4_target_answer", ""))
        selected_text = option_text(row, selected_choice)
        rows.append(
            {
                "id": f"{clean(row.get('id'))}__{clean(row.get('model_tag'))}__{target}",
                "original_id": clean(row.get("id")),
                "model": clean(row.get("model")),
                "source_system": target,
                "source_system_for_s4": target,
                "task_type": "multiple_choice",
                "question_format": "multiple_choice",
                "s4_focus_case_type": clean(row.get("s4_focus_case_type")),
                "expected_s4_audit": "detect_or_confirm_answer",
                "focus_reason": clean(row.get("s4_focus_reason")),
                "question": build_question_with_options(row),
                "parsed_answer": f"Proposed answer: {selected_choice}. {selected_text}",
                "original_question": clean(row.get("original_question", "")),
                "retrieval_query": clean(row.get("retrieval_query", "")),
                "A": clean(row.get("A", "")),
                "B": clean(row.get("B", "")),
                "C": clean(row.get("C", "")),
                "D": clean(row.get("D", "")),
                "gold_answer": clean(row.get("gold_answer", "")),
                "gold_answer_text": clean(row.get("gold_answer_text", "")),
                "s4_mc_final_choice": selected_choice,
                "s4_mc_final_option_text": selected_text,
                "s4_mc_gold_choice": clean(row.get("gold_answer", "")),
                "correct_systems": clean(row.get("correct_systems", "")),
                "wrong_systems": clean(row.get("wrong_systems", "")),
                "retrieved_context_json": clean(row.get(f"{target}_retrieved_context_json", "")),
                "retrieved_doc_ids_json": clean(row.get(f"{target}_retrieved_doc_ids_json", "")),
                "retrieved_titles_json": clean(row.get(f"{target}_retrieved_titles_json", "")),
                "retrieved_scores_json": clean(row.get(f"{target}_retrieved_scores_json", "")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gpt-5-nano,gpt-5-mini,gpt-4.1-mini")
    parser.add_argument("--s0-out-dir", type=Path, default=Path("outputs/eval_mc"))
    parser.add_argument("--rag-root", type=Path, default=Path("outputs/eval_mc/musique_mc_rag_500"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_mc/musique_mc_rag_500/posthoc"))
    parser.add_argument("--s4-per-model", type=int, default=25)
    args = parser.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wide = build_wide(models, args.s0_out_dir, args.rag_root)
    grid = build_grid(wide)
    s4_input = build_s4_input(grid, args.s4_per_model)

    grid_path = args.output_dir / "s4_disagreement_grid.csv"
    focus_path = args.output_dir / "s4_focus_candidates.csv"
    input_path = args.output_dir / "s4_focus_input.csv"

    grid.to_csv(grid_path, index=False)
    grid[grid["s4_focus_selected"].map(boolish)].to_csv(focus_path, index=False)
    s4_input.to_csv(input_path, index=False)

    summary = (
        grid.groupby(["model", "s4_focus_case_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model", "s4_focus_case_type"])
    )
    summary.to_csv(args.output_dir / "s4_disagreement_summary.csv", index=False)

    print(f"S4 grid:    {grid_path}")
    print(f"S4 focus:   {focus_path}")
    print(f"S4 input:   {input_path}")
    print(f"S4 summary: {args.output_dir / 's4_disagreement_summary.csv'}")


if __name__ == "__main__":
    main()

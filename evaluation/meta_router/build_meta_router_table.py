#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROBUSTNESS_DIR = Path("outputs/eval_mc/robustness_musique/gpt_5_4_mini")
QUESTIONS_PATH = Path("data/eval_mc/robustness_musique/questions.csv")

OUT_DIR = Path("outputs/eval_mc/meta_router")
OUT_CSV = OUT_DIR / "meta_router_question_table.csv"
OUT_REPORT = OUT_DIR / "meta_router_question_table_report.md"
DOWNLOADS_REPORT = Path.home() / "Downloads" / "meta_router_question_table_report.md"

CONDITIONS = ["clean", "noisy", "adversarial"]
SYSTEMS = ["s0", "s1", "s2", "s3_mc"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def first_existing_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def get_col(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    col = first_existing_col(df, names)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def boolish(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "si", "sí"}


def parse_json_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s or s == "[]":
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def score_features_from_json(x) -> dict:
    values = []
    obj = parse_json_list(x)

    for item in obj:
        if isinstance(item, dict):
            score = item.get("score", item.get("similarity", item.get("retrieval_score")))
        else:
            score = item
        try:
            values.append(float(score))
        except Exception:
            pass

    if not values:
        return {
            "retrieved_top1_score": pd.NA,
            "retrieved_top2_score": pd.NA,
            "retrieved_top1_top2_gap": pd.NA,
            "retrieved_top5_mean_score": pd.NA,
        }

    top1 = values[0]
    top2 = values[1] if len(values) > 1 else pd.NA
    gap = top1 - top2 if not pd.isna(top2) else pd.NA
    top5_mean = sum(values[:5]) / min(len(values), 5)

    return {
        "retrieved_top1_score": top1,
        "retrieved_top2_score": top2,
        "retrieved_top1_top2_gap": gap,
        "retrieved_top5_mean_score": top5_mean,
    }


def extract_system_frame(path: Path, prefix: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    if "id" not in df.columns:
        raise ValueError(f"El archivo no tiene columna id: {path}")

    out = pd.DataFrame()
    out["id"] = df["id"]

    out[f"{prefix}_answer"] = get_col(df, ["parsed_answer"])
    out[f"{prefix}_confidence"] = get_col(df, ["parsed_confidence"])
    out[f"{prefix}_correct"] = get_col(df, ["eval_correct"])
    out[f"{prefix}_valid_format"] = get_col(df, ["valid_format"])
    out[f"{prefix}_run_error"] = get_col(df, ["run_error_present"])
    out[f"{prefix}_tokens"] = get_col(df, ["total_tokens"])
    out[f"{prefix}_input_tokens"] = get_col(df, ["input_tokens"])
    out[f"{prefix}_output_tokens"] = get_col(df, ["output_tokens"])
    out[f"{prefix}_latency_seconds"] = get_col(df, ["latency_seconds"])

    out[f"{prefix}_n_docs_retrieved"] = get_col(df, ["n_docs_retrieved"], default=0)
    out[f"{prefix}_retrieved_doc_ids_json"] = get_col(df, ["retrieved_doc_ids_json"], default="[]")
    out[f"{prefix}_retrieved_scores_json"] = get_col(df, ["retrieved_scores_json"], default="[]")

    score_rows = out[f"{prefix}_retrieved_scores_json"].apply(score_features_from_json).apply(pd.Series)
    for c in score_rows.columns:
        out[f"{prefix}_{c}"] = score_rows[c]

    if prefix == "s2":
        out["s2_route"] = get_col(df, ["predicted_route", "parsed_route"])
        out["s2_router_reason"] = get_col(df, ["router_reason"])
        out["s2_top1_score"] = get_col(df, ["top1_score"])
        out["s2_top2_score"] = get_col(df, ["top2_score"])
        out["s2_top3_score"] = get_col(df, ["top3_score"])
        out["s2_top1_top2_gap"] = get_col(df, ["top1_top2_gap"])
        out["s2_top3_mean_score"] = get_col(df, ["top3_mean_score"])
        out["s2_top5_mean_score"] = get_col(df, ["top5_mean_score"])
        out["s2_policy_name"] = get_col(df, ["s2_policy_name"])
        out["s2_policy_threshold"] = get_col(df, ["s2_policy_threshold"])
        out["s2_policy_min_gap"] = get_col(df, ["s2_policy_min_gap"])

        def retrieved(row) -> bool:
            route = str(row.get("s2_route", "")).strip().lower()
            n_docs = row.get("s2_n_docs_retrieved", 0)
            try:
                n_docs = int(n_docs)
            except Exception:
                n_docs = 0
            return "retrieve" in route or n_docs > 0

        out["s2_retrieved"] = out.apply(retrieved, axis=1)

    if prefix == "s3_mc":
        active_col = first_existing_col(
            df,
            [
                "s3_mc_active_retrieval",
                "active_retrieval",
                "used_retrieval",
                "retrieval_triggered",
                "did_retrieve",
            ],
        )
        route_col = first_existing_col(
            df,
            [
                "s3_mc_route",
                "route",
                "predicted_route",
                "parsed_route",
            ],
        )

        if active_col is not None:
            out["s3_mc_active_retrieval"] = df[active_col]
        else:
            out["s3_mc_active_retrieval"] = out["s3_mc_n_docs_retrieved"].fillna(0).astype(int) > 0

        if route_col is not None:
            out["s3_mc_route"] = df[route_col]
        else:
            out["s3_mc_route"] = pd.NA

    return out


def oracle_row(row: pd.Series) -> pd.Series:
    correct_systems = []
    for system in SYSTEMS:
        correct = boolish(row.get(f"{system}_correct"))
        if not correct:
            continue

        tokens_raw = row.get(f"{system}_tokens")
        try:
            tokens = float(tokens_raw)
        except Exception:
            tokens = math.inf

        correct_systems.append((system, tokens, row.get(f"{system}_answer")))

    if not correct_systems:
        return pd.Series(
            {
                "oracle_any_correct": False,
                "oracle_correct_systems": "",
                "oracle_n_correct_systems": 0,
                "oracle_min_cost_system": pd.NA,
                "oracle_min_cost_answer": pd.NA,
                "oracle_min_cost_tokens": pd.NA,
            }
        )

    correct_systems_sorted = sorted(
        correct_systems,
        key=lambda x: (
            math.inf if pd.isna(x[1]) else x[1],
            SYSTEMS.index(x[0]) if x[0] in SYSTEMS else 999,
        ),
    )

    best = correct_systems_sorted[0]

    return pd.Series(
        {
            "oracle_any_correct": True,
            "oracle_correct_systems": ",".join(x[0] for x in correct_systems_sorted),
            "oracle_n_correct_systems": len(correct_systems_sorted),
            "oracle_min_cost_system": best[0],
            "oracle_min_cost_answer": best[2],
            "oracle_min_cost_tokens": best[1],
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    require_file(QUESTIONS_PATH)
    questions = pd.read_csv(QUESTIONS_PATH)

    base_cols = [
        "id",
        "question_id",
        "dataset",
        "case_type",
        "difficulty",
        "original_question",
        "question",
        "A",
        "B",
        "C",
        "D",
        "gold_answer",
        "gold_answer_text",
        "expected_answer",
        "requires_retrieval",
    ]
    base_cols = [c for c in base_cols if c in questions.columns]

    s0 = extract_system_frame(ROBUSTNESS_DIR / "s0_evaluated.csv", "s0")

    frames = []

    for condition in CONDITIONS:
        frame = questions[base_cols].copy()
        frame["condition"] = condition
        frame["model"] = "gpt-5.4-mini"

        frame = frame.merge(s0, on="id", how="left", validate="one_to_one")

        for system in ["s1", "s2", "s3_mc"]:
            path = ROBUSTNESS_DIR / condition / f"{system}_evaluated.csv"
            sys_frame = extract_system_frame(path, system)
            frame = frame.merge(sys_frame, on="id", how="left", validate="one_to_one")

        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)

    table["agreement_s1_s2"] = table["s1_answer"].astype(str) == table["s2_answer"].astype(str)
    table["agreement_s1_s3_mc"] = table["s1_answer"].astype(str) == table["s3_mc_answer"].astype(str)
    table["agreement_s2_s3_mc"] = table["s2_answer"].astype(str) == table["s3_mc_answer"].astype(str)
    table["agreement_s0_s1"] = table["s0_answer"].astype(str) == table["s1_answer"].astype(str)
    table["agreement_s0_s2"] = table["s0_answer"].astype(str) == table["s2_answer"].astype(str)
    table["agreement_s0_s3_mc"] = table["s0_answer"].astype(str) == table["s3_mc_answer"].astype(str)
    table["all_s1_s2_s3_agree"] = (
        table["agreement_s1_s2"] & table["agreement_s1_s3_mc"] & table["agreement_s2_s3_mc"]
    )
    table["all_s0_s1_s2_s3_agree"] = (
        table["agreement_s0_s1"]
        & table["agreement_s0_s2"]
        & table["agreement_s0_s3_mc"]
        & table["all_s1_s2_s3_agree"]
    )

    oracle = table.apply(oracle_row, axis=1)
    table = pd.concat([table, oracle], axis=1)

    for system in SYSTEMS:
        table[f"{system}_correct_bool"] = table[f"{system}_correct"].apply(boolish)

    table["none_correct"] = ~(
        table["s0_correct_bool"]
        | table["s1_correct_bool"]
        | table["s2_correct_bool"]
        | table["s3_mc_correct_bool"]
    )
    table["only_s1_correct"] = (
        table["s1_correct_bool"]
        & ~table["s0_correct_bool"]
        & ~table["s2_correct_bool"]
        & ~table["s3_mc_correct_bool"]
    )
    table["only_s2_correct"] = (
        table["s2_correct_bool"]
        & ~table["s0_correct_bool"]
        & ~table["s1_correct_bool"]
        & ~table["s3_mc_correct_bool"]
    )
    table["only_s3_mc_correct"] = (
        table["s3_mc_correct_bool"]
        & ~table["s0_correct_bool"]
        & ~table["s1_correct_bool"]
        & ~table["s2_correct_bool"]
    )

    table.to_csv(OUT_CSV, index=False)

    report_lines = []
    report_lines.append("# Meta-Router Question Table Report")
    report_lines.append("")
    report_lines.append("## Archivos generados")
    report_lines.append("")
    report_lines.append(f"- `{OUT_CSV}`")
    report_lines.append(f"- `{OUT_REPORT}`")
    report_lines.append("")
    report_lines.append("## Shape")
    report_lines.append("")
    report_lines.append(f"- Filas: `{len(table)}`")
    report_lines.append(f"- Columnas: `{len(table.columns)}`")
    report_lines.append("")
    report_lines.append("## Filas por condición")
    report_lines.append("")
    report_lines.append(table["condition"].value_counts().sort_index().to_markdown())
    report_lines.append("")

    report_lines.append("## Accuracy por sistema y condición")
    report_lines.append("")
    rows = []
    for condition in CONDITIONS:
        sub = table[table["condition"] == condition]
        for system in SYSTEMS:
            rows.append(
                {
                    "condition": condition,
                    "system": system,
                    "n": len(sub),
                    "accuracy": sub[f"{system}_correct_bool"].mean(),
                    "avg_tokens": pd.to_numeric(sub[f"{system}_tokens"], errors="coerce").mean(),
                }
            )
    summary = pd.DataFrame(rows)
    report_lines.append(summary.to_markdown(index=False))
    report_lines.append("")

    report_lines.append("## Oracle mínimo costo")
    report_lines.append("")
    oracle_rows = []
    for condition in CONDITIONS:
        sub = table[table["condition"] == condition]
        oracle_rows.append(
            {
                "condition": condition,
                "oracle_accuracy": sub["oracle_any_correct"].mean(),
                "oracle_avg_min_cost_tokens": pd.to_numeric(
                    sub["oracle_min_cost_tokens"], errors="coerce"
                ).mean(),
                "none_correct_rate": sub["none_correct"].mean(),
                "only_s1_correct": int(sub["only_s1_correct"].sum()),
                "only_s2_correct": int(sub["only_s2_correct"].sum()),
                "only_s3_mc_correct": int(sub["only_s3_mc_correct"].sum()),
            }
        )
    oracle_summary = pd.DataFrame(oracle_rows)
    report_lines.append(oracle_summary.to_markdown(index=False))
    report_lines.append("")

    report_lines.append("## Señales útiles disponibles para S5")
    report_lines.append("")
    useful_cols = [
        "condition",
        "s2_route",
        "s2_retrieved",
        "s2_top1_score",
        "s2_top1_top2_gap",
        "s2_top5_mean_score",
        "s3_mc_active_retrieval",
        "s3_mc_confidence",
        "agreement_s1_s2",
        "agreement_s1_s3_mc",
        "agreement_s2_s3_mc",
        "oracle_min_cost_system",
    ]
    useful_cols = [c for c in useful_cols if c in table.columns]
    for col in useful_cols:
        report_lines.append(f"- `{col}`")
    report_lines.append("")

    report_lines.append("## Preview")
    report_lines.append("")
    preview_cols = [
        "id",
        "condition",
        "gold_answer",
        "s0_answer",
        "s0_correct_bool",
        "s1_answer",
        "s1_correct_bool",
        "s2_answer",
        "s2_correct_bool",
        "s2_route",
        "s2_retrieved",
        "s3_mc_answer",
        "s3_mc_correct_bool",
        "s3_mc_active_retrieval",
        "oracle_min_cost_system",
        "oracle_correct_systems",
    ]
    preview_cols = [c for c in preview_cols if c in table.columns]
    report_lines.append("```text")
    report_lines.append(table[preview_cols].head(15).to_string(index=False))
    report_lines.append("```")
    report_lines.append("")

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    DOWNLOADS_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK: tabla generada en {OUT_CSV}")
    print(f"OK: reporte generado en {OUT_REPORT}")
    print(f"OK: copia del reporte en {DOWNLOADS_REPORT}")
    print(f"shape: {table.shape}")


if __name__ == "__main__":
    main()

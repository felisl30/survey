#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


IN_CSV = Path("outputs/eval_mc/meta_router/meta_router_question_table.csv")

OUT_DIR = Path("outputs/eval_mc/meta_router")
OUT_SUMMARY = OUT_DIR / "oracle_router_summary.csv"
OUT_BY_CONDITION = OUT_DIR / "oracle_router_by_condition.csv"
OUT_SELECTION_DISTRIBUTION = OUT_DIR / "oracle_router_selection_distribution.csv"
OUT_PATTERN_SUMMARY = OUT_DIR / "oracle_router_pattern_summary.csv"
OUT_REPORT = OUT_DIR / "oracle_router_report.md"
DOWNLOADS_REPORT = Path.home() / "Downloads" / "oracle_router_report.md"

SYSTEMS = ["s0", "s1", "s2", "s3_mc"]
CONDITIONS_ORDER = ["clean", "noisy", "adversarial"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def boolish(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def system_metrics(df: pd.DataFrame, system: str) -> dict:
    correct_col = f"{system}_correct_bool"
    if correct_col not in df.columns:
        correct_col = f"{system}_correct"

    tokens_col = f"{system}_tokens"

    correct = df[correct_col].apply(boolish)
    tokens = pd.to_numeric(df[tokens_col], errors="coerce") if tokens_col in df.columns else pd.Series(dtype=float)

    avg_tokens = float(tokens.mean()) if len(tokens) else float("nan")
    accuracy = float(correct.mean()) if len(correct) else float("nan")

    return {
        "system": system,
        "n": len(df),
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "accuracy_per_1000_tokens": accuracy * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
    }


def summarize_scope(df: pd.DataFrame, condition: str) -> dict:
    rows = [system_metrics(df, s) for s in SYSTEMS]
    best_single = max(rows, key=lambda r: r["accuracy"])

    oracle_accuracy = float(df["oracle_any_correct"].apply(boolish).mean())
    oracle_tokens = safe_mean(df["oracle_min_cost_tokens"])
    oracle_acc_per_1000 = oracle_accuracy * 1000 / oracle_tokens if oracle_tokens and oracle_tokens > 0 else float("nan")

    return {
        "condition": condition,
        "n": len(df),
        "best_single_system": best_single["system"],
        "best_single_accuracy": best_single["accuracy"],
        "best_single_avg_tokens": best_single["avg_tokens"],
        "oracle_accuracy": oracle_accuracy,
        "oracle_avg_min_cost_tokens": oracle_tokens,
        "oracle_accuracy_per_1000_tokens": oracle_acc_per_1000,
        "oracle_gain_vs_best_single": oracle_accuracy - best_single["accuracy"],
        "oracle_gain_vs_s1": oracle_accuracy - rows[1]["accuracy"],
        "oracle_gain_vs_s2": oracle_accuracy - rows[2]["accuracy"],
        "oracle_gain_vs_s3_mc": oracle_accuracy - rows[3]["accuracy"],
        "none_correct_rate": float(df["none_correct"].apply(boolish).mean()) if "none_correct" in df.columns else float("nan"),
        "all_s1_s2_s3_agree_rate": float(df["all_s1_s2_s3_agree"].apply(boolish).mean()) if "all_s1_s2_s3_agree" in df.columns else float("nan"),
        "all_s0_s1_s2_s3_agree_rate": float(df["all_s0_s1_s2_s3_agree"].apply(boolish).mean()) if "all_s0_s1_s2_s3_agree" in df.columns else float("nan"),
    }


def build_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    patterns = []

    def add(name: str, mask: pd.Series):
        patterns.append(
            {
                "pattern": name,
                "n": int(mask.sum()),
                "rate": float(mask.mean()),
            }
        )

    correct = {s: df[f"{s}_correct_bool"].apply(boolish) for s in SYSTEMS}

    add("none_correct", ~(correct["s0"] | correct["s1"] | correct["s2"] | correct["s3_mc"]))
    add("all_four_correct", correct["s0"] & correct["s1"] & correct["s2"] & correct["s3_mc"])
    add("only_s0_correct", correct["s0"] & ~correct["s1"] & ~correct["s2"] & ~correct["s3_mc"])
    add("only_s1_correct", correct["s1"] & ~correct["s0"] & ~correct["s2"] & ~correct["s3_mc"])
    add("only_s2_correct", correct["s2"] & ~correct["s0"] & ~correct["s1"] & ~correct["s3_mc"])
    add("only_s3_mc_correct", correct["s3_mc"] & ~correct["s0"] & ~correct["s1"] & ~correct["s2"])
    add("s1_correct_s2_wrong", correct["s1"] & ~correct["s2"])
    add("s2_correct_s1_wrong", correct["s2"] & ~correct["s1"])
    add("s3_mc_correct_s1_s2_wrong", correct["s3_mc"] & ~correct["s1"] & ~correct["s2"])
    add("s1_s2_s3_all_correct", correct["s1"] & correct["s2"] & correct["s3_mc"])
    add("s1_s2_s3_all_wrong", ~correct["s1"] & ~correct["s2"] & ~correct["s3_mc"])

    return pd.DataFrame(patterns)


def main() -> None:
    global IN_CSV, OUT_DIR, OUT_SUMMARY, OUT_BY_CONDITION, OUT_SELECTION_DISTRIBUTION, OUT_PATTERN_SUMMARY, OUT_REPORT, DOWNLOADS_REPORT

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=IN_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-suffix", default="")
    args = parser.parse_args()

    IN_CSV = args.input_csv
    OUT_DIR = args.output_dir
    OUT_SUMMARY = OUT_DIR / "oracle_router_summary.csv"
    OUT_BY_CONDITION = OUT_DIR / "oracle_router_by_condition.csv"
    OUT_SELECTION_DISTRIBUTION = OUT_DIR / "oracle_router_selection_distribution.csv"
    OUT_PATTERN_SUMMARY = OUT_DIR / "oracle_router_pattern_summary.csv"
    OUT_REPORT = OUT_DIR / "oracle_router_report.md"
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    DOWNLOADS_REPORT = Path.home() / "Downloads" / f"oracle_router_report{suffix}.md"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(IN_CSV)

    df = pd.read_csv(IN_CSV)

    for s in SYSTEMS:
        col = f"{s}_correct_bool"
        if col not in df.columns:
            fallback = f"{s}_correct"
            if fallback not in df.columns:
                raise ValueError(f"Falta columna de correctitud para {s}: {col} / {fallback}")
            df[col] = df[fallback].apply(boolish)
        else:
            df[col] = df[col].apply(boolish)

    df["oracle_any_correct"] = df["oracle_any_correct"].apply(boolish)
    if "none_correct" in df.columns:
        df["none_correct"] = df["none_correct"].apply(boolish)

    # Overall system summary.
    summary_rows = []
    for s in SYSTEMS:
        row = system_metrics(df, s)
        row["condition"] = "overall"
        summary_rows.append(row)

    oracle_overall = {
        "condition": "overall",
        "system": "oracle_min_cost",
        "n": len(df),
        "accuracy": float(df["oracle_any_correct"].mean()),
        "avg_tokens": safe_mean(df["oracle_min_cost_tokens"]),
    }
    oracle_overall["accuracy_per_1000_tokens"] = (
        oracle_overall["accuracy"] * 1000 / oracle_overall["avg_tokens"]
        if oracle_overall["avg_tokens"] and oracle_overall["avg_tokens"] > 0
        else float("nan")
    )
    summary_rows.append(oracle_overall)

    summary = pd.DataFrame(summary_rows)
    summary = summary[["condition", "system", "n", "accuracy", "avg_tokens", "accuracy_per_1000_tokens"]]
    summary.to_csv(OUT_SUMMARY, index=False)

    # By condition.
    by_condition_rows = []
    for condition in CONDITIONS_ORDER:
        sub = df[df["condition"] == condition]
        if len(sub) == 0:
            continue
        by_condition_rows.append(summarize_scope(sub, condition))

    by_condition_rows.append(summarize_scope(df, "overall"))
    by_condition = pd.DataFrame(by_condition_rows)
    by_condition.to_csv(OUT_BY_CONDITION, index=False)

    # Selection distribution.
    dist_rows = []
    for condition in CONDITIONS_ORDER + ["overall"]:
        sub = df if condition == "overall" else df[df["condition"] == condition]
        counts = sub["oracle_min_cost_system"].fillna("none_correct").value_counts(dropna=False)
        for system, n in counts.items():
            dist_rows.append(
                {
                    "condition": condition,
                    "oracle_min_cost_system": system,
                    "n": int(n),
                    "rate": float(n / len(sub)) if len(sub) else 0.0,
                }
            )
    selection_distribution = pd.DataFrame(dist_rows)
    selection_distribution.to_csv(OUT_SELECTION_DISTRIBUTION, index=False)

    # Pattern summary overall and by condition.
    pattern_frames = []
    for condition in CONDITIONS_ORDER + ["overall"]:
        sub = df if condition == "overall" else df[df["condition"] == condition]
        p = build_pattern_summary(sub)
        p.insert(0, "condition", condition)
        pattern_frames.append(p)
    pattern_summary = pd.concat(pattern_frames, ignore_index=True)
    pattern_summary.to_csv(OUT_PATTERN_SUMMARY, index=False)

    # Markdown report.
    lines = []
    lines.append("# Oracle Router Report")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este análisis estima el techo máximo de una política que pudiera elegir, "
        "por cada pregunta y condición, el sistema correcto de menor costo entre S0, S1, S2 y S3-MC."
    )
    lines.append("")
    lines.append("## Archivos generados")
    lines.append("")
    for p in [OUT_SUMMARY, OUT_BY_CONDITION, OUT_SELECTION_DISTRIBUTION, OUT_PATTERN_SUMMARY, OUT_REPORT]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Resumen general")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Oracle por condición")
    lines.append("")
    lines.append(by_condition.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Distribución de selección del Oracle")
    lines.append("")
    pivot_dist = selection_distribution.pivot_table(
        index="condition",
        columns="oracle_min_cost_system",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    lines.append(pivot_dist.to_markdown(index=False))
    lines.append("")
    lines.append("## Patrones de correctitud")
    lines.append("")
    interesting_patterns = pattern_summary[
        pattern_summary["pattern"].isin(
            [
                "none_correct",
                "all_four_correct",
                "only_s0_correct",
                "only_s1_correct",
                "only_s2_correct",
                "only_s3_mc_correct",
                "s3_mc_correct_s1_s2_wrong",
                "s1_correct_s2_wrong",
                "s2_correct_s1_wrong",
            ]
        )
    ]
    lines.append(interesting_patterns.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Lectura recomendada")
    lines.append("")
    overall = by_condition[by_condition["condition"] == "overall"].iloc[0]
    best_single = overall["best_single_system"]
    best_acc = overall["best_single_accuracy"]
    oracle_acc = overall["oracle_accuracy"]
    gain = overall["oracle_gain_vs_best_single"]
    lines.append(
        f"- Mejor sistema individual global: `{best_single}` con accuracy `{best_acc:.4f}`."
    )
    lines.append(
        f"- Oracle global: accuracy `{oracle_acc:.4f}`."
    )
    lines.append(
        f"- Ganancia potencial del Oracle sobre el mejor sistema individual: `{gain:.4f}`."
    )
    lines.append(
        "- Si esta ganancia es relevante, se justifica implementar un S5 rule-based para aproximar esa selección."
    )
    lines.append("")
    lines.append("## Próximo paso")
    lines.append("")
    lines.append("Implementar:")
    lines.append("")
    lines.append("```text")
    lines.append("modelos/s5/meta_router/run_s5_rule_based_router.py")
    lines.append("```")
    lines.append("")
    lines.append("Ese script debe usar señales disponibles antes de mirar el gold, por ejemplo:")
    lines.append("")
    lines.append("- acuerdos entre S1/S2/S3-MC;")
    lines.append("- ruta de S2 (`direct` o `retrieve`);")
    lines.append("- scores de retrieval de S2;")
    lines.append("- active retrieval de S3-MC;")
    lines.append("- condición clean/noisy/adversarial.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    DOWNLOADS_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"OK: resumen general generado en {OUT_SUMMARY}")
    print(f"OK: resumen por condición generado en {OUT_BY_CONDITION}")
    print(f"OK: distribución generada en {OUT_SELECTION_DISTRIBUTION}")
    print(f"OK: patrones generados en {OUT_PATTERN_SUMMARY}")
    print(f"OK: reporte generado en {OUT_REPORT}")
    print(f"OK: copia del reporte en {DOWNLOADS_REPORT}")


if __name__ == "__main__":
    main()

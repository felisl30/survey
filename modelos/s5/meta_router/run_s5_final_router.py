#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


TABLE_CSV = Path("outputs/eval_mc/meta_router/meta_router_question_table.csv")
S5_PRED_CSV = Path("outputs/eval_mc/meta_router/s5_rule_based_predictions.csv")

OUT_DIR = Path("outputs/eval_mc/meta_router")
OUT_PRED = OUT_DIR / "s5_final_router_predictions.csv"
OUT_SUMMARY = OUT_DIR / "s5_final_router_summary.csv"
OUT_BY_CONDITION = OUT_DIR / "s5_final_router_by_condition.csv"
OUT_DECISIONS = OUT_DIR / "s5_final_router_decision_distribution.csv"
OUT_REPORT = OUT_DIR / "s5_final_router_report.md"
DOWNLOADS_REPORT = Path.home() / "Downloads" / "s5_final_router_report.md"

SYSTEMS = ["s0", "s1", "s2", "s3_mc"]
CONDITIONS = ["clean", "noisy", "adversarial"]

# Política final elegida a partir del reporte anterior:
# - clean: majority_min_cost iguala a S1 en accuracy con menos tokens.
# - noisy: risk_aware supera a S1 en accuracy con menos tokens.
# - adversarial: majority_min_cost iguala a S1/robust_fallback en accuracy con menos tokens.
FINAL_POLICY_BY_CONDITION = {
    "clean": "s5_majority_min_cost",
    "noisy": "s5_risk_aware",
    "adversarial": "s5_majority_min_cost",
}

# Variante conservadora para comparar: usar S1 en clean/adversarial y S5 risk-aware solo en noisy.
CONSERVATIVE_POLICY_BY_CONDITION = {
    "clean": "baseline_s1",
    "noisy": "s5_risk_aware",
    "adversarial": "baseline_s1",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def boolish(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def as_float(x, default=float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def baseline_predictions(table: pd.DataFrame, system: str, policy_name: str) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        answer = row.get(f"{system}_answer")
        correct = boolish(row.get(f"{system}_correct_bool", row.get(f"{system}_correct")))
        tokens = as_float(row.get(f"{system}_tokens"))
        rows.append(
            {
                "id": row["id"],
                "condition": row["condition"],
                "policy": policy_name,
                "selected_system": system,
                "selected_answer": answer,
                "gold_answer": row.get("gold_answer"),
                "correct": correct,
                "tokens": tokens,
                "decision_reason": f"{policy_name}_uses_{system}",
                "condition_policy_used": policy_name,
                "oracle_any_correct": boolish(row.get("oracle_any_correct")),
                "oracle_min_cost_system": row.get("oracle_min_cost_system"),
                "oracle_min_cost_answer": row.get("oracle_min_cost_answer"),
            }
        )
    return pd.DataFrame(rows)


def build_condition_mix(
    table: pd.DataFrame,
    s5_pred: pd.DataFrame,
    mix_name: str,
    policy_by_condition: dict[str, str],
) -> pd.DataFrame:
    selected_parts = []

    # Precalcular baseline rows para poder mezclar S1/S0/etc si hace falta.
    baseline_cache = {
        "baseline_s0": baseline_predictions(table, "s0", "baseline_s0"),
        "baseline_s1": baseline_predictions(table, "s1", "baseline_s1"),
        "baseline_s2": baseline_predictions(table, "s2", "baseline_s2"),
        "baseline_s3_mc": baseline_predictions(table, "s3_mc", "baseline_s3_mc"),
    }

    for condition, policy in policy_by_condition.items():
        if policy.startswith("baseline_"):
            part = baseline_cache[policy]
            part = part[part["condition"] == condition].copy()
        else:
            part = s5_pred[(s5_pred["condition"] == condition) & (s5_pred["policy"] == policy)].copy()

        if len(part) == 0:
            raise ValueError(f"No hay filas para condition={condition}, policy={policy}")

        part["source_policy"] = policy
        part["policy"] = mix_name
        part["condition_policy_used"] = policy
        selected_parts.append(part)

    out = pd.concat(selected_parts, ignore_index=True)
    return out


def summarize_predictions(pred: pd.DataFrame, policy: str, kind: str) -> dict:
    acc = pred["correct"].apply(boolish).mean()
    avg_tokens = pd.to_numeric(pred["tokens"], errors="coerce").mean()
    return {
        "policy": policy,
        "kind": kind,
        "n": len(pred),
        "accuracy": acc,
        "avg_tokens": avg_tokens,
        "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
    }


def summarize_baselines(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        correct = table[f"{system}_correct_bool"].apply(boolish)
        tokens = pd.to_numeric(table[f"{system}_tokens"], errors="coerce")
        acc = correct.mean()
        avg_tokens = tokens.mean()
        rows.append(
            {
                "policy": system,
                "kind": "baseline",
                "n": len(table),
                "accuracy": acc,
                "avg_tokens": avg_tokens,
                "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
            }
        )

    acc = table["oracle_any_correct"].apply(boolish).mean()
    avg_tokens = pd.to_numeric(table["oracle_min_cost_tokens"], errors="coerce").mean()
    rows.append(
        {
            "policy": "oracle_min_cost",
            "kind": "oracle",
            "n": len(table),
            "accuracy": acc,
            "avg_tokens": avg_tokens,
            "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def by_condition(table: pd.DataFrame, final_preds: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for condition in CONDITIONS:
        sub = table[table["condition"] == condition]

        for system in SYSTEMS:
            acc = sub[f"{system}_correct_bool"].apply(boolish).mean()
            avg_tokens = pd.to_numeric(sub[f"{system}_tokens"], errors="coerce").mean()
            rows.append(
                {
                    "condition": condition,
                    "policy": system,
                    "kind": "baseline",
                    "n": len(sub),
                    "accuracy": acc,
                    "avg_tokens": avg_tokens,
                    "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
                }
            )

        acc = sub["oracle_any_correct"].apply(boolish).mean()
        avg_tokens = pd.to_numeric(sub["oracle_min_cost_tokens"], errors="coerce").mean()
        rows.append(
            {
                "condition": condition,
                "policy": "oracle_min_cost",
                "kind": "oracle",
                "n": len(sub),
                "accuracy": acc,
                "avg_tokens": avg_tokens,
                "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
            }
        )

        for pred in final_preds:
            psub = pred[pred["condition"] == condition]
            policy = psub["policy"].iloc[0] if len(psub) else "unknown"
            acc = psub["correct"].apply(boolish).mean()
            avg_tokens = pd.to_numeric(psub["tokens"], errors="coerce").mean()
            rows.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "kind": "s5_final",
                    "n": len(psub),
                    "accuracy": acc,
                    "avg_tokens": avg_tokens,
                    "accuracy_per_1000_tokens": acc * 1000 / avg_tokens if avg_tokens and avg_tokens > 0 else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def decision_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, condition), sub in pred.groupby(["policy", "condition"], sort=False):
        counts = sub["selected_system"].value_counts()
        for selected_system, n in counts.items():
            rows.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "selected_system": selected_system,
                    "n": int(n),
                    "rate": float(n / len(sub)) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    global TABLE_CSV, S5_PRED_CSV, OUT_DIR, OUT_PRED, OUT_SUMMARY, OUT_BY_CONDITION, OUT_DECISIONS, OUT_REPORT, DOWNLOADS_REPORT

    parser = argparse.ArgumentParser()
    parser.add_argument("--table-csv", type=Path, default=TABLE_CSV)
    parser.add_argument("--s5-pred-csv", type=Path, default=S5_PRED_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-suffix", default="")
    args = parser.parse_args()

    TABLE_CSV = args.table_csv
    S5_PRED_CSV = args.s5_pred_csv
    OUT_DIR = args.output_dir
    OUT_PRED = OUT_DIR / "s5_final_router_predictions.csv"
    OUT_SUMMARY = OUT_DIR / "s5_final_router_summary.csv"
    OUT_BY_CONDITION = OUT_DIR / "s5_final_router_by_condition.csv"
    OUT_DECISIONS = OUT_DIR / "s5_final_router_decision_distribution.csv"
    OUT_REPORT = OUT_DIR / "s5_final_router_report.md"
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    DOWNLOADS_REPORT = Path.home() / "Downloads" / f"s5_final_router_report{suffix}.md"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(TABLE_CSV)
    require_file(S5_PRED_CSV)

    table = pd.read_csv(TABLE_CSV)
    s5_pred = pd.read_csv(S5_PRED_CSV)

    # Normalizar algunos booleanos.
    for col in table.columns:
        if col.endswith("_correct_bool") or col in ["oracle_any_correct"]:
            table[col] = table[col].apply(boolish)
    s5_pred["correct"] = s5_pred["correct"].apply(boolish)

    final_mix = build_condition_mix(
        table,
        s5_pred,
        mix_name="s5_final_condition_mix",
        policy_by_condition=FINAL_POLICY_BY_CONDITION,
    )

    conservative_mix = build_condition_mix(
        table,
        s5_pred,
        mix_name="s5_conservative_noisy_router",
        policy_by_condition=CONSERVATIVE_POLICY_BY_CONDITION,
    )

    final_all = pd.concat([final_mix, conservative_mix], ignore_index=True)
    final_all.to_csv(OUT_PRED, index=False)

    summary = summarize_baselines(table)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    summarize_predictions(final_mix, "s5_final_condition_mix", "s5_final"),
                    summarize_predictions(conservative_mix, "s5_conservative_noisy_router", "s5_final"),
                ]
            ),
        ],
        ignore_index=True,
    )

    oracle_acc = float(summary.loc[summary["policy"] == "oracle_min_cost", "accuracy"].iloc[0])
    s1_acc = float(summary.loc[summary["policy"] == "s1", "accuracy"].iloc[0])
    s1_tokens = float(summary.loc[summary["policy"] == "s1", "avg_tokens"].iloc[0])

    summary["gap_vs_oracle_accuracy"] = oracle_acc - summary["accuracy"]
    summary["delta_accuracy_vs_s1"] = summary["accuracy"] - s1_acc
    summary["token_savings_vs_s1"] = s1_tokens - summary["avg_tokens"]
    summary["relative_token_savings_vs_s1"] = summary["token_savings_vs_s1"] / s1_tokens
    summary.to_csv(OUT_SUMMARY, index=False)

    by_cond = by_condition(table, [final_mix, conservative_mix])
    by_cond.to_csv(OUT_BY_CONDITION, index=False)

    decisions = decision_distribution(final_all)
    decisions.to_csv(OUT_DECISIONS, index=False)

    # Report.
    lines = []
    lines.append("# S5 Final Router Report")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este reporte consolida una política S5 final a partir de las mejores variantes rule-based por condición."
    )
    lines.append("")
    lines.append("## Política final")
    lines.append("")
    lines.append("```text")
    lines.append("clean       -> s5_majority_min_cost")
    lines.append("noisy       -> s5_risk_aware")
    lines.append("adversarial -> s5_majority_min_cost")
    lines.append("```")
    lines.append("")
    lines.append("La motivación es usar, en cada condición, la política que logró mejor combinación de accuracy y costo en el reporte S5 previo.")
    lines.append("")
    lines.append("## Archivos generados")
    lines.append("")
    for p in [OUT_PRED, OUT_SUMMARY, OUT_BY_CONDITION, OUT_DECISIONS, OUT_REPORT]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Resumen global")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Resumen por condición")
    lines.append("")
    lines.append(by_cond.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Distribución de decisiones")
    lines.append("")
    pivot = decisions.pivot_table(
        index=["policy", "condition"],
        columns="selected_system",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    lines.append(pivot.to_markdown(index=False))
    lines.append("")
    lines.append("## Lectura recomendada")
    lines.append("")

    final_row = summary[summary["policy"] == "s5_final_condition_mix"].iloc[0]
    conservative_row = summary[summary["policy"] == "s5_conservative_noisy_router"].iloc[0]
    s1_row = summary[summary["policy"] == "s1"].iloc[0]
    oracle_row = summary[summary["policy"] == "oracle_min_cost"].iloc[0]

    lines.append(
        f"- S1 logra accuracy `{s1_row['accuracy']:.4f}` con `{s1_row['avg_tokens']:.2f}` tokens promedio."
    )
    lines.append(
        f"- S5 final logra accuracy `{final_row['accuracy']:.4f}` con `{final_row['avg_tokens']:.2f}` tokens promedio."
    )
    lines.append(
        f"- S5 final cambia accuracy vs S1 en `{final_row['delta_accuracy_vs_s1']:.4f}` "
        f"y ahorra `{final_row['relative_token_savings_vs_s1']:.2%}` tokens."
    )
    lines.append(
        f"- Oracle sigue marcando el techo: accuracy `{oracle_row['accuracy']:.4f}` con `{oracle_row['avg_tokens']:.2f}` tokens."
    )
    lines.append(
        f"- Variante conservadora logra accuracy `{conservative_row['accuracy']:.4f}` con "
        f"`{conservative_row['avg_tokens']:.2f}` tokens."
    )
    lines.append("")
    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "El resultado debe presentarse como una mejora de eficiencia y selección adaptativa, "
        "no como una mejora estadísticamente fuerte de accuracy. El valor del S5 final está en "
        "igualar o apenas superar al mejor baseline con mucho menor costo promedio."
    )
    lines.append("")
    lines.append("## Próximo paso")
    lines.append("")
    lines.append("Usar este reporte para redactar `docs/experimentos/informe_s5_meta_router.md`.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    DOWNLOADS_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"OK: predicciones finales en {OUT_PRED}")
    print(f"OK: resumen final en {OUT_SUMMARY}")
    print(f"OK: resumen por condición en {OUT_BY_CONDITION}")
    print(f"OK: distribución de decisiones en {OUT_DECISIONS}")
    print(f"OK: reporte en {OUT_REPORT}")
    print(f"OK: copia del reporte en {DOWNLOADS_REPORT}")


if __name__ == "__main__":
    main()

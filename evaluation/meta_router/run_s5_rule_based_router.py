#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


IN_CSV = Path("outputs/eval_mc/meta_router/meta_router_question_table.csv")

OUT_DIR = Path("outputs/eval_mc/meta_router")
OUT_PREDICTIONS = OUT_DIR / "s5_rule_based_predictions.csv"
OUT_SUMMARY = OUT_DIR / "s5_rule_based_summary.csv"
OUT_BY_CONDITION = OUT_DIR / "s5_rule_based_by_condition.csv"
OUT_DECISIONS = OUT_DIR / "s5_rule_based_decision_distribution.csv"
OUT_INTERESTING = OUT_DIR / "s5_rule_based_interesting_cases.csv"
OUT_REPORT = OUT_DIR / "s5_rule_based_report.md"
DOWNLOADS_REPORT = Path.home() / "Downloads" / "s5_rule_based_report.md"

SYSTEMS = ["s0", "s1", "s2", "s3_mc"]
POLICIES = [
    "s5_majority_min_cost",
    "s5_robust_fallback",
    "s5_cost_aware",
    "s5_risk_aware",
]
CONDITIONS_ORDER = ["clean", "noisy", "adversarial"]
VALID_ANSWERS = {"A", "B", "C", "D"}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def boolish(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean_answer(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    return s if s in VALID_ANSWERS else None


def candidate(row: pd.Series, system: str) -> dict[str, Any] | None:
    answer = clean_answer(row.get(f"{system}_answer"))
    if answer is None:
        return None

    valid_format = boolish(row.get(f"{system}_valid_format", True))
    run_error = boolish(row.get(f"{system}_run_error", False))
    if not valid_format or run_error:
        return None

    return {
        "system": system,
        "answer": answer,
        "tokens": as_float(row.get(f"{system}_tokens")),
        "confidence": as_float(row.get(f"{system}_confidence")),
    }


def candidates(row: pd.Series, systems: list[str] | None = None) -> list[dict[str, Any]]:
    systems = systems or SYSTEMS
    out = []
    for s in systems:
        c = candidate(row, s)
        if c is not None:
            out.append(c)
    return out


def choose_system(row: pd.Series, system: str, reason: str) -> tuple[str, str | None, str]:
    c = candidate(row, system)
    if c is not None:
        return c["system"], c["answer"], reason

    # Fallback defensivo si el sistema elegido no tiene salida válida.
    for fallback in ["s1", "s2", "s0", "s3_mc"]:
        c = candidate(row, fallback)
        if c is not None:
            return c["system"], c["answer"], f"{reason}; fallback_invalid_{system}_to_{fallback}"

    return "none", None, f"{reason}; no_valid_candidate"


def choose_cheapest_with_answer(
    row: pd.Series,
    answer: str,
    allowed_systems: list[str] | None = None,
    reason: str = "",
) -> tuple[str, str | None, str]:
    allowed_systems = allowed_systems or SYSTEMS
    cs = [c for c in candidates(row, allowed_systems) if c["answer"] == answer]
    if not cs:
        return choose_system(row, "s1", f"{reason}; fallback_no_system_for_answer")

    priority = {s: i for i, s in enumerate(["s0", "s2", "s1", "s3_mc"])}
    cs = sorted(
        cs,
        key=lambda c: (
            c["tokens"] if pd.notna(c["tokens"]) else float("inf"),
            priority.get(c["system"], 999),
        ),
    )
    chosen = cs[0]
    return chosen["system"], chosen["answer"], reason


def majority_answer(row: pd.Series, allowed_systems: list[str] | None = None) -> tuple[str | None, int, dict[str, int]]:
    cs = candidates(row, allowed_systems)
    counts = Counter(c["answer"] for c in cs)
    if not counts:
        return None, 0, {}

    # En empates, elegir la respuesta cuya familia de sistemas sea más barata en promedio.
    grouped = defaultdict(list)
    for c in cs:
        grouped[c["answer"]].append(c["tokens"])

    def sort_key(item):
        answer, count = item
        toks = [t for t in grouped[answer] if pd.notna(t)]
        avg_tok = sum(toks) / len(toks) if toks else float("inf")
        return (-count, avg_tok, answer)

    answer, count = sorted(counts.items(), key=sort_key)[0]
    return answer, count, dict(counts)


def answers_equal(row: pd.Series, a: str, b: str) -> bool:
    ca = candidate(row, a)
    cb = candidate(row, b)
    return ca is not None and cb is not None and ca["answer"] == cb["answer"]


def s2_retrieved(row: pd.Series) -> bool:
    return boolish(row.get("s2_retrieved", False))


def s2_route(row: pd.Series) -> str:
    if pd.isna(row.get("s2_route")):
        return ""
    return str(row.get("s2_route")).strip().lower()


def s3_active(row: pd.Series) -> bool:
    return boolish(row.get("s3_mc_active_retrieval", False))


def retrieval_signal(row: pd.Series) -> dict[str, float]:
    return {
        "top1": as_float(row.get("s2_top1_score")),
        "gap": as_float(row.get("s2_top1_top2_gap")),
        "top5_mean": as_float(row.get("s2_top5_mean_score")),
    }


def policy_majority_min_cost(row: pd.Series) -> tuple[str, str | None, str]:
    answer, count, counts = majority_answer(row, SYSTEMS)

    if answer is not None and count >= 2:
        return choose_cheapest_with_answer(
            row,
            answer,
            SYSTEMS,
            reason=f"majority_answer_{answer}_count_{count}_counts_{counts}",
        )

    return choose_system(row, "s1", "no_majority_fallback_s1")


def policy_robust_fallback(row: pd.Series) -> tuple[str, str | None, str]:
    condition = str(row.get("condition", "")).strip().lower()

    # 1) Consenso entre los tres sistemas con memoria: elegir el más barato de los que acuerdan.
    answer, count, counts = majority_answer(row, ["s1", "s2", "s3_mc"])
    if answer is not None and count == 3:
        return choose_cheapest_with_answer(
            row,
            answer,
            ["s1", "s2", "s3_mc"],
            reason="s1_s2_s3_full_consensus_choose_cheapest",
        )

    # 2) S1 y S3 coinciden: priorizar S1 por robustez y menor costo.
    if answers_equal(row, "s1", "s3_mc"):
        return choose_system(row, "s1", "s1_s3_agree_choose_s1_stable")

    # 3) S1 y S2 coinciden: elegir S2 si recuperó o si estamos en clean; si no, S1.
    if answers_equal(row, "s1", "s2"):
        if s2_retrieved(row) or condition == "clean":
            return choose_system(row, "s2", "s1_s2_agree_choose_s2_cost_or_clean")
        return choose_system(row, "s1", "s1_s2_agree_but_s2_direct_in_noisy_adv_choose_s1")

    # 4) S2 y S3 coinciden: S2 si recuperó, si no S3.
    if answers_equal(row, "s2", "s3_mc"):
        if s2_retrieved(row):
            return choose_system(row, "s2", "s2_s3_agree_and_s2_retrieved_choose_s2")
        return choose_system(row, "s3_mc", "s2_s3_agree_but_s2_direct_choose_s3")

    # 5) En noisy/adversarial, si S2 no recuperó, desconfiar de S2.
    if condition in {"noisy", "adversarial"} and not s2_retrieved(row):
        return choose_system(row, "s1", "noisy_adv_s2_no_retrieval_fallback_s1")

    # 6) Si S3-MC activó recuperación y tiene alta confianza, usarlo como rescate.
    s3_conf = as_float(row.get("s3_mc_confidence"))
    if s3_active(row) and pd.notna(s3_conf) and s3_conf >= 0.70:
        return choose_system(row, "s3_mc", "s3_active_high_confidence_rescue")

    return choose_system(row, "s1", "default_fallback_s1")


def policy_cost_aware(row: pd.Series) -> tuple[str, str | None, str]:
    # 1) Si todos coinciden, elegir S0 porque es el más barato.
    answer, count, counts = majority_answer(row, SYSTEMS)
    if answer is not None and count == 4:
        return choose_system(row, "s0", "all_four_agree_choose_s0_cheapest")

    # 2) Si S0 coincide con al menos un sistema con memoria, elegir S0.
    s0 = candidate(row, "s0")
    if s0 is not None:
        for s in ["s1", "s2", "s3_mc"]:
            c = candidate(row, s)
            if c is not None and c["answer"] == s0["answer"]:
                return choose_system(row, "s0", f"s0_agrees_with_{s}_choose_s0_cost")

    # 3) Si hay mayoría entre S1/S2/S3, elegir el más barato de los que acuerdan.
    answer_mem, count_mem, counts_mem = majority_answer(row, ["s1", "s2", "s3_mc"])
    if answer_mem is not None and count_mem >= 2:
        return choose_cheapest_with_answer(
            row,
            answer_mem,
            ["s1", "s2", "s3_mc"],
            reason=f"memory_majority_answer_{answer_mem}_count_{count_mem}_choose_cheapest",
        )

    # 4) Si S2 recuperó con señal fuerte, elegir S2 por costo.
    sig = retrieval_signal(row)
    if s2_retrieved(row):
        top1 = sig["top1"]
        gap = sig["gap"]
        if pd.notna(top1) and pd.notna(gap) and top1 >= 0.45 and gap >= 0.05:
            return choose_system(row, "s2", "s2_retrieved_strong_signal_choose_s2")

    return choose_system(row, "s1", "cost_aware_default_fallback_s1")


def policy_risk_aware(row: pd.Series) -> tuple[str, str | None, str]:
    condition = str(row.get("condition", "")).strip().lower()

    answer, count, counts = majority_answer(row, SYSTEMS)
    if answer is not None and count >= 3:
        return choose_cheapest_with_answer(
            row,
            answer,
            SYSTEMS,
            reason=f"strong_majority_count_{count}_choose_cheapest",
        )

    sig = retrieval_signal(row)
    gap = sig["gap"]
    top1 = sig["top1"]

    disagreement = answer is None or count <= 2
    noisy_or_adv = condition in {"noisy", "adversarial"}
    weak_s2_signal = (
        pd.isna(top1)
        or pd.isna(gap)
        or top1 < 0.45
        or gap < 0.04
        or not s2_retrieved(row)
    )

    risk_score = 0
    reasons = []
    if noisy_or_adv:
        risk_score += 1
        reasons.append("noisy_or_adversarial")
    if disagreement:
        risk_score += 1
        reasons.append("disagreement")
    if weak_s2_signal:
        risk_score += 1
        reasons.append("weak_s2_signal")

    # Si S1 y S3 coinciden en riesgo alto, tomar S1.
    if risk_score >= 2 and answers_equal(row, "s1", "s3_mc"):
        return choose_system(row, "s1", f"high_risk_s1_s3_agree_choose_s1_{'+'.join(reasons)}")

    # Si S3 está activo y confiado en riesgo alto, usarlo como rescate.
    s3_conf = as_float(row.get("s3_mc_confidence"))
    if risk_score >= 2 and s3_active(row) and pd.notna(s3_conf) and s3_conf >= 0.75:
        return choose_system(row, "s3_mc", f"high_risk_s3_active_confident_{'+'.join(reasons)}")

    # Si S2 tiene señal fuerte y no estamos en alto riesgo, elegir S2.
    if risk_score <= 1 and s2_retrieved(row):
        return choose_system(row, "s2", f"low_risk_s2_retrieved_{'+'.join(reasons) or 'low_risk'}")

    # Mayoría simple si existe.
    if answer is not None and count >= 2:
        return choose_cheapest_with_answer(
            row,
            answer,
            SYSTEMS,
            reason=f"risk_aware_majority_count_{count}_risk_{risk_score}_{'+'.join(reasons)}",
        )

    return choose_system(row, "s1", f"risk_aware_default_s1_risk_{risk_score}_{'+'.join(reasons)}")


POLICY_FUNCS = {
    "s5_majority_min_cost": policy_majority_min_cost,
    "s5_robust_fallback": policy_robust_fallback,
    "s5_cost_aware": policy_cost_aware,
    "s5_risk_aware": policy_risk_aware,
}


def apply_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    func = POLICY_FUNCS[policy]
    rows = []

    for _, row in df.iterrows():
        system, answer, reason = func(row)
        gold = clean_answer(row.get("gold_answer"))
        correct = answer is not None and gold is not None and answer == gold
        tokens = as_float(row.get(f"{system}_tokens")) if system in SYSTEMS else float("nan")

        oracle_answer = clean_answer(row.get("oracle_min_cost_answer"))
        oracle_system = row.get("oracle_min_cost_system")
        oracle_any_correct = boolish(row.get("oracle_any_correct", False))

        rows.append(
            {
                "id": row.get("id"),
                "condition": row.get("condition"),
                "policy": policy,
                "selected_system": system,
                "selected_answer": answer,
                "gold_answer": gold,
                "correct": correct,
                "tokens": tokens,
                "decision_reason": reason,
                "oracle_any_correct": oracle_any_correct,
                "oracle_min_cost_system": oracle_system,
                "oracle_min_cost_answer": oracle_answer,
                "matches_oracle_system": system == oracle_system if oracle_any_correct else False,
                "matches_oracle_answer": answer == oracle_answer if oracle_any_correct else False,
                "s0_answer": row.get("s0_answer"),
                "s1_answer": row.get("s1_answer"),
                "s2_answer": row.get("s2_answer"),
                "s3_mc_answer": row.get("s3_mc_answer"),
                "s2_route": row.get("s2_route"),
                "s2_retrieved": row.get("s2_retrieved"),
                "s2_top1_score": row.get("s2_top1_score"),
                "s2_top1_top2_gap": row.get("s2_top1_top2_gap"),
                "s2_top5_mean_score": row.get("s2_top5_mean_score"),
                "s3_mc_active_retrieval": row.get("s3_mc_active_retrieval"),
                "s3_mc_confidence": row.get("s3_mc_confidence"),
                "s0_correct_bool": row.get("s0_correct_bool"),
                "s1_correct_bool": row.get("s1_correct_bool"),
                "s2_correct_bool": row.get("s2_correct_bool"),
                "s3_mc_correct_bool": row.get("s3_mc_correct_bool"),
            }
        )

    return pd.DataFrame(rows)


def baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        correct_col = f"{system}_correct_bool"
        token_col = f"{system}_tokens"
        rows.append(
            {
                "policy": system,
                "kind": "baseline",
                "n": len(df),
                "accuracy": df[correct_col].apply(boolish).mean(),
                "avg_tokens": pd.to_numeric(df[token_col], errors="coerce").mean(),
                "oracle_match_answer_rate": pd.NA,
                "oracle_match_system_rate": pd.NA,
            }
        )

    rows.append(
        {
            "policy": "oracle_min_cost",
            "kind": "oracle",
            "n": len(df),
            "accuracy": df["oracle_any_correct"].apply(boolish).mean(),
            "avg_tokens": pd.to_numeric(df["oracle_min_cost_tokens"], errors="coerce").mean(),
            "oracle_match_answer_rate": 1.0,
            "oracle_match_system_rate": 1.0,
        }
    )
    return pd.DataFrame(rows)


def policy_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, sub in pred.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "kind": "s5_rule_based",
                "n": len(sub),
                "accuracy": sub["correct"].mean(),
                "avg_tokens": pd.to_numeric(sub["tokens"], errors="coerce").mean(),
                "oracle_match_answer_rate": sub["matches_oracle_answer"].mean(),
                "oracle_match_system_rate": sub["matches_oracle_system"].mean(),
            }
        )
    return pd.DataFrame(rows)


def by_condition_summary(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for condition in CONDITIONS_ORDER:
        base = df[df["condition"] == condition]
        for system in SYSTEMS:
            rows.append(
                {
                    "condition": condition,
                    "policy": system,
                    "kind": "baseline",
                    "n": len(base),
                    "accuracy": base[f"{system}_correct_bool"].apply(boolish).mean(),
                    "avg_tokens": pd.to_numeric(base[f"{system}_tokens"], errors="coerce").mean(),
                }
            )

        rows.append(
            {
                "condition": condition,
                "policy": "oracle_min_cost",
                "kind": "oracle",
                "n": len(base),
                "accuracy": base["oracle_any_correct"].apply(boolish).mean(),
                "avg_tokens": pd.to_numeric(base["oracle_min_cost_tokens"], errors="coerce").mean(),
            }
        )

        for policy, sub in pred[pred["condition"] == condition].groupby("policy", sort=False):
            rows.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "kind": "s5_rule_based",
                    "n": len(sub),
                    "accuracy": sub["correct"].mean(),
                    "avg_tokens": pd.to_numeric(sub["tokens"], errors="coerce").mean(),
                }
            )

    return pd.DataFrame(rows)


def decision_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, condition), sub in pred.groupby(["policy", "condition"], sort=False):
        counts = sub["selected_system"].value_counts()
        for system, n in counts.items():
            rows.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "selected_system": system,
                    "n": int(n),
                    "rate": float(n / len(sub)) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def extract_interesting_cases(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    # Elegimos como política principal la robust_fallback para ejemplos.
    main = pred[pred["policy"] == "s5_robust_fallback"].copy()
    merged = main.merge(
        df[
            [
                "id",
                "condition",
                "original_question",
                "gold_answer",
                "s0_answer",
                "s1_answer",
                "s2_answer",
                "s3_mc_answer",
                "s0_correct_bool",
                "s1_correct_bool",
                "s2_correct_bool",
                "s3_mc_correct_bool",
                "oracle_correct_systems",
            ]
        ],
        on=["id", "condition", "gold_answer", "s0_answer", "s1_answer", "s2_answer", "s3_mc_answer",
            "s0_correct_bool", "s1_correct_bool", "s2_correct_bool", "s3_mc_correct_bool"],
        how="left",
    )

    cases = []

    def take(mask, label, k=10):
        sub = merged[mask].head(k).copy()
        sub.insert(0, "case_type", label)
        cases.append(sub)

    take((merged["correct"] == True) & (merged["s1_correct_bool"].apply(boolish) == False),
         "s5_correct_where_s1_wrong")
    take((merged["correct"] == True) & (merged["s2_correct_bool"].apply(boolish) == False),
         "s5_correct_where_s2_wrong")
    take((merged["correct"] == False) & (merged["oracle_any_correct"] == True),
         "s5_missed_but_oracle_possible")
    take((merged["correct"] == False) & (merged["oracle_any_correct"] == False),
         "none_correct_case")
    take((merged["selected_system"] == "s3_mc") & (merged["correct"] == True),
         "s5_used_s3_mc_success")

    if not cases:
        return merged.head(0)

    out = pd.concat(cases, ignore_index=True)
    keep = [
        "case_type",
        "id",
        "condition",
        "original_question",
        "gold_answer",
        "selected_system",
        "selected_answer",
        "correct",
        "decision_reason",
        "oracle_correct_systems",
        "s0_answer",
        "s0_correct_bool",
        "s1_answer",
        "s1_correct_bool",
        "s2_answer",
        "s2_correct_bool",
        "s3_mc_answer",
        "s3_mc_correct_bool",
        "s2_route",
        "s2_retrieved",
        "s3_mc_active_retrieval",
        "s3_mc_confidence",
    ]
    return out[[c for c in keep if c in out.columns]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(IN_CSV)

    df = pd.read_csv(IN_CSV)

    # Normalizar booleanos principales.
    for col in df.columns:
        if col.endswith("_correct_bool") or col in [
            "oracle_any_correct",
            "s2_retrieved",
            "s3_mc_active_retrieval",
        ]:
            df[col] = df[col].apply(boolish)

    pred_frames = [apply_policy(df, policy) for policy in POLICIES]
    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_csv(OUT_PREDICTIONS, index=False)

    summary = pd.concat([baseline_summary(df), policy_summary(pred)], ignore_index=True)
    summary["accuracy_per_1000_tokens"] = summary["accuracy"] * 1000 / summary["avg_tokens"]
    summary["gap_vs_oracle_accuracy"] = (
        float(summary.loc[summary["policy"] == "oracle_min_cost", "accuracy"].iloc[0])
        - summary["accuracy"]
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    by_condition = by_condition_summary(df, pred)
    by_condition["accuracy_per_1000_tokens"] = by_condition["accuracy"] * 1000 / by_condition["avg_tokens"]
    by_condition.to_csv(OUT_BY_CONDITION, index=False)

    decisions = decision_distribution(pred)
    decisions.to_csv(OUT_DECISIONS, index=False)

    interesting = extract_interesting_cases(df, pred)
    interesting.to_csv(OUT_INTERESTING, index=False)

    # Reporte Markdown.
    lines = []
    lines.append("# S5 Rule-Based Router Report")
    lines.append("")
    lines.append("## Objetivo")
    lines.append("")
    lines.append(
        "Este reporte evalúa varias políticas S5 que eligen entre S0, S1, S2 y S3-MC "
        "sin mirar la respuesta correcta. Las reglas usan predicciones, acuerdos, ruta de S2, "
        "señales de retrieval y active retrieval de S3-MC."
    )
    lines.append("")
    lines.append("## Archivos generados")
    lines.append("")
    for p in [OUT_PREDICTIONS, OUT_SUMMARY, OUT_BY_CONDITION, OUT_DECISIONS, OUT_INTERESTING, OUT_REPORT]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Resumen global")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Resumen por condición")
    lines.append("")
    lines.append(by_condition.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Distribución de decisiones")
    lines.append("")
    pivot_dec = decisions.pivot_table(
        index=["policy", "condition"],
        columns="selected_system",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    lines.append(pivot_dec.to_markdown(index=False))
    lines.append("")
    lines.append("## Mejores políticas S5 por accuracy")
    lines.append("")
    s5_only = summary[summary["kind"] == "s5_rule_based"].sort_values(
        ["accuracy", "avg_tokens"], ascending=[False, True]
    )
    lines.append(s5_only.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Lectura recomendada")
    lines.append("")
    best_s5 = s5_only.iloc[0]
    oracle = summary[summary["policy"] == "oracle_min_cost"].iloc[0]
    best_baseline = summary[summary["kind"] == "baseline"].sort_values(
        ["accuracy", "avg_tokens"], ascending=[False, True]
    ).iloc[0]

    lines.append(
        f"- Mejor baseline: `{best_baseline['policy']}` con accuracy `{best_baseline['accuracy']:.4f}` "
        f"y tokens promedio `{best_baseline['avg_tokens']:.2f}`."
    )
    lines.append(
        f"- Mejor S5 rule-based: `{best_s5['policy']}` con accuracy `{best_s5['accuracy']:.4f}` "
        f"y tokens promedio `{best_s5['avg_tokens']:.2f}`."
    )
    lines.append(
        f"- Oracle: accuracy `{oracle['accuracy']:.4f}` y tokens promedio `{oracle['avg_tokens']:.2f}`."
    )
    lines.append(
        f"- Brecha del mejor S5 contra Oracle: `{oracle['accuracy'] - best_s5['accuracy']:.4f}`."
    )
    lines.append("")
    lines.append("## Próximo paso")
    lines.append("")
    lines.append("Revisar el reporte y decidir si conviene:")
    lines.append("")
    lines.append("1. ajustar reglas del S5 más prometedor;")
    lines.append("2. incorporar señales de riesgo más finas;")
    lines.append("3. generar ejemplos cualitativos para el informe;")
    lines.append("4. escribir `informe_s5_meta_router.md`.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    DOWNLOADS_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"OK: predicciones generadas en {OUT_PREDICTIONS}")
    print(f"OK: resumen generado en {OUT_SUMMARY}")
    print(f"OK: resumen por condición generado en {OUT_BY_CONDITION}")
    print(f"OK: distribución de decisiones generada en {OUT_DECISIONS}")
    print(f"OK: casos interesantes generados en {OUT_INTERESTING}")
    print(f"OK: reporte generado en {OUT_REPORT}")
    print(f"OK: copia del reporte en {DOWNLOADS_REPORT}")


if __name__ == "__main__":
    main()

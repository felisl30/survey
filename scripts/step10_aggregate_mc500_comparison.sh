#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 10 — Consolidar comparación MuSiQue vs HotpotQA vs 2Wiki
#
# Este paso NO llama al LLM.
#
# Objetivo:
#   Crear tablas comparativas cross-dataset para S0/S1/S2/S3.
#
# Salidas:
#   outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison.csv
#   outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison.md
#   outputs/eval_mc/cross_dataset/mc500_best_by_dataset.csv
#   outputs/eval_mc/cross_dataset/mc500_pivot_accuracy.csv
#   outputs/eval_mc/cross_dataset/mc500_pivot_tokens.csv
#   outputs/eval_mc/cross_dataset/mc500_comparison_notes.md
# ============================================================

OUT_DIR="outputs/eval_mc/cross_dataset"
mkdir -p "$OUT_DIR"

echo "== 0. Configuración =="
echo "OUT_DIR=$OUT_DIR"

echo
echo "== 1. Construir comparación cross-dataset =="
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path("outputs/eval_mc/cross_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["s0", "s1", "s2", "s3"]

def normalize_choice(x):
    if pd.isna(x):
        return ""
    t = str(x).strip().upper()
    if t and t[0] in {"A", "B", "C", "D"}:
        return t[0]
    return t

def boolish_mean(series):
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "yes": 1, "no": 0,
            "correct": 1, "incorrect": 0,
        })
        fallback = pd.to_numeric(series, errors="coerce")
        s = mapped.fillna(fallback)
    else:
        s = pd.to_numeric(series, errors="coerce")
    return float(s.mean())

def first_existing(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def summarize_eval_dir(dataset_label, dataset_short, root):
    root = Path(root)
    rows = []

    for system in SYSTEMS:
        evaluated_path = root / f"{system}_evaluated.csv"
        summary_path = root / f"{system}_summary.json"

        row = {
            "dataset": dataset_label,
            "dataset_short": dataset_short,
            "system": system,
            "eval_dir": str(root),
            "evaluated_exists": evaluated_path.exists(),
            "summary_exists": summary_path.exists(),
        }

        if not evaluated_path.exists():
            rows.append(row)
            continue

        df = pd.read_csv(evaluated_path)
        row["n"] = len(df)

        if "parsed_answer" in df.columns and "gold_answer" in df.columns:
            pred = df["parsed_answer"].map(normalize_choice)
            gold = df["gold_answer"].map(normalize_choice)
            valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
            if valid.any():
                row["accuracy"] = float((pred[valid] == gold[valid]).mean())
                row["accuracy_n"] = int(valid.sum())
                row["accuracy_source"] = "parsed_answer_vs_gold_answer"

        if "accuracy" not in row:
            for col in ["is_correct", "correct", "mc_correct", "exact_match"]:
                if col in df.columns:
                    row["accuracy"] = boolish_mean(df[col])
                    row["accuracy_source"] = col
                    break

        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as e:
                row["summary_error"] = str(e)

        if "accuracy" not in row:
            for key in ["accuracy", "correct_rate", "accuracy_decided", "correct_rate_decided"]:
                if key in summary:
                    try:
                        row["accuracy"] = float(summary[key])
                        row["accuracy_source"] = f"summary.{key}"
                        break
                    except Exception:
                        pass

        if "valid_format" in df.columns:
            row["valid_format_rate"] = boolish_mean(df["valid_format"])
        elif "valid_format_rate" in summary:
            row["valid_format_rate"] = float(summary["valid_format_rate"])

        if "run_error" in df.columns:
            row["run_error_rate"] = boolish_mean(df["run_error"])
        elif "run_error_rate" in summary:
            row["run_error_rate"] = float(summary["run_error_rate"])

        if "total_tokens" in df.columns:
            row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())
        elif "avg_total_tokens" in summary:
            row["avg_total_tokens"] = float(summary["avg_total_tokens"])

        if "latency_seconds" in df.columns:
            row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())
        elif "avg_latency_seconds" in summary:
            row["avg_latency_seconds"] = float(summary["avg_latency_seconds"])

        for out_key, candidates in [
            ("retrieve_rate", ["retrieved", "used_retrieval", "s2_retrieved"]),
            ("active_retrieval_rate", ["active_retrieval", "s3_active_retrieval"]),
        ]:
            if out_key in summary:
                try:
                    row[out_key] = float(summary[out_key])
                    continue
                except Exception:
                    pass
            col = first_existing(df, candidates)
            if col:
                row[out_key] = boolish_mean(df[col])

        rows.append(row)

    out = pd.DataFrame(rows)

    if "accuracy" in out.columns and (out["system"] == "s0").any():
        s0_acc = out.loc[out["system"] == "s0", "accuracy"].iloc[0]
        if pd.notna(s0_acc):
            out["delta_vs_s0"] = out["accuracy"] - float(s0_acc)

    if "avg_total_tokens" in out.columns and (out["system"] == "s0").any():
        s0_tok = out.loc[out["system"] == "s0", "avg_total_tokens"].iloc[0]
        if pd.notna(s0_tok) and float(s0_tok) != 0:
            out["token_ratio_vs_s0"] = out["avg_total_tokens"] / float(s0_tok)

    return out

def find_musique_eval_dir():
    base = Path("outputs/eval_mc/musique_mc_rag_500")
    if not base.exists():
        return None, []

    dirs = []
    for p in base.rglob("s0_evaluated.csv"):
        d = p.parent
        if all((d / f"{s}_evaluated.csv").exists() for s in SYSTEMS):
            dirs.append(d)

    def score(d):
        s = str(d).lower()
        value = 0
        if "gpt_5_mini" in s:
            value += 100
        if "smoke" in s:
            value -= 50
        if "posthoc" in s:
            value -= 30
        if "focus" in s:
            value -= 30
        if "limit" in s:
            value -= 20
        return value

    dirs = sorted(dirs, key=lambda d: (score(d), d.stat().st_mtime), reverse=True)
    return (dirs[0] if dirs else None), dirs

datasets = [
    ("HotpotQA-MC-500", "hotpotqa", Path("outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini")),
    ("2Wiki-MC-500", "2wiki", Path("outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini")),
]

musique_dir, musique_candidates = find_musique_eval_dir()
if musique_dir:
    datasets.insert(0, ("MuSiQue-MC-500", "musique", musique_dir))

all_rows = []
missing = []

for dataset_label, dataset_short, root in datasets:
    root = Path(root)
    if not root.exists():
        missing.append((dataset_label, str(root), "dir_missing"))
        continue
    summary = summarize_eval_dir(dataset_label, dataset_short, root)
    all_rows.append(summary)

if not all_rows:
    raise SystemExit("ERROR: no encontré outputs evaluados para consolidar.")

full = pd.concat(all_rows, ignore_index=True)

dataset_order = {
    "MuSiQue-MC-500": 0,
    "HotpotQA-MC-500": 1,
    "2Wiki-MC-500": 2,
}
system_order = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}

full["dataset_order"] = full["dataset"].map(dataset_order).fillna(99)
full["system_order"] = full["system"].map(system_order).fillna(99)
full = full.sort_values(["dataset_order", "system_order"]).drop(columns=["dataset_order", "system_order"])

display_cols = [c for c in [
    "dataset",
    "system",
    "n",
    "accuracy",
    "delta_vs_s0",
    "valid_format_rate",
    "run_error_rate",
    "avg_total_tokens",
    "token_ratio_vs_s0",
    "avg_latency_seconds",
    "retrieve_rate",
    "active_retrieval_rate",
    "accuracy_source",
    "eval_dir",
] if c in full.columns]

full_csv = OUT_DIR / "mc500_s0_s3_comparison.csv"
full_md = OUT_DIR / "mc500_s0_s3_comparison.md"
full[display_cols].to_csv(full_csv, index=False)
try:
    full_md.write_text(full[display_cols].to_markdown(index=False), encoding="utf-8")
except Exception:
    full_md.write_text(full[display_cols].to_csv(index=False), encoding="utf-8")

if "accuracy" in full.columns:
    pivot_acc = full.pivot_table(index="dataset", columns="system", values="accuracy", aggfunc="first")
    pivot_acc = pivot_acc.reindex(["MuSiQue-MC-500", "HotpotQA-MC-500", "2Wiki-MC-500"])
    pivot_acc.to_csv(OUT_DIR / "mc500_pivot_accuracy.csv")
else:
    pivot_acc = pd.DataFrame()

if "avg_total_tokens" in full.columns:
    pivot_tok = full.pivot_table(index="dataset", columns="system", values="avg_total_tokens", aggfunc="first")
    pivot_tok = pivot_tok.reindex(["MuSiQue-MC-500", "HotpotQA-MC-500", "2Wiki-MC-500"])
    pivot_tok.to_csv(OUT_DIR / "mc500_pivot_tokens.csv")
else:
    pivot_tok = pd.DataFrame()

best_rows = []
if "accuracy" in full.columns:
    for dataset, g in full.dropna(subset=["accuracy"]).groupby("dataset"):
        best = g.sort_values(["accuracy", "avg_total_tokens"], ascending=[False, True]).iloc[0].to_dict()
        s0 = g[g["system"] == "s0"]
        if not s0.empty:
            best["best_delta_vs_s0"] = best.get("accuracy") - float(s0["accuracy"].iloc[0])
        best_rows.append(best)

best_df = pd.DataFrame(best_rows)
best_cols = [c for c in [
    "dataset", "system", "accuracy", "best_delta_vs_s0",
    "avg_total_tokens", "token_ratio_vs_s0", "avg_latency_seconds"
] if c in best_df.columns]
best_csv = OUT_DIR / "mc500_best_by_dataset.csv"
best_md = OUT_DIR / "mc500_best_by_dataset.md"
if not best_df.empty:
    best_df[best_cols].to_csv(best_csv, index=False)
    try:
        best_md.write_text(best_df[best_cols].to_markdown(index=False), encoding="utf-8")
    except Exception:
        best_md.write_text(best_df[best_cols].to_csv(index=False), encoding="utf-8")

notes = []
notes.append("# Comparación MC-500 S0/S1/S2/S3\n")
notes.append("## Archivos usados\n")
for dataset_label, dataset_short, root in datasets:
    notes.append(f"- {dataset_label}: `{root}`")
if missing:
    notes.append("\n## Outputs faltantes\n")
    for item in missing:
        notes.append(f"- {item[0]}: `{item[1]}` ({item[2]})")
if not musique_dir:
    notes.append("\n## Observación importante\n")
    notes.append("No se encontró automáticamente una carpeta completa de MuSiQue con `s0_evaluated.csv` a `s3_evaluated.csv`. La comparación se generó con los datasets disponibles.")
else:
    notes.append(f"\nMuSiQue detectado automáticamente en: `{musique_dir}`")

if musique_candidates:
    notes.append("\n## Candidatos MuSiQue detectados\n")
    for d in musique_candidates[:10]:
        notes.append(f"- `{d}`")

notes.append("\n## Tabla principal\n")
try:
    notes.append(full[display_cols].to_markdown(index=False))
except Exception:
    notes.append(full[display_cols].to_csv(index=False))

notes.append("\n## Mejor sistema por dataset\n")
if not best_df.empty:
    try:
        notes.append(best_df[best_cols].to_markdown(index=False))
    except Exception:
        notes.append(best_df[best_cols].to_csv(index=False))
else:
    notes.append("No disponible.")

notes_path = OUT_DIR / "mc500_comparison_notes.md"
notes_path.write_text("\n".join(notes), encoding="utf-8")

print("comparison_csv:", full_csv)
print("comparison_md:", full_md)
print("pivot_accuracy:", OUT_DIR / "mc500_pivot_accuracy.csv")
print("pivot_tokens:", OUT_DIR / "mc500_pivot_tokens.csv")
print("best_by_dataset_csv:", best_csv)
print("best_by_dataset_md:", best_md)
print("notes_md:", notes_path)

print()
print("== Tabla principal ==")
print(full[display_cols].to_string(index=False))

print()
print("== Accuracy pivot ==")
if pivot_acc.empty:
    print("No disponible")
else:
    print(pivot_acc.to_string())

print()
print("== Mejor sistema por dataset ==")
if best_df.empty:
    print("No disponible")
else:
    print(best_df[best_cols].to_string(index=False))

if not musique_dir:
    print()
    print("WARNING: no encontré MuSiQue automáticamente. El script dejó una comparación parcial.")
else:
    print()
    print("OK: comparación cross-dataset generada.")
PY

echo
echo "== 2. Archivos generados =="
find "$OUT_DIR" -maxdepth 1 -type f | sort

echo
echo "== 3. Estado Git resumido =="
git status --short

echo
echo "LISTO: comparación cross-dataset generada."

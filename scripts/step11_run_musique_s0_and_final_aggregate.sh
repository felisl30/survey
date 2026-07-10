#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 11 — Completar MuSiQue-MC-500 con S0 gpt-5-mini
#          y regenerar comparación cross-dataset final
#
# Este paso SÍ llama al LLM, pero solo para S0 MuSiQue.
#
# Motivo:
#   Ya existen S1/S2/S3 completos para:
#     outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/
#
#   Falta:
#     outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/s0_evaluated.csv
#
# Salidas:
#   outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/s0_raw.csv
#   outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/s0_parsed.csv
#   outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/s0_evaluated.csv
#   outputs/eval_mc/musique_mc_rag_500/gpt_5_mini/s0_summary.json
#
#   outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.csv
#   outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.md
#   outputs/eval_mc/cross_dataset/mc500_pivot_accuracy_final.csv
#   outputs/eval_mc/cross_dataset/mc500_pivot_tokens_final.csv
#   outputs/eval_mc/cross_dataset/mc500_best_by_dataset_final.csv
#   outputs/eval_mc/cross_dataset/mc500_comparison_final_notes.md
#
# La corrida usa --resume.
# ============================================================

MODEL="${MODEL:-gpt-5-mini}"
MUSIQUE_DIR="outputs/eval_mc/musique_mc_rag_500/gpt_5_mini"
MUSIQUE_QUESTIONS="data/eval_mc/musique_mc_rag_500/questions.csv"
OUT_DIR="outputs/eval_mc/cross_dataset"

mkdir -p "$MUSIQUE_DIR" "$OUT_DIR"

echo "== 0. Configuración =="
echo "MODEL=$MODEL"
echo "MUSIQUE_QUESTIONS=$MUSIQUE_QUESTIONS"
echo "MUSIQUE_DIR=$MUSIQUE_DIR"
echo "OUT_DIR=$OUT_DIR"

echo
echo "== 1. Precheck =="
test -f "$MUSIQUE_QUESTIONS"
test -f modelos/s0/run_s0_direct.py
test -f modelos/s0/parse_s0_outputs.py
test -f modelos/s0/evaluate_s0.py

for s in s1 s2 s3; do
  test -f "$MUSIQUE_DIR/${s}_evaluated.csv"
  echo "OK: existe $MUSIQUE_DIR/${s}_evaluated.csv"
done

echo "OK: inputs necesarios encontrados."

echo
echo "== 2. Preflight del modelo =="
python - "$MODEL" <<'PY'
import sys
from direct_llm import ask_direct_llm_with_metadata

model = sys.argv[1]
result = ask_direct_llm_with_metadata(
    'Return exactly this JSON: {"answer":"A","confidence":1.0}',
    model=model,
    max_retries=0,
)
print(f"MODEL_OK {model} total_tokens={result.get('total_tokens')}")
PY

echo
echo "== 3. Correr S0 MuSiQue-MC-500 completo =="
python modelos/s0/run_s0_direct.py \
  --input-path "$MUSIQUE_QUESTIONS" \
  --output-path "$MUSIQUE_DIR/s0_raw.csv" \
  --model "$MODEL" \
  --save-every 1 \
  --resume

echo
echo "== 4. Parsear y evaluar S0 MuSiQue =="
python modelos/s0/parse_s0_outputs.py \
  --input-path "$MUSIQUE_DIR/s0_raw.csv" \
  --output-path "$MUSIQUE_DIR/s0_parsed.csv"

python modelos/s0/evaluate_s0.py \
  --input-path "$MUSIQUE_DIR/s0_parsed.csv" \
  --output-path "$MUSIQUE_DIR/s0_evaluated.csv" \
  --summary-path "$MUSIQUE_DIR/s0_summary.json" \
  --group-summary-path "$MUSIQUE_DIR/s0_group_summary.csv"

echo
echo "== 5. Verificar MuSiQue S0-S3 =="
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path("outputs/eval_mc/musique_mc_rag_500/gpt_5_mini")

def normalize_choice(x):
    if pd.isna(x):
        return ""
    t = str(x).strip().upper()
    if t and t[0] in {"A", "B", "C", "D"}:
        return t[0]
    return t

rows = []
for s in ["s0", "s1", "s2", "s3"]:
    p = root / f"{s}_evaluated.csv"
    sp = root / f"{s}_summary.json"
    row = {"system": s, "path": str(p), "exists": p.exists(), "summary_exists": sp.exists()}
    if p.exists():
        df = pd.read_csv(p)
        row["n"] = len(df)
        if "parsed_answer" in df.columns and "gold_answer" in df.columns:
            pred = df["parsed_answer"].map(normalize_choice)
            gold = df["gold_answer"].map(normalize_choice)
            valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
            row["accuracy"] = float((pred[valid] == gold[valid]).mean())
            row["accuracy_n"] = int(valid.sum())
        if "valid_format" in df.columns:
            row["valid_format_rate"] = float(pd.to_numeric(df["valid_format"], errors="coerce").mean())
        if "run_error" in df.columns:
            row["run_error_rate"] = float(pd.to_numeric(df["run_error"], errors="coerce").mean())
        if "total_tokens" in df.columns:
            row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())
        if "latency_seconds" in df.columns:
            row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())
    rows.append(row)

out = pd.DataFrame(rows)
cols = [c for c in ["system", "n", "accuracy", "valid_format_rate", "run_error_rate", "avg_total_tokens", "avg_latency_seconds", "exists", "summary_exists", "path"] if c in out.columns]
print(out[cols].to_string(index=False))

bad = []
for _, r in out.iterrows():
    if not bool(r.get("exists", False)):
        bad.append(f"{r['system']} missing")
    if int(r.get("n", 0)) != 500:
        bad.append(f"{r['system']} n={r.get('n')}")
    if float(r.get("valid_format_rate", 1.0)) < 0.95:
        bad.append(f"{r['system']} valid_format_rate={r.get('valid_format_rate')}")
    if float(r.get("run_error_rate", 0.0)) > 0.05:
        bad.append(f"{r['system']} run_error_rate={r.get('run_error_rate')}")

if bad:
    print("WARNING:", "; ".join(bad))
else:
    print("OK: MuSiQue S0-S3 completo y consistente.")
PY

echo
echo "== 6. Regenerar comparación final MuSiQue + HotpotQA + 2Wiki =="
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

OUT_DIR = Path("outputs/eval_mc/cross_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["s0", "s1", "s2", "s3"]

DATASETS = [
    ("MuSiQue-MC-500", "musique", Path("outputs/eval_mc/musique_mc_rag_500/gpt_5_mini")),
    ("HotpotQA-MC-500", "hotpotqa", Path("outputs/eval_mc/hotpotqa_mc_rag_500/gpt_5_mini")),
    ("2Wiki-MC-500", "2wiki", Path("outputs/eval_mc/2wiki_mc_rag_500/gpt_5_mini")),
]

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

def summarize(dataset_label, dataset_short, root):
    rows = []
    for s in SYSTEMS:
        p = root / f"{s}_evaluated.csv"
        sp = root / f"{s}_summary.json"
        row = {
            "dataset": dataset_label,
            "dataset_short": dataset_short,
            "system": s,
            "eval_dir": str(root),
            "evaluated_exists": p.exists(),
            "summary_exists": sp.exists(),
        }

        if not p.exists():
            rows.append(row)
            continue

        df = pd.read_csv(p)
        row["n"] = len(df)

        if "parsed_answer" in df.columns and "gold_answer" in df.columns:
            pred = df["parsed_answer"].map(normalize_choice)
            gold = df["gold_answer"].map(normalize_choice)
            valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
            if valid.any():
                row["accuracy"] = float((pred[valid] == gold[valid]).mean())
                row["accuracy_n"] = int(valid.sum())
                row["accuracy_source"] = "parsed_answer_vs_gold_answer"

        if "valid_format" in df.columns:
            row["valid_format_rate"] = boolish_mean(df["valid_format"])

        if "run_error" in df.columns:
            row["run_error_rate"] = boolish_mean(df["run_error"])

        if "total_tokens" in df.columns:
            row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())

        if "latency_seconds" in df.columns:
            row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())

        if sp.exists():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
            for key in ["accuracy", "correct_rate"]:
                if "accuracy" not in row and key in summary:
                    row["accuracy"] = float(summary[key])
                    row["accuracy_source"] = f"summary.{key}"
            for key in ["valid_format_rate", "run_error_rate", "avg_total_tokens", "avg_latency_seconds", "active_retrieval_rate", "retrieve_rate"]:
                if key not in row and key in summary:
                    try:
                        row[key] = float(summary[key])
                    except Exception:
                        pass

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

full = pd.concat([summarize(*d) for d in DATASETS], ignore_index=True)

dataset_order = {"MuSiQue-MC-500": 0, "HotpotQA-MC-500": 1, "2Wiki-MC-500": 2}
system_order = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
full["dataset_order"] = full["dataset"].map(dataset_order)
full["system_order"] = full["system"].map(system_order)
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
    "accuracy_source",
    "eval_dir",
] if c in full.columns]

comparison_csv = OUT_DIR / "mc500_s0_s3_comparison_final.csv"
comparison_md = OUT_DIR / "mc500_s0_s3_comparison_final.md"
full[display_cols].to_csv(comparison_csv, index=False)
try:
    comparison_md.write_text(full[display_cols].to_markdown(index=False), encoding="utf-8")
except Exception:
    comparison_md.write_text(full[display_cols].to_csv(index=False), encoding="utf-8")

pivot_acc = full.pivot_table(index="dataset", columns="system", values="accuracy", aggfunc="first")
pivot_acc = pivot_acc.reindex(["MuSiQue-MC-500", "HotpotQA-MC-500", "2Wiki-MC-500"])
pivot_acc.to_csv(OUT_DIR / "mc500_pivot_accuracy_final.csv")

pivot_tok = full.pivot_table(index="dataset", columns="system", values="avg_total_tokens", aggfunc="first")
pivot_tok = pivot_tok.reindex(["MuSiQue-MC-500", "HotpotQA-MC-500", "2Wiki-MC-500"])
pivot_tok.to_csv(OUT_DIR / "mc500_pivot_tokens_final.csv")

best_rows = []
for dataset, g in full.dropna(subset=["accuracy"]).groupby("dataset"):
    best = g.sort_values(["accuracy", "avg_total_tokens"], ascending=[False, True]).iloc[0].to_dict()
    s0 = g[g["system"] == "s0"]
    if not s0.empty:
        best["best_delta_vs_s0"] = best["accuracy"] - float(s0["accuracy"].iloc[0])
    best_rows.append(best)

best_df = pd.DataFrame(best_rows)
best_cols = [c for c in ["dataset", "system", "accuracy", "best_delta_vs_s0", "avg_total_tokens", "token_ratio_vs_s0", "avg_latency_seconds"] if c in best_df.columns]
best_csv = OUT_DIR / "mc500_best_by_dataset_final.csv"
best_md = OUT_DIR / "mc500_best_by_dataset_final.md"
best_df[best_cols].to_csv(best_csv, index=False)
try:
    best_md.write_text(best_df[best_cols].to_markdown(index=False), encoding="utf-8")
except Exception:
    best_md.write_text(best_df[best_cols].to_csv(index=False), encoding="utf-8")

notes = []
notes.append("# Comparación final MC-500 S0/S1/S2/S3\n")
notes.append("## Tabla principal\n")
try:
    notes.append(full[display_cols].to_markdown(index=False))
except Exception:
    notes.append(full[display_cols].to_csv(index=False))
notes.append("\n## Accuracy pivot\n")
try:
    notes.append(pivot_acc.to_markdown())
except Exception:
    notes.append(pivot_acc.to_csv())
notes.append("\n## Mejor sistema por dataset\n")
try:
    notes.append(best_df[best_cols].to_markdown(index=False))
except Exception:
    notes.append(best_df[best_cols].to_csv(index=False))
notes.append("\n## Lectura breve\n")
notes.append("- La comparación usa 500 casos por dataset y `gpt-5-mini`.")
notes.append("- S0 es baseline directo sin recuperación.")
notes.append("- S1/S2/S3 incorporan recuperación con distintas políticas.")
notes.append("- `delta_vs_s0` mide mejora absoluta de accuracy sobre el baseline directo dentro del mismo dataset.")
notes_path = OUT_DIR / "mc500_comparison_final_notes.md"
notes_path.write_text("\n".join(notes), encoding="utf-8")

print("comparison_csv:", comparison_csv)
print("comparison_md:", comparison_md)
print("pivot_accuracy:", OUT_DIR / "mc500_pivot_accuracy_final.csv")
print("pivot_tokens:", OUT_DIR / "mc500_pivot_tokens_final.csv")
print("best_by_dataset_csv:", best_csv)
print("best_by_dataset_md:", best_md)
print("notes_md:", notes_path)

print()
print("== Tabla principal final ==")
print(full[display_cols].to_string(index=False))

print()
print("== Accuracy pivot final ==")
print(pivot_acc.to_string())

print()
print("== Mejor sistema por dataset final ==")
print(best_df[best_cols].to_string(index=False))

bad = []
for _, r in full.iterrows():
    if int(r.get("n", 0)) != 500:
        bad.append(f"{r['dataset']} {r['system']} n={r.get('n')}")
    if float(r.get("valid_format_rate", 1.0)) < 0.95:
        bad.append(f"{r['dataset']} {r['system']} valid_format_rate={r.get('valid_format_rate')}")
    if float(r.get("run_error_rate", 0.0)) > 0.05:
        bad.append(f"{r['dataset']} {r['system']} run_error_rate={r.get('run_error_rate')}")
    if pd.isna(r.get("accuracy")):
        bad.append(f"{r['dataset']} {r['system']} accuracy missing")

if bad:
    print()
    print("WARNING:", "; ".join(bad))
else:
    print()
    print("OK: comparación final completa y consistente.")
PY

echo
echo "== 7. Archivos finales generados =="
find "$OUT_DIR" -maxdepth 1 -type f | grep -E 'final|musique' | sort

echo
echo "== 8. Estado Git resumido =="
git status --short

echo
echo "LISTO: MuSiQue S0 completado y comparación final generada."

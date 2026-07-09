#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 10B v2 — Buscar outputs MuSiQue S0/S1/S2/S3
#
# Este paso NO llama al LLM.
#
# Corrige el error del script anterior y hace una búsqueda más robusta:
#   1) lista archivos evaluated relacionados con MuSiQue;
#   2) busca carpetas completas s0-s3;
#   3) busca carpetas RAG con s1-s3 aunque falte s0;
#   4) busca posibles s0 aunque estén en otra carpeta;
#   5) imprime una recomendación de combinación probable.
# ============================================================

BASE="${BASE:-outputs/eval_mc}"
OUT="outputs/eval_mc/cross_dataset/musique_candidates_audit_v2.txt"
mkdir -p outputs/eval_mc/cross_dataset

# Guardar toda la salida también en archivo.
exec > >(tee "$OUT") 2>&1

echo "== 0. Configuración =="
echo "BASE=$BASE"
echo "OUT=$OUT"

echo
echo "== 1. Archivos evaluated relacionados con MuSiQue por ruta =="
find "$BASE" -type f \
  \( -name "s0_evaluated.csv" -o -name "s1_evaluated.csv" -o -name "s2_evaluated.csv" -o -name "s3_evaluated.csv" \) \
  | grep -Ei 'musique|mu_si_que|musi' \
  | sort || true

echo
echo "== 2. Auditoría robusta de candidatos MuSiQue =="
python - <<'PY'
from pathlib import Path
import json
import re

import pandas as pd

base = Path("outputs/eval_mc")
systems = ["s0", "s1", "s2", "s3"]

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

def read_summary(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def infer_system(path):
    m = re.search(r"(s[0-5])_evaluated\.csv$", path.name)
    return m.group(1) if m else None

def summarize_eval_file(path):
    path = Path(path)
    system = infer_system(path)
    summary_path = path.with_name(path.name.replace("_evaluated.csv", "_summary.json"))
    row = {
        "system": system,
        "path": str(path),
        "dir": str(path.parent),
        "summary_exists": summary_path.exists(),
    }

    try:
        df = pd.read_csv(path)
    except Exception as e:
        row["read_error"] = str(e)
        return row

    row["n"] = len(df)
    row["columns_n"] = len(df.columns)

    # dataset values, if any.
    for col in ["dataset", "source_dataset", "benchmark_name"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            row[col + "_values"] = "|".join(vals[:8])

    # Accuracy.
    if "parsed_answer" in df.columns and "gold_answer" in df.columns:
        pred = df["parsed_answer"].map(normalize_choice)
        gold = df["gold_answer"].map(normalize_choice)
        valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
        if valid.any():
            row["accuracy"] = float((pred[valid] == gold[valid]).mean())
            row["accuracy_n"] = int(valid.sum())
            row["accuracy_source"] = "parsed_answer_vs_gold_answer"

    for col, out in [
        ("valid_format", "valid_format_rate"),
        ("run_error", "run_error_rate"),
    ]:
        if col in df.columns:
            row[out] = boolish_mean(df[col])

    if "total_tokens" in df.columns:
        row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())

    if "latency_seconds" in df.columns:
        row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())

    summary = read_summary(summary_path)
    if "accuracy" not in row:
        for key in ["accuracy", "correct_rate", "accuracy_decided", "correct_rate_decided"]:
            if key in summary:
                try:
                    row["accuracy"] = float(summary[key])
                    row["accuracy_source"] = f"summary.{key}"
                    break
                except Exception:
                    pass

    for key in ["valid_format_rate", "run_error_rate", "avg_total_tokens", "avg_latency_seconds"]:
        if key not in row and key in summary:
            try:
                row[key] = float(summary[key])
            except Exception:
                pass

    # Mark whether content appears MuSiQue by path or by columns.
    hay = " ".join(str(row.get(k, "")) for k in [
        "path", "dataset_values", "source_dataset_values", "benchmark_name_values"
    ]).lower()
    row["looks_musique"] = any(token in hay for token in ["musique", "mu_si_que", "musi"])
    row["looks_smoke"] = "smoke" in str(path).lower()
    row["looks_limit"] = "limit" in str(path).lower()
    row["looks_robustness"] = "robustness" in str(path).lower()
    row["looks_posthoc"] = "posthoc" in str(path).lower()
    row["looks_focus"] = "focus" in str(path).lower()

    return row

def print_df(title, df, max_rows=30):
    print()
    print(title)
    if df.empty:
        print("No disponible.")
        return
    cols = [c for c in [
        "system", "n", "accuracy", "valid_format_rate", "run_error_rate",
        "avg_total_tokens", "avg_latency_seconds",
        "dataset_values", "source_dataset_values", "benchmark_name_values",
        "path"
    ] if c in df.columns]
    print(df[cols].head(max_rows).to_string(index=False))

# All evaluated files under outputs/eval_mc, then filter by path/content.
all_files = sorted(base.rglob("s*_evaluated.csv"))
rows = [summarize_eval_file(p) for p in all_files]
all_df = pd.DataFrame(rows)

if all_df.empty:
    raise SystemExit("ERROR: no encontré archivos *_evaluated.csv en outputs/eval_mc")

musique_df = all_df[all_df.get("looks_musique", False) == True].copy()
musique_df = musique_df.sort_values(["system", "path"])

print("total_evaluated_files:", len(all_df))
print("musique_related_files:", len(musique_df))

print_df("== 2A. Todos los archivos MuSiQue detectados ==", musique_df, max_rows=80)

# Complete dirs s0-s3 in same folder.
complete_dirs = []
for d, g in musique_df.groupby("dir"):
    systems_here = set(g["system"].dropna())
    if set(systems).issubset(systems_here):
        complete_dirs.append(Path(d))

def dir_score(d):
    s = str(d).lower()
    value = 0
    if "gpt_5_mini" in s or "gpt-5-mini" in s:
        value += 100
    if "musique_mc_rag_500" in s:
        value += 80
    if "500" in s:
        value += 30
    if "smoke" in s:
        value -= 100
    if "limit" in s:
        value -= 70
    if "posthoc" in s:
        value -= 60
    if "focus" in s:
        value -= 60
    if "robustness" in s:
        value -= 50
    return value

complete_dirs = sorted(complete_dirs, key=lambda d: (dir_score(d), d.stat().st_mtime), reverse=True)

print()
print("== 2B. Carpetas completas s0-s3 en el mismo directorio ==")
if not complete_dirs:
    print("No encontré carpetas completas s0-s3.")
else:
    for i, d in enumerate(complete_dirs, 1):
        print()
        print(f"--- COMPLETE CANDIDATE {i} ---")
        print("dir:", d)
        print("score:", dir_score(d))
        df = musique_df[musique_df["dir"] == str(d)].sort_values("system")
        print_df("resumen:", df, max_rows=10)

# RAG dirs with s1-s3.
rag_dirs = []
for d, g in musique_df.groupby("dir"):
    systems_here = set(g["system"].dropna())
    if {"s1", "s2", "s3"}.issubset(systems_here):
        rag_dirs.append(Path(d))
rag_dirs = sorted(rag_dirs, key=lambda d: (dir_score(d), d.stat().st_mtime), reverse=True)

print()
print("== 2C. Carpetas con S1-S3 completas, aunque falte S0 ==")
if not rag_dirs:
    print("No encontré carpetas con S1-S3.")
else:
    for i, d in enumerate(rag_dirs, 1):
        print()
        print(f"--- RAG CANDIDATE {i} ---")
        print("dir:", d)
        print("score:", dir_score(d))
        df = musique_df[musique_df["dir"] == str(d)].sort_values("system")
        print_df("resumen:", df, max_rows=10)

# S0 candidates anywhere musique-related.
s0_df = musique_df[musique_df["system"] == "s0"].copy()
if not s0_df.empty:
    def s0_score(row):
        s = str(row["path"]).lower()
        v = 0
        if "gpt_5_mini" in s or "gpt-5-mini" in s:
            v += 100
        if int(row.get("n", 0) or 0) == 500:
            v += 50
        if "model_grid_musique" in s:
            v += 20
        if "smoke" in s:
            v -= 100
        if "limit" in s:
            v -= 60
        if "robustness" in s:
            v -= 50
        return v
    s0_df["candidate_score"] = s0_df.apply(s0_score, axis=1)
    s0_df = s0_df.sort_values(["candidate_score", "path"], ascending=[False, True])

print()
print("== 2D. Posibles S0 MuSiQue ==")
print_df("s0 candidates:", s0_df, max_rows=50)

# Recommendation.
print()
print("== 2E. Recomendación automática ==")
if complete_dirs:
    best = complete_dirs[0]
    print("Usar carpeta completa:")
    print(best)
elif rag_dirs and not s0_df.empty:
    best_rag = rag_dirs[0]
    best_s0 = Path(s0_df.iloc[0]["path"]).parent
    print("No hay carpeta única completa. Recomiendo combinar:")
    print("MUSIQUE_RAG_DIR=", best_rag)
    print("MUSIQUE_S0_DIR=", best_s0)
    print()
    print("Ojo: revisar que el modelo de S0 sea comparable con S1-S3.")
else:
    print("No puedo recomendar combinación todavía.")
    print("Necesitamos ubicar al menos S1-S3 y un S0 de MuSiQue.")

# Save a machine-readable candidate table.
out_csv = Path("outputs/eval_mc/cross_dataset/musique_candidates_audit_v2.csv")
musique_df.to_csv(out_csv, index=False)
print()
print("candidate_table_csv:", out_csv)
PY

echo
echo "== 3. Estado Git resumido =="
git status --short

echo
echo "LISTO: búsqueda MuSiQue v2 terminada."

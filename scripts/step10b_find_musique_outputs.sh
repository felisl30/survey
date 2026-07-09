#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paso 10B — Buscar outputs MuSiQue S0/S1/S2/S3 completos
#
# Este paso NO llama al LLM.
#
# Objetivo:
#   Encontrar carpetas de MuSiQue que tengan:
#     s0_evaluated.csv
#     s1_evaluated.csv
#     s2_evaluated.csv
#     s3_evaluated.csv
#
# También imprime un resumen de accuracy/tokens si puede leerlas.
# ============================================================

BASE="${BASE:-outputs/eval_mc}"
OUT="outputs/eval_mc/cross_dataset/musique_candidates_audit.txt"
mkdir -p outputs/eval_mc/cross_dataset

echo "== 0. Configuración =="
echo "BASE=$BASE"
echo "OUT=$OUT"

echo
echo "== 1. Buscar archivos evaluated de MuSiQue =="
find "$BASE" -type f \
  \( -name "s0_evaluated.csv" -o -name "s1_evaluated.csv" -o -name "s2_evaluated.csv" -o -name "s3_evaluated.csv" \) \
  | grep -Ei 'musique|mu_si_que|musi' \
  | sort \
  | tee "$OUT"

echo
echo "== 2. Buscar carpetas candidatas completas =="
python - <<'PY'
from pathlib import Path
import json

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

def summarize_dir(d):
    rows = []
    for s in systems:
        p = d / f"{s}_evaluated.csv"
        sp = d / f"{s}_summary.json"

        row = {"system": s, "path": str(p), "exists": p.exists()}
        if p.exists():
            df = pd.read_csv(p)
            row["n"] = len(df)

            if "parsed_answer" in df.columns and "gold_answer" in df.columns:
                pred = df["parsed_answer"].map(normalize_choice)
                gold = df["gold_answer"].map(normalize_choice)
                valid = pred.isin(["A", "B", "C", "D"]) & gold.isin(["A", "B", "C", "D"])
                if valid.any():
                    row["accuracy"] = float((pred[valid] == gold[valid]).mean())

            if "valid_format" in df.columns:
                row["valid_format_rate"] = float(pd.to_numeric(df["valid_format"], errors="coerce").mean())

            if "run_error" in df.columns:
                row["run_error_rate"] = float(pd.to_numeric(df["run_error"], errors="coerce").mean())

            if "total_tokens" in df.columns:
                row["avg_total_tokens"] = float(pd.to_numeric(df["total_tokens"], errors="coerce").mean())

            if "latency_seconds" in df.columns:
                row["avg_latency_seconds"] = float(pd.to_numeric(df["latency_seconds"], errors="coerce").mean())

        if sp.exists():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
                for k in ["accuracy", "correct_rate", "valid_format_rate", "run_error_rate", "avg_total_tokens", "avg_latency_seconds"]:
                    if k in summary and k not in row:
                        row[k] = summary[k]
            except Exception as e:
                row["summary_error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)

candidate_dirs = []
for p in base.rglob("s0_evaluated.csv"):
    d = p.parent
    d_str = str(d).lower()
    if not any(token in d_str for token in ["musique", "mu_si_que", "musi"]):
        continue
    if all((d / f"{s}_evaluated.csv").exists() for s in systems):
        candidate_dirs.append(d)

def score(d):
    s = str(d).lower()
    value = 0
    if "gpt_5_mini" in s:
        value += 100
    if "500" in s:
        value += 30
    if "smoke" in s:
        value -= 100
    if "limit" in s:
        value -= 50
    if "posthoc" in s:
        value -= 40
    if "focus" in s:
        value -= 40
    if "s4" in s:
        value -= 40
    return value

candidate_dirs = sorted(candidate_dirs, key=lambda d: (score(d), d.stat().st_mtime), reverse=True)

print("candidate_dirs_count:", len(candidate_dirs))
for i, d in enumerate(candidate_dirs, 1):
    print()
    print(f"===== CANDIDATE {i} =====")
    print("dir:", d)
    print("score:", score(d))
    df = summarize_dir(d)
    cols = [c for c in [
        "system", "n", "accuracy", "valid_format_rate", "run_error_rate",
        "avg_total_tokens", "avg_latency_seconds", "path"
    ] if c in df.columns]
    print(df[cols].to_string(index=False))

if not candidate_dirs:
    print()
    print("WARNING: no encontré ninguna carpeta completa de MuSiQue.")
    print("Sugerencia: correr los comandos del bloque 3 para listar rutas relacionadas.")
PY | tee -a "$OUT"

echo
echo "== 3. Búsqueda amplia de rutas MuSiQue relevantes =="
find outputs -type f 2>/dev/null \
  | grep -Ei 'musique|mu_si_que|musi' \
  | grep -Ei 'summary|evaluated|group_summary|raw|parsed' \
  | sort \
  | sed -n '1,200p'

echo
echo "== 4. Estado Git resumido =="
git status --short

echo
echo "LISTO: búsqueda de outputs MuSiQue terminada."

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE = Path("outputs/eval_mc/model_grid_musique")
OUT_DIR = BASE / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gpt-5.4-nano": "gpt_5_4_nano",
    "gpt-5.4-mini": "gpt_5_4_mini",
    "gpt-5.4": "gpt_5_4",
}

SYSTEMS = ["s0", "s1", "s2", "s3_mc"]

META_COLS = [
    "original_question",
    "question",
    "A", "B", "C", "D",
    "option_a", "option_b", "option_c", "option_d",
    "gold_answer",
    "correct_answer",
    "dataset",
]


def find_col(df: pd.DataFrame, candidates, required=True, label="column"):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"No encontré {label}. Candidatas: {candidates}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    return None


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "correct": True,
        "incorrect": False,
    }).fillna(False)


def read_eval(model_tag: str, system: str) -> pd.DataFrame:
    path = BASE / model_tag / f"{system}_evaluated.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    id_col = find_col(
        df,
        ["id", "question_id", "example_id"],
        required=True,
        label="id column",
    )

    correct_col = find_col(
        df,
        ["eval_correct", "mc_correct", "correct", "is_correct"],
        required=True,
        label="correctness column",
    )

    answer_col = find_col(
        df,
        ["parsed_answer", "mc_pred", "predicted_answer", "answer", "final_answer"],
        required=False,
        label="answer column",
    )

    rename = {}
    keep = []

    # ID normalizado.
    keep.append(id_col)
    if id_col != "id":
        rename[id_col] = "id"

    # Metadatos no prefijados. Luego se conservan solo desde el primer sistema.
    for col in META_COLS:
        if col in df.columns and col not in keep:
            keep.append(col)

    # Columnas propias del sistema, siempre prefijadas.
    system_col_map = {
        correct_col: "correct",
    }

    if answer_col is not None:
        system_col_map[answer_col] = "answer"

    optional_map = {
        "valid_format": "valid_format",
        "run_error_present": "run_error",
        "total_tokens": "tokens",
        "latency_seconds": "latency",
        "retrieval_used": "retrieval_used",
        "retrieval_triggered": "retrieval_triggered",
        "active_retrieval_triggered": "active_retrieval",
        "predicted_route": "route",
        "route": "route",
        "confidence": "confidence",
        "final_confidence": "confidence",
    }

    for col, alias in optional_map.items():
        if col in df.columns:
            system_col_map[col] = alias

    for col, alias in system_col_map.items():
        if col in df.columns and col not in keep:
            keep.append(col)
        rename[col] = f"{system}_{alias}"

    out = df[keep].copy()
    out = out.rename(columns=rename)

    correct_name = f"{system}_correct"
    out[correct_name] = to_bool(out[correct_name])

    return out


all_summary_rows = []
all_interesting_cases = []

for model_name, tag in MODELS.items():
    merged = None

    for system in SYSTEMS:
        df = read_eval(tag, system)

        if merged is None:
            merged = df
        else:
            # Evita duplicar pregunta/opciones/gold en cada merge.
            df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")
            merged = merged.merge(df, on="id", how="outer")

    # Asegurar booleanos.
    for system in SYSTEMS:
        col = f"{system}_correct"
        if col not in merged.columns:
            raise KeyError(f"Falta columna esperada: {col}")
        merged[col] = to_bool(merged[col])

    merged.insert(0, "model", model_name)

    # Patrones vs S0.
    for system in ["s1", "s2", "s3_mc"]:
        merged[f"{system}_corrected_s0"] = (~merged["s0_correct"]) & merged[f"{system}_correct"]
        merged[f"{system}_regressed_s0"] = merged["s0_correct"] & (~merged[f"{system}_correct"])

    # Patrones entre sistemas.
    correct_cols = [f"{s}_correct" for s in SYSTEMS]
    merged["all_systems_correct"] = merged[correct_cols].all(axis=1)
    merged["all_systems_wrong"] = (~merged[correct_cols]).all(axis=1)

    merged["only_s3_correct"] = (
        (~merged["s0_correct"])
        & (~merged["s1_correct"])
        & (~merged["s2_correct"])
        & merged["s3_mc_correct"]
    )

    merged["s2_beats_s1_case"] = (~merged["s1_correct"]) & merged["s2_correct"]
    merged["s1_beats_s2_case"] = merged["s1_correct"] & (~merged["s2_correct"])

    # Guardar comparación por pregunta.
    per_question_path = OUT_DIR / f"{tag}_question_level_comparison.csv"
    merged.to_csv(per_question_path, index=False)

    # Resumen por sistema.
    for system in SYSTEMS:
        token_col = f"{system}_tokens"
        latency_col = f"{system}_latency"
        valid_col = f"{system}_valid_format"
        run_error_col = f"{system}_run_error"

        all_summary_rows.append({
            "model": model_name,
            "section": "system_accuracy",
            "system": system,
            "n": int(len(merged)),
            "accuracy": float(merged[f"{system}_correct"].mean()),
            "avg_tokens": float(pd.to_numeric(merged[token_col], errors="coerce").mean())
                if token_col in merged.columns else None,
            "avg_latency": float(pd.to_numeric(merged[latency_col], errors="coerce").mean())
                if latency_col in merged.columns else None,
            "valid_format_rate": float(to_bool(merged[valid_col]).mean())
                if valid_col in merged.columns else None,
            "run_error_rate": float(to_bool(merged[run_error_col]).mean())
                if run_error_col in merged.columns else None,
        })

    # Resumen de correcciones/regresiones vs S0.
    for system in ["s1", "s2", "s3_mc"]:
        corrected = int(merged[f"{system}_corrected_s0"].sum())
        regressed = int(merged[f"{system}_regressed_s0"].sum())

        all_summary_rows.append({
            "model": model_name,
            "section": "vs_s0",
            "system": system,
            "n": int(len(merged)),
            "corrected_s0_errors": corrected,
            "regressed_s0_correct": regressed,
            "net_gain_vs_s0": corrected - regressed,
        })

    # Resumen de patrones.
    all_summary_rows.append({
        "model": model_name,
        "section": "case_patterns",
        "system": "patterns",
        "n": int(len(merged)),
        "all_systems_correct": int(merged["all_systems_correct"].sum()),
        "all_systems_wrong": int(merged["all_systems_wrong"].sum()),
        "only_s3_correct": int(merged["only_s3_correct"].sum()),
        "s2_beats_s1_cases": int(merged["s2_beats_s1_case"].sum()),
        "s1_beats_s2_cases": int(merged["s1_beats_s2_case"].sum()),
    })

    # Casos interesantes para inspección manual.
    interesting_mask = (
        merged["only_s3_correct"]
        | merged["s2_beats_s1_case"]
        | merged["s1_beats_s2_case"]
        | merged["s1_corrected_s0"]
        | merged["s2_corrected_s0"]
        | merged["s3_mc_corrected_s0"]
        | merged["s1_regressed_s0"]
        | merged["s2_regressed_s0"]
        | merged["s3_mc_regressed_s0"]
    )

    interesting = merged[interesting_mask].copy()
    all_interesting_cases.append(interesting)

summary = pd.DataFrame(all_summary_rows)

summary_path = OUT_DIR / "model_grid_error_pattern_summary.csv"
txt_path = OUT_DIR / "model_grid_error_pattern_summary.txt"

summary.to_csv(summary_path, index=False)

with txt_path.open("w", encoding="utf-8") as f:
    f.write("Model grid error-pattern analysis\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary.to_string(index=False))
    f.write("\n")

if all_interesting_cases:
    interesting_all = pd.concat(all_interesting_cases, ignore_index=True)
    interesting_path = OUT_DIR / "model_grid_interesting_cases.csv"
    interesting_all.to_csv(interesting_path, index=False)
else:
    interesting_path = None

print(summary.to_string(index=False))
print()
print("Guardado:")
print(summary_path)
print(txt_path)
if interesting_path:
    print(interesting_path)

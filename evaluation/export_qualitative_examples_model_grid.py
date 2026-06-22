#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd

BASE = Path("outputs/eval_mc/model_grid_musique/analysis")
IN = BASE / "model_grid_interesting_cases.csv"
OUT_MD = BASE / "model_grid_qualitative_examples.md"
OUT_CSV = BASE / "model_grid_qualitative_examples.csv"

df = pd.read_csv(IN)

def clean_question(prompt: str) -> str:
    text = str(prompt)
    m = re.search(r"Pregunta:\s*(.*?)\n\s*Opciones:", text, flags=re.S)
    if m:
        return " ".join(m.group(1).split())
    return " ".join(text.split())

def clean_options(prompt: str) -> str:
    text = str(prompt)
    m = re.search(r"Opciones:\s*(.*?)\n\s*Formato obligatorio:", text, flags=re.S)
    if m:
        return "\n".join(line.strip() for line in m.group(1).splitlines() if line.strip())
    return ""

def label_case(row) -> str:
    if bool(row.get("only_s3_correct", False)):
        return "only_s3_correct"
    if bool(row.get("s2_beats_s1_case", False)):
        return "s2_beats_s1_case"
    if bool(row.get("s1_beats_s2_case", False)):
        return "s1_beats_s2_case"
    return "other"

df["case_type"] = df.apply(label_case, axis=1)
df["question_clean"] = df["question"].map(clean_question)
df["options_clean"] = df["question"].map(clean_options)

# Elegimos pocos ejemplos por modelo/categoría para inspección humana.
selected = (
    df[df["case_type"].isin(["only_s3_correct", "s2_beats_s1_case", "s1_beats_s2_case"])]
    .sort_values(["model", "case_type", "id"])
    .groupby(["model", "case_type"], as_index=False)
    .head(3)
    .copy()
)

cols = [
    "model", "case_type", "id",
    "question_clean", "options_clean", "gold_answer",
    "s0_answer", "s0_correct",
    "s1_answer", "s1_correct",
    "s2_answer", "s2_correct",
    "s3_mc_answer", "s3_mc_correct",
]
cols = [c for c in cols if c in selected.columns]

selected[cols].to_csv(OUT_CSV, index=False)

with OUT_MD.open("w", encoding="utf-8") as f:
    f.write("# Qualitative examples: model grid S0-S3\n\n")

    for (model, case_type), sub in selected.groupby(["model", "case_type"]):
        f.write(f"## {model} — {case_type}\n\n")

        for _, row in sub.iterrows():
            f.write(f"### {row['id']}\n\n")
            f.write(f"**Question:** {row['question_clean']}\n\n")
            if row.get("options_clean", ""):
                f.write("**Options:**\n\n")
                f.write("```text\n")
                f.write(str(row["options_clean"]).strip() + "\n")
                f.write("```\n\n")
            f.write(f"**Gold:** {row.get('gold_answer', '')}\n\n")
            f.write("| System | Answer | Correct |\n")
            f.write("|---|---:|---:|\n")
            for s in ["s0", "s1", "s2", "s3_mc"]:
                f.write(f"| {s} | {row.get(f'{s}_answer', '')} | {row.get(f'{s}_correct', '')} |\n")
            f.write("\n")

            if case_type == "only_s3_correct":
                f.write("**Interpretation:** S3-MC is the only system that recovers the correct answer, suggesting that active retrieval/regeneration helps in this multihop case.\n\n")
            elif case_type == "s2_beats_s1_case":
                f.write("**Interpretation:** S2 outperforms fixed RAG here, suggesting that adaptive routing can avoid harmful or unnecessary retrieval.\n\n")
            elif case_type == "s1_beats_s2_case":
                f.write("**Interpretation:** Fixed RAG outperforms S2 here, suggesting that the adaptive policy may under-retrieve or rely too much on the base model.\n\n")

print("Guardado:")
print(OUT_MD)
print(OUT_CSV)
print()
print(selected[cols].to_string(index=False))

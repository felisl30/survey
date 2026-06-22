#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

IN = Path("outputs/eval_mc/model_grid_musique/model_grid_s0_s3_summary.csv")
OUT_DIR = Path("outputs/eval_mc/model_grid_musique/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

rows = []

for model, group in df.groupby("model"):
    base = group[group["system"] == "s0"].iloc[0]

    for _, row in group.iterrows():
        rows.append({
            "model": model,
            "system": row["system"],
            "accuracy": row["accuracy"],
            "delta_accuracy_vs_s0": row["accuracy"] - base["accuracy"],
            "avg_total_tokens": row["avg_total_tokens"],
            "delta_tokens_vs_s0": row["avg_total_tokens"] - base["avg_total_tokens"],
            "retrieval_rate": row["retrieval_rate"],
            "accuracy_per_1000_tokens": row["accuracy"] / row["avg_total_tokens"] * 1000,
            "avg_latency_seconds": row["avg_latency_seconds"],
        })

out = pd.DataFrame(rows)

out_path = OUT_DIR / "model_grid_deltas_vs_s0.csv"
out.to_csv(out_path, index=False)

pivot_acc = df.pivot(index="model", columns="system", values="accuracy")
pivot_tokens = df.pivot(index="model", columns="system", values="avg_total_tokens")
pivot_retrieval = df.pivot(index="model", columns="system", values="retrieval_rate")

report_path = OUT_DIR / "model_grid_deltas_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("Model grid delta analysis\n")
    f.write("=" * 80 + "\n\n")

    f.write("Accuracy pivot:\n")
    f.write(pivot_acc.to_string())
    f.write("\n\n")

    f.write("Average tokens pivot:\n")
    f.write(pivot_tokens.to_string())
    f.write("\n\n")

    f.write("Retrieval rate pivot:\n")
    f.write(pivot_retrieval.to_string())
    f.write("\n\n")

    f.write("Deltas vs S0:\n")
    f.write(out.to_string(index=False))
    f.write("\n\n")

    for model, group in out.groupby("model"):
        best_acc = group.sort_values("accuracy", ascending=False).iloc[0]
        best_eff = group.sort_values("accuracy_per_1000_tokens", ascending=False).iloc[0]

        f.write(f"Model: {model}\n")
        f.write(f"- Best accuracy: {best_acc['system']} acc={best_acc['accuracy']:.3f}, tokens={best_acc['avg_total_tokens']:.2f}\n")
        f.write(f"- Best accuracy/token: {best_eff['system']} acc={best_eff['accuracy']:.3f}, tokens={best_eff['avg_total_tokens']:.2f}, acc_per_1000_tokens={best_eff['accuracy_per_1000_tokens']:.3f}\n")
        f.write("\n")

print(out.to_string(index=False))
print()
print("Guardado:")
print(out_path)
print(report_path)

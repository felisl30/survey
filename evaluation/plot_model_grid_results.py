#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs/eval_mc/model_grid_musique")
OUT = BASE / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "model_grid_s0_s3_summary.csv")

order_models = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]
order_systems = ["s0", "s1", "s2", "s3_mc"]

df["model"] = pd.Categorical(df["model"], categories=order_models, ordered=True)
df["system"] = pd.Categorical(df["system"], categories=order_systems, ordered=True)
df = df.sort_values(["model", "system"])

for metric, ylabel, filename in [
    ("accuracy", "Accuracy", "accuracy_by_model_system.png"),
    ("avg_total_tokens", "Average total tokens", "tokens_by_model_system.png"),
    ("retrieval_rate", "Retrieval rate", "retrieval_rate_by_model_system.png"),
]:
    pivot = df.pivot(index="model", columns="system", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by model and system")
    ax.legend(title="System")
    plt.tight_layout()
    path = OUT / filename
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)

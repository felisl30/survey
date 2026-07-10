# MuSiQue-500 Model Grid Summary

| Model | System | n | Complete | Accuracy | Avg tokens | Acc/1k tok | Avg latency | Retrieval rate | Delta acc vs S0 | Token ratio vs S0 |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4.1-mini | S0 direct | 500 | yes | 0.404 | 350.33 | 1.153 | 1.28s |  | 0.000 | 1.00 |
| gpt-4.1-mini | S1 RAG top-5 | 500 | yes | 0.558 | 995.41 | 0.561 | 1.25s | 1.000 | 0.154 | 2.84 |
| gpt-4.1-mini | S2 adaptive RAG | 500 | yes | 0.468 | 529.33 | 0.884 | 1.35s | 0.404 | 0.064 | 1.51 |
| gpt-4.1-mini | S3 MC FLARE-like | 500 | yes | 0.622 | 1231.01 | 0.505 | 3.30s | 0.870 | 0.218 | 3.51 |
| gpt-5-mini | S0 direct | 500 | yes | 0.706 | 993.17 | 0.711 | 10.39s |  | 0.000 | 1.00 |
| gpt-5-mini | S1 RAG top-5 | 500 | yes | 0.782 | 1462.38 | 0.535 | 5.67s | 1.000 | 0.076 | 1.47 |
| gpt-5-mini | S2 adaptive RAG | 500 | yes | 0.754 | 1065.10 | 0.708 | 6.84s | 0.404 | 0.048 | 1.07 |
| gpt-5-mini | S3 MC FLARE-like | 500 | yes | 0.808 | 2295.71 | 0.352 | 15.18s | 0.750 | 0.102 | 2.31 |
| gpt-5-nano | S0 direct | 500 | yes | 0.554 | 1615.26 | 0.343 | 10.80s |  | 0.000 | 1.00 |
| gpt-5-nano | S1 RAG top-5 | 500 | yes | 0.716 | 1934.43 | 0.370 | 7.55s | 1.000 | 0.162 | 1.20 |
| gpt-5-nano | S2 adaptive RAG | 500 | yes | 0.630 | 1420.25 | 0.444 | 6.67s | 0.404 | 0.076 | 0.88 |
| gpt-5-nano | S3 MC FLARE-like | 500 | yes | 0.744 | 3996.22 | 0.186 | 21.32s | 0.994 | 0.190 | 2.47 |

## Reading Guide

- `Accuracy` compares answer correctness.
- `Avg tokens` is the average total token usage per question.
- `Acc/1k tok` is an efficiency metric: higher means more correct answers per token budget.
- `Delta acc vs S0` measures the gain from adding RAG/adaptive/FLARE over the direct baseline for the same model.
- `Token ratio vs S0` shows how much more/less expensive each system is relative to S0 for the same model.

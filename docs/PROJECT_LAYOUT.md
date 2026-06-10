# Project Layout

This repository is organized around three experimental systems.

## Canonical Folders

| Folder | Meaning |
| --- | --- |
| `data/s0` equivalent: `data/questions_s0.csv` | Normalized input for the direct baseline. S0 still keeps its small source files directly under `data/` for backward compatibility. |
| `data/s1/hotpotqa_mini/` | S1 input tables: `questions_s1.csv`, `corpus_s1.csv`, `qrels_s1.csv`, `selected_examples.jsonl`. |
| `data/s2/adaptive_rag/` | S2 input tables: `questions_s2.csv`, `corpus_s2.csv`, `qrels_s2.csv`, `summary_s2.json`. |
| `indexes/s1/hotpotqa_mini/` | S1 vector index artifacts: `chunks.csv`, `embeddings.npy`, `metadata.json`. |
| `indexes/s2/adaptive_rag/` | S2 vector index artifacts. |
| `outputs/s0/` | S0 raw, parsed, evaluated, and summary outputs. |
| `outputs/s1/generation/` | S1 model generations and parsed generations. |
| `outputs/s1/retrieval/` | S1 retrieval diagnostics. |
| `outputs/s1/evaluation/` | S1 answer evaluation outputs. |
| `outputs/s2/routing/` | S2 router outputs and summaries. |
| `outputs/s2/generation/` | S2 final generations. |
| `outputs/s2/evaluation/` | S2 evaluation outputs. |

`project_paths.py` is the source of truth for default paths used by scripts.
If a folder changes, update that file first and keep script defaults importing
from it.

## Model Code

| System | Scripts |
| --- | --- |
| S0 direct baseline | `prepare_s0_dataset.py`, `run_s0_direct.py`, `parse_s0_outputs.py`, `evaluate_s0.py` |
| S1 basic RAG | `s1_model_code/prepare_hotpotqa_mini.py`, `build_s1_index.py`, `evaluate_s1_retrieval.py`, `run_s1_rag.py`, `parse_s1_outputs.py`, `evaluate_s1_answers.py` |
| S2 Adaptive-RAG | `s2_model_code/prepare_s2_dataset.py`, `build_s2_index.py`, `router_s2.py`, `run_s2_adaptive_rag.py`, `parse_s2_outputs.py`, `evaluate_s2_routing.py`, `evaluate_s2_answers.py` |

## Recommended Smoke Run

These commands avoid paid model calls where possible and verify that artifacts
land in the canonical folders.

```powershell
python prepare_s0_dataset.py
python s1_model_code/prepare_hotpotqa_mini.py --per-topic 1
python s2_model_code/prepare_s2_dataset.py --n-direct-mmlu 1 --n-direct-truthfulqa 1 --n-retrieve 1 --n-abstain 1 --n-clarify 1
python s2_model_code/router_s2.py --strategy rules --limit 5
python s2_model_code/run_s2_adaptive_rag.py --router-strategy rules --limit 5 --dry-run --output-path outputs/s2/generation/adaptive_rag_s2_raw_smoke.csv
```

`run_s2_adaptive_rag.py --dry-run` still loads the S2 index if any routed row
uses `retrieve`; build it first with:

```powershell
python s2_model_code/build_s2_index.py --overwrite
```

## Legacy Paths

Older files under `data/hotpotqa_mini/` are kept only for compatibility with
previous runs. New S1 scripts default to `data/s1/hotpotqa_mini/`.

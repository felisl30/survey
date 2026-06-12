# Pipeline Basico S0-S4

This document is the minimal command map for showing that the common frame is
assembled and interpretable. It separates:

- data and indexes under `data/` and `indexes/`;
- generated artifacts under `outputs/<system>/`;
- model code under `s*_model_code/`;
- canonical paths in `project_paths.py`.

## Systems

| System | Role | Main input | Main outputs |
| --- | --- | --- | --- |
| S0 | Direct LLM baseline | `data/questions_s0.csv` | `outputs/s0/` |
| S1 | Basic RAG | `data/s1/hotpotqa_mini/` | `indexes/s1/`, `outputs/s1/` |
| S2 | Adaptive-RAG router/generator | `data/s2/adaptive_rag/questions_s2.csv` | `outputs/s2/routing/`, `outputs/s2/generation/`, `outputs/s2/evaluation/` |
| S3 | FLARE-like active retrieval | `data/s2/adaptive_rag/questions_s2.csv` | `outputs/s3/generation/`, `outputs/s3/evaluation/` |
| S4 | FIRE-like claim verification | parsed S2 or S3 outputs | `outputs/s4/generation/`, `outputs/s4/evaluation/`, `outputs/s4/verification/` |

## Demo sin llamadas nuevas al modelo

These commands use existing small raw outputs already in the repo and only run
parsing/evaluation/verification. They are useful to prove the frame works
without spending API tokens.

Full S1-S4 one-command demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frame_pipeline.ps1 -Mode demo -Tag frame_demo
```

This produces metric summaries for:

- S1 answer evaluation.
- S2 routing evaluation.
- S2 answer evaluation.
- S3 answer evaluation.
- S4 claim-level evaluation.
- S4 answer-level evaluation.
- S4 final sanity check.

Equivalent one-command PowerShell demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo_pipeline.ps1
```

```powershell
python s3_model_code/parse_s3_outputs.py `
  --input-path outputs/s3/generation/flare_like_s3_raw_test_5_v4.csv `
  --output-path outputs/s3/generation/flare_like_s3_parsed_demo.csv

python s3_model_code/evaluate_s3_answers.py `
  --input-path outputs/s3/generation/flare_like_s3_parsed_demo.csv `
  --output-path outputs/s3/evaluation/flare_like_s3_answer_results_demo.csv `
  --summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_demo.json `
  --group-summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_by_group_demo.csv

python s4_model_code/parse_s4_outputs.py `
  --input-path outputs/s4/generation/fire_like_s4_raw_retrieve_test_5_hybrid_claims_llm_verify.csv `
  --output-path outputs/s4/generation/fire_like_s4_parsed_demo.csv

python s4_model_code/evaluate_s4_claims.py `
  --input-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --claim-output-path outputs/s4/evaluation/fire_like_s4_claim_results_demo.csv `
  --summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_demo.json `
  --group-summary-path outputs/s4/evaluation/fire_like_s4_claim_summary_by_group_demo.csv

python s4_model_code/evaluate_s4_answers.py `
  --input-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --output-path outputs/s4/evaluation/fire_like_s4_answer_results_demo.csv `
  --summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_demo.json `
  --group-summary-path outputs/s4/evaluation/fire_like_s4_answer_summary_by_group_demo.csv

python s4_model_code/verify_s4_results.py `
  --s4-parsed-path outputs/s4/generation/fire_like_s4_parsed_demo.csv `
  --claim-results-path outputs/s4/evaluation/fire_like_s4_claim_results_demo.csv `
  --answer-results-path outputs/s4/evaluation/fire_like_s4_answer_results_demo.csv `
  --output-dir outputs/s4/verification `
  --prefix demo
```

## Demo con generacion nueva

These commands do call the model if `--dry-run` is not used and require
`OPENAI_API_KEY`.

Full S1-S4 dry run, generating new raw outputs without model calls:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frame_pipeline.ps1 -Mode dry-run -Limit 5 -Tag dry_5
```

Full S1-S4 real run, generating new raw outputs and metrics:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frame_pipeline.ps1 -Mode real -Limit 5 -Tag real_5
```

If inputs or indexes need to be regenerated first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_frame_pipeline.ps1 -Mode real -Limit 5 -Tag real_5 -PrepareData -BuildIndexes
```

```powershell
python prepare_s0_dataset.py
python s1_model_code/prepare_hotpotqa_mini.py --per-topic 1
python s2_model_code/prepare_s2_dataset.py --n-direct-mmlu 1 --n-direct-truthfulqa 1 --n-retrieve 1 --n-abstain 1 --n-clarify 1
python s2_model_code/build_s2_index.py --overwrite

python s3_model_code/run_s3_flare_like.py `
  --limit 5 `
  --max-steps 3 `
  --top-k-per-step 2 `
  --max-retrieval-steps 2 `
  --max-total-chunks 4 `
  --retrieval-strategy rules `
  --output-path outputs/s3/generation/flare_like_s3_raw_demo.csv

python s3_model_code/parse_s3_outputs.py `
  --input-path outputs/s3/generation/flare_like_s3_raw_demo.csv `
  --output-path outputs/s3/generation/flare_like_s3_parsed_demo.csv

python s3_model_code/evaluate_s3_answers.py `
  --input-path outputs/s3/generation/flare_like_s3_parsed_demo.csv `
  --output-path outputs/s3/evaluation/flare_like_s3_answer_results_demo.csv `
  --summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_demo.json `
  --group-summary-path outputs/s3/evaluation/flare_like_s3_answer_summary_by_group_demo.csv
```

S4 can then consume a parsed S2/S3 output. For real S4 experiments, prefer
retrieve cases and use `claim-strategy hybrid` plus `verification-strategy llm`
when API calls are available.

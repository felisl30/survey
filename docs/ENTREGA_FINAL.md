# Entrega final

Este archivo enumera los artefactos que respaldan el informe final. Si hay dudas entre archivos parecidos, usar esta lista como fuente canonica.

## Informe

- Overleaf: proyecto `6a4a9ced0273a6945654292b`.
- PDF final en el repo: `docs/informe/informe_final.pdf`.
- Fuente editable: Overleaf. No se versiona la copia local de `main.tex` ni archivos auxiliares de compilacion.

## Archivos de soporte en raiz

Estos archivos se conservan en la raiz porque son configuracion o utilidades compartidas, no resultados experimentales:

| Archivo | Uso |
| --- | --- |
| `.env.example` | Plantilla para `OPENAI_API_KEY` y variables locales. |
| `.gitignore` | Mantiene fuera de Git caches, zips, copias locales y temporales. |
| `README.md` | Guia general del repositorio. Este archivo, `docs/ENTREGA_FINAL.md`, es el inventario canonico. |
| `direct_llm.py` | Cliente comun para llamadas directas al modelo. |
| `project_paths.py` | Constantes de rutas usadas por scripts finales y auxiliares. |
| `requirements.txt` | Dependencias de ejecucion y analisis. |

## Datasets finales

| Bloque | Archivos |
| --- | --- |
| MuSiQue-MC-500 | `data/eval_mc/musique_mc_rag_500/questions.csv`, `data/eval_mc/musique_mc_rag_500/corpus.csv`, `data/eval_mc/musique_mc_rag_500/qrels.csv`, `data/eval_mc/musique_mc_rag_500/build_summary.json` |
| HotpotQA-MC-500 | `data/eval_mc/hotpotqa_mc_rag_500/questions.csv`, `data/eval_mc/hotpotqa_mc_rag_500/corpus.csv`, `data/eval_mc/hotpotqa_mc_rag_500/qrels.csv`, `data/eval_mc/hotpotqa_mc_rag_500/build_summary.json` |
| 2Wiki-MC-500 | `data/eval_mc/2wiki_mc_rag_500/questions.csv`, `data/eval_mc/2wiki_mc_rag_500/corpus.csv`, `data/eval_mc/2wiki_mc_rag_500/qrels.csv`, `data/eval_mc/2wiki_mc_rag_500/build_summary.json` |
| Robustez MuSiQue | `data/eval_mc/robustness_musique/questions.csv`, `data/eval_mc/robustness_musique/qrels.csv`, `data/eval_mc/robustness_musique/corpus_clean.csv`, `data/eval_mc/robustness_musique/corpus_noisy.csv`, `data/eval_mc/robustness_musique/corpus_adversarial.csv` |
| Saturacion MC-100 | `data/eval_mc/questions_*_mc_100.csv` y `data/eval_mc/build_summary_*_mc_100.json` |

## Resultados finales

`docs/experimentos/` conserva solo las notas interpretativas usadas para S0-S3 y S4. S5 no tiene una nota manual versionada en esa carpeta: sus resultados finales son los reportes y tablas generados en `outputs/eval_mc/meta_router_gpt_5_mini/` y `outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.csv`.

### Saturacion inicial

- `outputs/eval_mc/s0_benchmark_model_grid_summary.csv`

### S0-S3 MuSiQue

- `outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.csv`
- `outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.md`

### Multi-dataset MC-500

- `outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.md`
- `outputs/eval_mc/cross_dataset/mc500_pivot_accuracy_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_pivot_tokens_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_best_by_dataset_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_comparison_final_notes.md`

### Robustez

- `outputs/eval_mc/robustness_musique/gpt_5_mini/analysis/robustness_s0_s3_summary.csv`
- `outputs/eval_mc/robustness_musique/gpt_5_mini/analysis/robustness_qualitative_examples.md`

### S4 auditor factual

- `docs/experimentos/informe_s4_focus_musique_500.md`
- `outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_raw_gpt_5_mini_limit25_rules.csv`
- `logs/musique_500_s4_focus_gpt_5_mini_limit25_rules.log`
- `logs/musique_500_s4_focus_20260705_145849.log`

### S5 meta-router

- `outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.csv`
- `outputs/eval_mc/meta_router_gpt_5_mini/oracle_router_report.md`
- `outputs/eval_mc/meta_router_gpt_5_mini/s5_final_router_report.md`
- `outputs/eval_mc/meta_router_gpt_5_mini/s5_rule_based_report.md`
- `outputs/eval_mc/meta_router_gpt_5_mini/meta_router_question_table_report.md`

## Logs finales

- `logs/model_grid_s0_s3_full.log`
- `logs/musique_500_full_20260705_002928.log`
- `logs/musique_500_s4_focus_gpt_5_mini_limit25_rules.log`
- `logs/musique_500_s4_focus_20260705_145849.log`

## Scripts a conservar

La organizacion por sistema esta resumida en `modelos/README.md`.

| Proposito | Scripts |
| --- | --- |
| Construir datasets MC-500 | `evaluation/build_mc_eval_dataset.py`, `evaluation/build_mc_rag_dataset.py`, `evaluation/validate_freeze_mc_benchmark.py`, `evaluation/inspect_mc_rag_readiness.py`, `scripts/step2_build_hotpotqa_mc500.sh`, `scripts/step6_build_2wiki_mc500.sh` |
| Construir indices RAG | `evaluation/build_mc_rag_index.py`, `scripts/step3_build_hotpotqa_rag_index.sh`, `scripts/step7_build_2wiki_rag_index.sh` |
| Correr S0-S3 | `modelos/s0/run_s0_direct.py`, `modelos/s1/run_s1_mc_rag.py`, `modelos/s2/run_s2_mc_real_adaptive.py`, `modelos/s3/run_s3_mc_flare_like.py`, `evaluation/run_s0_mc_pipeline.ps1`, `evaluation/run_musique_500_model_grid.ps1`, `evaluation/evaluate_mc_accuracy.py` |
| Agregar MuSiQue y multi-dataset | `evaluation/summarize_musique_500_model_grid.py`, `scripts/step10_aggregate_mc500_comparison.sh`, `scripts/step11_run_musique_s0_and_final_aggregate.sh` |
| Robustez | `evaluation/build_mc_robustness_dataset.py`, `evaluation/summarize_musique_robustness_s0_s3.py`, `evaluation/analyze_musique_robustness_deep.py`, `evaluation/export_musique_robustness_qualitative_examples.py`, `scripts/run_musique_robustness_s0_s3.ps1` |
| S4 | `modelos/s4/`, `evaluation/build_musique_500_s4_disagreement_grid.py`, `evaluation/build_s4_robustness_focus_input.py` |
| S5 | `modelos/s5/meta_router/`, `evaluation/build_musique_500_s5_router.py` |

## Fuera de entrega

- Exportaciones locales comprimidas: `docs/*.zip`.
- Copias locales de Overleaf: `docs/overleaf_musique_rag/`.
- Material local del poster: `docs/ia_fest_poster/`.
- Handoffs y notas intermedias no citadas: `docs/handoff/` y notas viejas de `docs/experimentos/`.
- Carpetas temporales: `tmp_s4_extract/`.
- Smokes y pruebas chicas que no aparecen en el informe.
- Versiones viejas del informe/poster que no coinciden con Overleaf.

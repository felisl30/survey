# Plan: Pipeline unificado de evaluación de RAGs (S1/S2/S3)

## Contexto

El repositorio tiene 4 sistemas RAG implementados (S0 directo, S1 RAG clásico, S2 Adaptive-RAG, S3 FLARE-like) pero hasta ahora cada uno se evaluó con un dataset distinto y con N muy chico (5–20 preguntas). Eso impide comparar de manera justa cuál funciona mejor y en qué contexto.

El objetivo del trabajo es **comparar S1/S2/S3** (no S0) frente al **mismo set de preguntas** en términos de: calidad de respuesta, tiempo de ejecución, cantidad de documentos recuperados, decisiones de routing, costo en tokens. S0 queda **fuera** del pipeline de comparación. S4 (FIRE) también queda fuera (no implementado y se aparta del concepto RAG puro).

Punto clave del diseño: el set debe incluir una **minoría de preguntas que el LLM base podría responder sin RAG**, para que S2 (Adaptive-RAG) tenga oportunidad de demostrar su capacidad de routing eligiendo "direct" en esos casos. La mayoría de las preguntas sigue requiriendo retrieval, pero queremos ver al router decidir bien sobre el subconjunto "fácil".

Decisiones cerradas:
- **Sistemas comparados**: S1, S2, S3 (S0 fuera).
- **Tamaño**: 100 preguntas (80 retrieve + 20 direct).
- **Fuente "retrieve"**: HotpotQA-mini (`data/s1/hotpotqa_mini/`).
- **Fuente "direct"**: open-QA de TruthfulQA filtrado desde `data/questions_s0.csv`.
- **Tipo**: solo open-QA.
- **Output**: XLSX único con una hoja por sistema + hoja `resumen` + hoja `metadata`.
- **Modelos en este round**: el default. Pipeline preparado para multi-modelo después.

## Arquitectura

Tres scripts nuevos en la raíz, reusando los runners existentes (sin tocarlos):

```
build_eval_dataset.py        # Fase 1: arma el set de 100 (80 retrieve + 20 direct)
run_evaluation_pipeline.py   # Fase 2: corre S1/S2/S3 sobre el set
export_eval_results.py       # Fase 3: une todo en un XLSX
```

Datos:
```
data/eval/
  questions_eval.csv
  corpus_eval.csv
  qrels_eval.csv
  build_summary.json
```

Outputs:
```
outputs/eval/
  results_<model>_s1.csv
  results_<model>_s2.csv
  results_<model>_s3.csv
  evaluation_<model>.xlsx
```

## Fase 1 — Builder

`build_eval_dataset.py` arma `questions_eval.csv`:

- **80 retrieve**: sample balanceado por `hotpot_type` (40 bridge + 40 comparison) desde `questions_s1.csv`, priorizando preguntas con evidencia completa en el corpus de 200 docs. `expected_route = retrieve`.
- **20 direct**: sample de TruthfulQA open-QA desde `questions_s0.csv` (filas con `dataset == 'truthfulqa'` y `case_type == 'open_qa'`, `expected_behavior == 'answer'`). Diversidad por `truthfulqa_category`. `expected_route = direct`.
- Corpus: copia de `data/s2/adaptive_rag/corpus_s2.csv` (200 docs).
- qrels: subset de `data/s1/hotpotqa_mini/qrels_s1.csv` filtrado a las 80 ids retrieve.

No usamos S0 como filtro — a propósito dejamos pasar preguntas "fáciles" para estresar el router de S2.

## Fase 2 — Runner

`run_evaluation_pipeline.py` invoca por subprocess los runners + parsers + evaluators de cada sistema. No reimplementa lógica. Tiempos ya están en los CSVs. Default `--systems s1,s2,s3`. Soporta `--limit`, `--resume`, `--force`.

## Fase 3 — Exporter

`export_eval_results.py` une los 3 CSVs evaluados en un XLSX con `openpyxl`:

- Una hoja por sistema con schema unificado: `question_id`, `question`, `gold_answer`, `expected_route`, `source_dataset`, `model_answer`, `is_correct`, `token_f1`, `contains_gold`, `latency_total_s`, `latency_retrieval_s`, `latency_generation_s`, `latency_router_s` (S2), `n_docs_retrieved`, `retrieved_chunk_ids`, `retrieval_recall`, `retrieval_precision`, `predicted_route` (S2), `route_match_expected` (S2), `n_generation_steps` (S3), `n_retrieval_steps` (S3), `tokens_in/out/total`, `confidence`, `error`.
- Hoja `resumen` con métricas por sistema y por tipo (retrieve/direct). Para S2 incluye `routing_accuracy_global`, `routing_accuracy_on_retrieve`, `routing_accuracy_on_direct`.
- Hoja `metadata` con modelo, fecha, commit hash, N preguntas, paths.

Multi-modelo (futuro): `--models gpt-4o-mini gpt-3.5-turbo ...` → `comparative_all_models.xlsx`.

## Verificación

1. Smoke builder con 8+2.
2. Smoke runner con S1+S2 sobre el dryrun.
3. Smoke exporter, abrir XLSX, validar hojas y columnas.
4. Run completo: builder 80+20, runner s1,s2,s3, exporter.
5. Sanity manual sobre 5 filas por sistema.

## Riesgos

- Las 20 direct podrían no diferenciar a S2 → reportamos por subgrupo.
- HotpotQA-mini memorizable → reportamos `retrieval_recall` además de accuracy.
- S3 lento → `--limit` + `--resume`.
- Costo API: ~300 calls + overheads. <$5 USD por modelo.

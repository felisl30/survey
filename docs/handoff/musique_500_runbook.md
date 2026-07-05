# MuSiQue MC 500 - runbook para corrida nocturna

Este runbook tiene dos caminos:

1. **MuSiQue-500 principal**: correr S0/S1/S2/S3 sobre el nuevo dataset de 500 preguntas y despues generar S5 + grilla S4 post-hoc.
2. **Pipeline de Ion**: correr robustez clean/noisy/adversarial y despues S4/S5 sobre ese setup.

Siempre correr desde la raiz del repo:

```powershell
cd C:\Users\Usuario\Downloads\NLPSURVEY
```

## 1. Archivos clave

Dataset MC auditado:

```text
data/eval_mc/questions_musique_mc_500.csv
data/eval_mc/build_summary_musique_mc_500.json
```

Dataset RAG 500 ya construido:

```text
data/eval_mc/musique_mc_rag_500/questions.csv
data/eval_mc/musique_mc_rag_500/corpus.csv
data/eval_mc/musique_mc_rag_500/qrels.csv
```

Indice RAG 500 ya construido:

```text
indexes/eval_mc/musique_mc_rag_500/chunks.csv
indexes/eval_mc/musique_mc_rag_500/embeddings.npy
indexes/eval_mc/musique_mc_rag_500/metadata.json
```

Smoke outputs:

```text
outputs/eval_mc/musique_mc_rag_500/smoke20/
```

## 2. Chequeo ya realizado

MuSiQue-500 RAG:

- 500 preguntas
- 6000 documentos/chunks de evidencia
- 6000 qrels
- 0 preguntas sin contexto
- indice con 6000 embeddings, dimension 384

Smoke `gpt-5-mini`, limite 20:

| Sistema | n | Accuracy | Valid format | Run errors | Avg tokens | Avg latency |
|---|---:|---:|---:|---:|---:|---:|
| S0 directo | 20 | 0.500 | 1.000 | 0.000 | n/a | n/a |
| S1 RAG top-5 | 20 | 0.600 | 1.000 | 0.000 | 1560.10 | 5.105s |
| S2 adaptive | 20 | 0.650 | 1.000 | 0.000 | 1419.70 | 6.844s |
| S3 MC FLARE-like | 20 | 0.750 | 1.000 | 0.000 | 2560.25 | 17.224s |

Conclusion tecnica: **S0/S1/S2/S3 corren bien sobre MuSiQue-500**. El riesgo principal no es de formato, sino de tiempo/costo, especialmente S3.

## 3. Rehacer RAG/index si hiciera falta

No hace falta repetir esto si las carpetas existen, pero los comandos son:

```powershell
python evaluation\build_mc_rag_dataset.py --input-path data\eval_mc\questions_musique_mc_500.csv --output-dir data\eval_mc\musique_mc_rag_500 --benchmark-name musique_mc_500 --expected-n 500
```

```powershell
python evaluation\build_mc_rag_index.py --corpus-path data\eval_mc\musique_mc_rag_500\corpus.csv --output-dir indexes\eval_mc\musique_mc_rag_500
```

## 4. Comando unico recomendado para la grilla final

Este es el comando principal para el informe. Corre varios modelos contra varios sistemas y al final genera una tabla comparativa con:

- accuracy
- tokens promedio
- latencia promedio
- accuracy por 1000 tokens
- delta de accuracy contra S0
- ratio de tokens contra S0
- tasa de retrieval cuando aplica
- S5 post-hoc: routers que eligen entre S0/S1/S2/S3
- grilla S4: casos donde los sistemas estan en desacuerdo o donde S3 corrige/empeora

Comando recomendado, secuencial y con `resume`:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini" -Systems "s0,s1,s2,s3"
```

Si se quiere incluir tambien el modelo grande usado por Ion:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini,gpt-5.4" -Systems "s0,s1,s2,s3"
```

Para smoke barato antes de una corrida larga:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-mini" -Systems "s0,s1,s2,s3" -Limit 20
```

Outputs finales de la grilla:

```text
outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.csv
outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.md
outputs/eval_mc/musique_mc_rag_500/model_grid_summary/missing_runs.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.md
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_disagreement_grid.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_candidates.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_input.csv
```

`model_grid_metrics.csv` es el archivo principal para el informe.
`s5_policy_summary.csv` sirve para comparar routers.
`s4_disagreement_grid.csv` sirve para auditar en que preguntas difieren los sistemas.

Si se quiere correr S4 FIRE-like sobre los casos focalizados, agregar el switch:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-mini" -Systems "s0,s1,s2,s3" -RunS4Focus -S4FocusLimit 25
```

Esto usa `outputs\eval_mc\musique_mc_rag_500\posthoc\s4_focus_input.csv` y guarda:

```text
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_raw.csv
```

Para no gastar tokens, no usar `-RunS4Focus` en el primer intento largo. Primero mirar la grilla.

## 5. Corrida S0 sobre MuSiQue-500

Para un solo modelo:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_s0_mc_pipeline.ps1 -Mode real -Model "gpt-5-mini" -QuestionPath "data\eval_mc\questions_musique_mc_500.csv" -OutputPrefix "s0_gpt_5_mini_musique_500" -Resume
```

Para correr varios modelos S0:

```powershell
$models = @("gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini")
foreach ($m in $models) {
  $tag = $m -replace "[\.-]", "_"
  powershell -ExecutionPolicy Bypass -File .\evaluation\run_s0_mc_pipeline.ps1 `
    -Mode real `
    -Model $m `
    -QuestionPath "data\eval_mc\questions_musique_mc_500.csv" `
    -OutputPrefix "s0_${tag}_musique_500" `
    -Resume
}
```

Si se quiere probar tambien el modelo grande usado por Ion:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_s0_mc_pipeline.ps1 -Mode real -Model "gpt-5.4" -QuestionPath "data\eval_mc\questions_musique_mc_500.csv" -OutputPrefix "s0_gpt_5_4_musique_500" -Resume
```

## 6. Corrida S1/S2/S3 sobre MuSiQue-500

Usar una carpeta por modelo para no pisar resultados:

```powershell
$model = "gpt-5-mini"
$tag = $model -replace "[\.-]", "_"
$out = "outputs\eval_mc\musique_mc_rag_500\$tag"
New-Item -ItemType Directory -Force -Path $out | Out-Null
```

S1:

```powershell
python evaluation\run_s1_mc_rag.py --questions-path data\eval_mc\musique_mc_rag_500\questions.csv --index-dir indexes\eval_mc\musique_mc_rag_500 --output-path "$out\s1_raw.csv" --model $model --top-k 5 --resume
```

S2:

```powershell
python evaluation\run_s2_mc_real_adaptive.py --questions-path data\eval_mc\musique_mc_rag_500\questions.csv --index-dir indexes\eval_mc\musique_mc_rag_500 --output-path "$out\s2_raw.csv" --model $model --top-k 5 --threshold 0.45 --min-gap 0.05 --resume
```

S3:

```powershell
python evaluation\run_s3_mc_flare_like.py --questions-path data\eval_mc\musique_mc_rag_500\questions.csv --index-dir indexes\eval_mc\musique_mc_rag_500 --output-path "$out\s3_raw.csv" --model $model --top-k 5 --resume
```

## 7. Parseo/evaluacion S1/S2/S3

S1:

```powershell
python parse_s0_outputs.py --input-path "$out\s1_raw.csv" --output-path "$out\s1_parsed.csv"
python evaluate_s0.py --input-path "$out\s1_parsed.csv" --output-path "$out\s1_evaluated.csv" --summary-path "$out\s1_summary.json" --group-summary-path "$out\s1_group_summary.csv"
```

S2:

```powershell
python parse_s0_outputs.py --input-path "$out\s2_raw.csv" --output-path "$out\s2_parsed.csv"
python evaluate_s0.py --input-path "$out\s2_parsed.csv" --output-path "$out\s2_evaluated.csv" --summary-path "$out\s2_summary.json" --group-summary-path "$out\s2_group_summary.csv"
```

S3:

```powershell
python parse_s0_outputs.py --input-path "$out\s3_raw.csv" --output-path "$out\s3_parsed.csv"
python evaluate_s0.py --input-path "$out\s3_parsed.csv" --output-path "$out\s3_evaluated.csv" --summary-path "$out\s3_summary.json" --group-summary-path "$out\s3_group_summary.csv"
```

## 8. Orden recomendado para correr de noche

Para `gpt-5-mini`:

1. S0
2. S1
3. S2
4. S3
5. Parseo/evaluacion

No arrancaria todos en paralelo. Mejor secuencial, porque S3 consume bastante.

Estimacion desde smoke 20:

- S0 full 500: probablemente 60-90 min.
- S1 full 500: probablemente 40-60 min.
- S2 full 500: probablemente 60-90 min.
- S3 full 500: varias horas.

Los tiempos reales dependen de latencia de API y de cuantas veces S3 active retrieval.

## 9. Pipeline de Ion: robustness clean/noisy/adversarial

Este pipeline usa:

```text
data/eval_mc/robustness_musique/questions.csv
data/eval_mc/robustness_musique/corpus_clean.csv
data/eval_mc/robustness_musique/corpus_noisy.csv
data/eval_mc/robustness_musique/corpus_adversarial.csv
```

Ya deje construidos los indices:

```text
indexes/eval_mc/robustness_musique_clean/
indexes/eval_mc/robustness_musique_noisy/
indexes/eval_mc/robustness_musique_adversarial/
```

Si hubiera que reconstruirlos:

```powershell
python evaluation\build_mc_rag_index.py --corpus-path data\eval_mc\robustness_musique\corpus_clean.csv --output-dir indexes\eval_mc\robustness_musique_clean --force
python evaluation\build_mc_rag_index.py --corpus-path data\eval_mc\robustness_musique\corpus_noisy.csv --output-dir indexes\eval_mc\robustness_musique_noisy --force
python evaluation\build_mc_rag_index.py --corpus-path data\eval_mc\robustness_musique\corpus_adversarial.csv --output-dir indexes\eval_mc\robustness_musique_adversarial --force
```

El script de Ion es Bash, no PowerShell. Correrlo desde Git Bash o WSL:

```bash
MODEL="gpt-5.4-mini" bash scripts/run_musique_robustness_s0_s3.sh
```

Para smoke corto:

```bash
LIMIT=20 MODEL="gpt-5.4-mini" bash scripts/run_musique_robustness_s0_s3.sh
```

Outputs esperados:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
```

Despues resumir:

```powershell
python evaluation\summarize_musique_robustness_s0_s3.py --base-dir outputs\eval_mc\robustness_musique\gpt_5_4_mini
python evaluation\analyze_musique_robustness_deep.py
```

## 10. Pipeline de Ion: S4

S4 depende de que ya exista la corrida robustness S0-S3 anterior, especialmente los outputs de S3-MC evaluados.

Primero crear input focalizado:

```powershell
python evaluation\build_s4_robustness_focus_input.py --base-dir outputs\eval_mc\robustness_musique\gpt_5_4_mini --questions-path data\eval_mc\robustness_musique\questions.csv --preset core5 --output-path outputs\eval_mc\robustness_musique\gpt_5_4_mini\s4\input\s4_robustness_focus_core5.csv
```

Despues correr el script de Ion desde Git Bash o WSL:

```bash
bash scripts/run_s4_robustness_focus_rules.sh
```

Outputs esperados:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/
```

## 11. Pipeline de Ion: S5 meta-router

S5 depende de tener la tabla consolidada de S0/S1/S2/S3 de robustness. No es un modelo generativo nuevo: es un router que elige entre sistemas.

Orden:

```powershell
python evaluation\meta_router\build_meta_router_table.py
python evaluation\meta_router\analyze_oracle_router.py
python evaluation\meta_router\run_s5_rule_based_router.py
python evaluation\meta_router\run_s5_final_router.py
```

Outputs esperados:

```text
outputs/eval_mc/meta_router/
```

## 12. MuSiQue-500: S5 y grilla S4 post-hoc

Si ya existen las corridas S0/S1/S2/S3 y solo se quiere regenerar S5 + S4 sin volver a llamar modelos:

```powershell
python evaluation\build_musique_500_s5_router.py --models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini"
python evaluation\build_musique_500_s4_disagreement_grid.py --models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini" --s4-per-model 25
```

S5 no consume API: compara respuestas ya generadas y calcula politicas:

- `oracle_min_cost`: techo teorico, elige el sistema correcto mas barato si existe.
- `agreement_min_cost`: elige la opcion donde mas sistemas coinciden y desempata por costo.
- `confidence_then_cost`: elige mayor confianza reportada y desempata por costo.
- `risk_aware_s5`: router simple para probar si conviene confiar en acuerdos baratos antes de ir a S3.

La grilla S4 separa casos como:

- `s3_only_correct`: S3 arregla algo que los otros fallan.
- `s3_wrong_others_correct`: S3 empeora frente a sistemas mas baratos.
- `mixed_correctness_disagreement`: algunos aciertan y otros no.
- `all_wrong_disagreement`: todos fallan, pero por caminos distintos.

## 13. Mi chequeo de riesgo

Lo que pienso que va a correr bien:

- S0/S1/S2/S3 sobre MuSiQue-500: probado con limit 20.
- RAG/index MuSiQue-500: construido y consistente.
- S5 y grilla S4 post-hoc sobre MuSiQue-500: no deberian consumir API y dependen solo de que existan los CSV evaluados.
- Indices robustness de Ion: construidos para clean/noisy/adversarial.
- S5 de Ion: deberia correr bien una vez existan los outputs robustness S0-S3, porque sus scripts son de analisis sobre CSVs ya generados.

Lo que puede fallar o tardar:

- S3 full 500: no por formato, sino por costo/tiempo.
- Robustness full de Ion: es muchas llamadas, porque corre S0 y S1/S2/S3 en tres condiciones.
- S4: si se corre con `-RunS4Focus`, consume llamadas y conviene limitarlo a 25/50 casos al principio.
- Scripts `.sh`: en Windows necesitan Git Bash o WSL.

Recomendacion concreta para Ion:

1. Primero correr MuSiQue-500 S0/S1/S2/S3 con `gpt-5-mini`.
2. Si eso queda bien, correr otros tamanos/modelos.
3. Mirar `s5_policy_summary.csv` y `s4_disagreement_grid.csv` antes de gastar en S4.
4. Si la grilla muestra casos interesantes, correr `-RunS4Focus` con limite 25 o 50.

# Para Ion - corridas finales

Desde la raiz del repo:

```powershell
cd C:\Users\Usuario\Downloads\NLPSURVEY
```

## 1. Corrida principal MuSiQue-500

Este comando corre S0/S1/S2/S3 para varios tamanos de modelo y al final genera tambien S5 y la grilla de S4:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini" -Systems "s0,s1,s2,s3"
```

Si antes queres hacer un smoke barato:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-mini" -Systems "s0,s1,s2,s3" -Limit 20
```

## Que outputs mirar

Tabla principal para el informe:

```text
outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.csv
outputs/eval_mc/musique_mc_rag_500/model_grid_summary/model_grid_metrics.md
```

S5, o sea routers post-hoc que eligen entre S0/S1/S2/S3:

```text
outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s5_policy_summary.md
```

Grilla para S4, con casos donde los sistemas desacuerdan:

```text
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_disagreement_grid.csv
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_input.csv
```

## S4 opcional

Primero miraria la grilla. Si hay casos interesantes, correr S4 solo sobre un foco chico:

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluation\run_musique_500_model_grid.ps1 -Models "gpt-5-mini" -Systems "s0,s1,s2,s3" -RunS4Focus -S4FocusLimit 25
```

Eso guarda:

```text
outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_raw.csv
```

No usaria `-RunS4Focus` en la primera corrida larga porque consume tokens. Primero conviene tener S0-S3 + S5 + grilla S4.

## 2. Robustez frente al ruido

Este es el experimento clean/noisy/adversarial: sirve para ver que pasa cuando el RAG recibe contexto limpio, contexto con ruido o distractores adversariales.

Este script es Bash, correr desde Git Bash o WSL:

```bash
MODEL="gpt-5.4-mini" bash scripts/run_musique_robustness_s0_s3.sh
```

Smoke barato:

```bash
LIMIT=20 MODEL="gpt-5.4-mini" bash scripts/run_musique_robustness_s0_s3.sh
```

Despues resumir desde PowerShell:

```powershell
python evaluation\summarize_musique_robustness_s0_s3.py --base-dir outputs\eval_mc\robustness_musique\gpt_5_4_mini
python evaluation\analyze_musique_robustness_deep.py
```

Outputs a mirar:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/summary.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/deep_analysis/
```

## 3. S4 sobre robustez

Si ya corrio robustness y queremos auditar casos dificiles con FIRE-like:

```powershell
python evaluation\build_s4_robustness_focus_input.py --base-dir outputs\eval_mc\robustness_musique\gpt_5_4_mini --questions-path data\eval_mc\robustness_musique\questions.csv --preset core5 --output-path outputs\eval_mc\robustness_musique\gpt_5_4_mini\s4\input\s4_robustness_focus_core5.csv
```

Y despues desde Git Bash o WSL:

```bash
bash scripts/run_s4_robustness_focus_rules.sh
```

## Orden recomendado

1. Primero correr MuSiQue-500 principal.
2. Despues correr robustness clean/noisy/adversarial.
3. Mirar summaries.
4. Recien ahi decidir si correr S4 focalizado, para no gastar tokens de mas.

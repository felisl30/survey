# Para Ion - MuSiQue-500 final

Desde la raiz del repo:

```powershell
cd C:\Users\Usuario\Downloads\NLPSURVEY
```

## Corrida principal

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

# S5 Final Router Report

## Objetivo

Este reporte consolida una política S5 final a partir de las mejores variantes rule-based por condición.

## Política final

```text
clean       -> s5_majority_min_cost
noisy       -> s5_risk_aware
adversarial -> s5_majority_min_cost
```

La motivación es usar, en cada condición, la política que logró mejor combinación de accuracy y costo en el reporte S5 previo.

## Archivos generados

- `outputs\eval_mc\meta_router_gpt_5_mini\s5_final_router_predictions.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_final_router_summary.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_final_router_by_condition.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_final_router_decision_distribution.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_final_router_report.md`

## Resumen global

| policy                       | kind     |   n |   accuracy |   avg_tokens |   accuracy_per_1000_tokens |   gap_vs_oracle_accuracy |   delta_accuracy_vs_s1 |   token_savings_vs_s1 |   relative_token_savings_vs_s1 |
|:-----------------------------|:---------|----:|-----------:|-------------:|---------------------------:|-------------------------:|-----------------------:|----------------------:|-------------------------------:|
| s0                           | baseline | 300 |     0.7200 |     992.8700 |                     0.7252 |                   0.2000 |                -0.0700 |              361.0133 |                         0.2667 |
| s1                           | baseline | 300 |     0.7900 |    1353.8833 |                     0.5835 |                   0.1300 |                 0.0000 |                0.0000 |                         0.0000 |
| s2                           | baseline | 300 |     0.7667 |    1025.8033 |                     0.7474 |                   0.1533 |                -0.0233 |              328.0800 |                         0.2423 |
| s3_mc                        | baseline | 300 |     0.8200 |    2237.4933 |                     0.3665 |                   0.1000 |                 0.0300 |             -883.6100 |                        -0.6526 |
| oracle_min_cost              | oracle   | 300 |     0.9200 |     950.5797 |                     0.9678 |                   0.0000 |                 0.1300 |              403.3036 |                         0.2979 |
| s5_final_condition_mix       | s5_final | 300 |     0.8033 |     889.8900 |                     0.9027 |                   0.1167 |                 0.0133 |              463.9933 |                         0.3427 |
| s5_conservative_noisy_router | s5_final | 300 |     0.8100 |    1198.0500 |                     0.6761 |                   0.1100 |                 0.0200 |              155.8333 |                         0.1151 |

## Resumen por condición

| condition   | policy                       | kind     |   n |   accuracy |   avg_tokens |   accuracy_per_1000_tokens |
|:------------|:-----------------------------|:---------|----:|-----------:|-------------:|---------------------------:|
| clean       | s0                           | baseline | 100 |     0.7200 |     992.8700 |                     0.7252 |
| clean       | s1                           | baseline | 100 |     0.8100 |    1359.2800 |                     0.5959 |
| clean       | s2                           | baseline | 100 |     0.8100 |    1098.3600 |                     0.7375 |
| clean       | s3_mc                        | baseline | 100 |     0.8700 |    2314.7700 |                     0.3758 |
| clean       | oracle_min_cost              | oracle   | 100 |     0.9400 |    1016.0851 |                     0.9251 |
| clean       | s5_final_condition_mix       | s5_final | 100 |     0.8400 |     899.6500 |                     0.9337 |
| clean       | s5_conservative_noisy_router | s5_final | 100 |     0.8100 |    1359.2800 |                     0.5959 |
| noisy       | s0                           | baseline | 100 |     0.7200 |     992.8700 |                     0.7252 |
| noisy       | s1                           | baseline | 100 |     0.7500 |    1351.1400 |                     0.5551 |
| noisy       | s2                           | baseline | 100 |     0.7400 |     994.2200 |                     0.7443 |
| noisy       | s3_mc                        | baseline | 100 |     0.8000 |    2247.4600 |                     0.3560 |
| noisy       | oracle_min_cost              | oracle   | 100 |     0.9000 |     933.4556 |                     0.9642 |
| noisy       | s5_final_condition_mix       | s5_final | 100 |     0.8100 |     883.6400 |                     0.9167 |
| noisy       | s5_conservative_noisy_router | s5_final | 100 |     0.8100 |     883.6400 |                     0.9167 |
| adversarial | s0                           | baseline | 100 |     0.7200 |     992.8700 |                     0.7252 |
| adversarial | s1                           | baseline | 100 |     0.8100 |    1351.2300 |                     0.5995 |
| adversarial | s2                           | baseline | 100 |     0.7500 |     984.8300 |                     0.7616 |
| adversarial | s3_mc                        | baseline | 100 |     0.7900 |    2150.2500 |                     0.3674 |
| adversarial | oracle_min_cost              | oracle   | 100 |     0.9200 |     900.4022 |                     1.0218 |
| adversarial | s5_final_condition_mix       | s5_final | 100 |     0.7600 |     886.3800 |                     0.8574 |
| adversarial | s5_conservative_noisy_router | s5_final | 100 |     0.8100 |    1351.2300 |                     0.5995 |

## Distribución de decisiones

| policy                       | condition   |   s0 |   s1 |   s2 |   s3_mc |
|:-----------------------------|:------------|-----:|-----:|-----:|--------:|
| s5_conservative_noisy_router | adversarial |    0 |  100 |    0 |       0 |
| s5_conservative_noisy_router | clean       |    0 |  100 |    0 |       0 |
| s5_conservative_noisy_router | noisy       |   37 |   14 |   45 |       4 |
| s5_final_condition_mix       | adversarial |   38 |    6 |   53 |       3 |
| s5_final_condition_mix       | clean       |   54 |    8 |   35 |       3 |
| s5_final_condition_mix       | noisy       |   37 |   14 |   45 |       4 |

## Lectura recomendada

- S1 logra accuracy `0.7900` con `1353.88` tokens promedio.
- S5 final logra accuracy `0.8033` con `889.89` tokens promedio.
- S5 final cambia accuracy vs S1 en `0.0133` y ahorra `34.27%` tokens.
- Oracle sigue marcando el techo: accuracy `0.9200` con `950.58` tokens.
- Variante conservadora logra accuracy `0.8100` con `1198.05` tokens.

## Interpretación

El resultado debe presentarse como una mejora de eficiencia y selección adaptativa, no como una mejora estadísticamente fuerte de accuracy. El valor del S5 final está en igualar o apenas superar al mejor baseline con mucho menor costo promedio.

## Próximo paso

Usar este reporte para redactar `docs/experimentos/informe_s5_meta_router.md`.

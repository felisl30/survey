# S5 Rule-Based Router Report

## Objetivo

Este reporte evalúa varias políticas S5 que eligen entre S0, S1, S2 y S3-MC sin mirar la respuesta correcta. Las reglas usan predicciones, acuerdos, ruta de S2, señales de retrieval y active retrieval de S3-MC.

## Archivos generados

- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_predictions.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_summary.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_by_condition.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_decision_distribution.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_interesting_cases.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\s5_rule_based_report.md`

## Resumen global

| policy               | kind          |   n |   accuracy |   avg_tokens | oracle_match_answer_rate   | oracle_match_system_rate   |   accuracy_per_1000_tokens |   gap_vs_oracle_accuracy |
|:---------------------|:--------------|----:|-----------:|-------------:|:---------------------------|:---------------------------|---------------------------:|-------------------------:|
| s0                   | baseline      | 300 |     0.7200 |     992.8700 | <NA>                       | <NA>                       |                     0.7252 |                   0.2000 |
| s1                   | baseline      | 300 |     0.7900 |    1353.8833 | <NA>                       | <NA>                       |                     0.5835 |                   0.1300 |
| s2                   | baseline      | 300 |     0.7667 |    1025.8033 | <NA>                       | <NA>                       |                     0.7474 |                   0.1533 |
| s3_mc                | baseline      | 300 |     0.8200 |    2237.4933 | <NA>                       | <NA>                       |                     0.3665 |                   0.1000 |
| oracle_min_cost      | oracle        | 300 |     0.9200 |     950.5797 | 1.0                        | 1.0                        |                     0.9678 |                   0.0000 |
| s5_majority_min_cost | s5_rule_based | 300 |     0.8000 |     883.2267 | 0.8                        | 0.8                        |                     0.9058 |                   0.1200 |
| s5_robust_fallback   | s5_rule_based | 300 |     0.8133 |    1105.3900 | 0.8133333333333334         | 0.47                       |                     0.7358 |                   0.1067 |
| s5_cost_aware        | s5_rule_based | 300 |     0.8000 |     987.6800 | 0.8                        | 0.43333333333333335        |                     0.8100 |                   0.1200 |
| s5_risk_aware        | s5_rule_based | 300 |     0.8200 |     896.2233 | 0.82                       | 0.82                       |                     0.9150 |                   0.1000 |

## Resumen por condición

| condition   | policy               | kind          |   n |   accuracy |   avg_tokens |   accuracy_per_1000_tokens |
|:------------|:---------------------|:--------------|----:|-----------:|-------------:|---------------------------:|
| clean       | s0                   | baseline      | 100 |     0.7200 |     992.8700 |                     0.7252 |
| clean       | s1                   | baseline      | 100 |     0.8100 |    1359.2800 |                     0.5959 |
| clean       | s2                   | baseline      | 100 |     0.8100 |    1098.3600 |                     0.7375 |
| clean       | s3_mc                | baseline      | 100 |     0.8700 |    2314.7700 |                     0.3758 |
| clean       | oracle_min_cost      | oracle        | 100 |     0.9400 |    1016.0851 |                     0.9251 |
| clean       | s5_majority_min_cost | s5_rule_based | 100 |     0.8400 |     899.6500 |                     0.9337 |
| clean       | s5_robust_fallback   | s5_rule_based | 100 |     0.8300 |    1106.2200 |                     0.7503 |
| clean       | s5_cost_aware        | s5_rule_based | 100 |     0.8400 |     994.0900 |                     0.8450 |
| clean       | s5_risk_aware        | s5_rule_based | 100 |     0.8300 |     910.1200 |                     0.9120 |
| noisy       | s0                   | baseline      | 100 |     0.7200 |     992.8700 |                     0.7252 |
| noisy       | s1                   | baseline      | 100 |     0.7500 |    1351.1400 |                     0.5551 |
| noisy       | s2                   | baseline      | 100 |     0.7400 |     994.2200 |                     0.7443 |
| noisy       | s3_mc                | baseline      | 100 |     0.8000 |    2247.4600 |                     0.3560 |
| noisy       | oracle_min_cost      | oracle        | 100 |     0.9000 |     933.4556 |                     0.9642 |
| noisy       | s5_majority_min_cost | s5_rule_based | 100 |     0.8000 |     863.6500 |                     0.9263 |
| noisy       | s5_robust_fallback   | s5_rule_based | 100 |     0.7800 |    1117.2400 |                     0.6981 |
| noisy       | s5_cost_aware        | s5_rule_based | 100 |     0.8000 |     974.0700 |                     0.8213 |
| noisy       | s5_risk_aware        | s5_rule_based | 100 |     0.8100 |     883.6400 |                     0.9167 |
| adversarial | s0                   | baseline      | 100 |     0.7200 |     992.8700 |                     0.7252 |
| adversarial | s1                   | baseline      | 100 |     0.8100 |    1351.2300 |                     0.5995 |
| adversarial | s2                   | baseline      | 100 |     0.7500 |     984.8300 |                     0.7616 |
| adversarial | s3_mc                | baseline      | 100 |     0.7900 |    2150.2500 |                     0.3674 |
| adversarial | oracle_min_cost      | oracle        | 100 |     0.9200 |     900.4022 |                     1.0218 |
| adversarial | s5_majority_min_cost | s5_rule_based | 100 |     0.7600 |     886.3800 |                     0.8574 |
| adversarial | s5_robust_fallback   | s5_rule_based | 100 |     0.8300 |    1092.7100 |                     0.7596 |
| adversarial | s5_cost_aware        | s5_rule_based | 100 |     0.7600 |     994.8800 |                     0.7639 |
| adversarial | s5_risk_aware        | s5_rule_based | 100 |     0.8200 |     894.9100 |                     0.9163 |

## Distribución de decisiones

| policy               | condition   |   s0 |   s1 |   s2 |   s3_mc |
|:---------------------|:------------|-----:|-----:|-----:|--------:|
| s5_cost_aware        | adversarial |   91 |    3 |    6 |       0 |
| s5_cost_aware        | clean       |   84 |    4 |   12 |       0 |
| s5_cost_aware        | noisy       |   90 |    5 |    5 |       0 |
| s5_majority_min_cost | adversarial |   38 |    6 |   53 |       3 |
| s5_majority_min_cost | clean       |   54 |    8 |   35 |       3 |
| s5_majority_min_cost | noisy       |   39 |    9 |   48 |       4 |
| s5_risk_aware        | adversarial |   35 |   12 |   50 |       3 |
| s5_risk_aware        | clean       |   53 |    7 |   37 |       3 |
| s5_risk_aware        | noisy       |   37 |   14 |   45 |       4 |
| s5_robust_fallback   | adversarial |    0 |   30 |   61 |       9 |
| s5_robust_fallback   | clean       |    0 |   17 |   73 |      10 |
| s5_robust_fallback   | noisy       |    0 |   25 |   62 |      13 |

## Mejores políticas S5 por accuracy

| policy               | kind          |   n |   accuracy |   avg_tokens |   oracle_match_answer_rate |   oracle_match_system_rate |   accuracy_per_1000_tokens |   gap_vs_oracle_accuracy |
|:---------------------|:--------------|----:|-----------:|-------------:|---------------------------:|---------------------------:|---------------------------:|-------------------------:|
| s5_risk_aware        | s5_rule_based | 300 |     0.8200 |     896.2233 |                     0.8200 |                     0.8200 |                     0.9150 |                   0.1000 |
| s5_robust_fallback   | s5_rule_based | 300 |     0.8133 |    1105.3900 |                     0.8133 |                     0.4700 |                     0.7358 |                   0.1067 |
| s5_majority_min_cost | s5_rule_based | 300 |     0.8000 |     883.2267 |                     0.8000 |                     0.8000 |                     0.9058 |                   0.1200 |
| s5_cost_aware        | s5_rule_based | 300 |     0.8000 |     987.6800 |                     0.8000 |                     0.4333 |                     0.8100 |                   0.1200 |

## Lectura recomendada

- Mejor baseline: `s3_mc` con accuracy `0.8200` y tokens promedio `2237.49`.
- Mejor S5 rule-based: `s5_risk_aware` con accuracy `0.8200` y tokens promedio `896.22`.
- Oracle: accuracy `0.9200` y tokens promedio `950.58`.
- Brecha del mejor S5 contra Oracle: `0.1000`.

## Próximo paso

Revisar el reporte y decidir si conviene:

1. ajustar reglas del S5 más prometedor;
2. incorporar señales de riesgo más finas;
3. generar ejemplos cualitativos para el informe;
4. escribir `informe_s5_meta_router.md`.

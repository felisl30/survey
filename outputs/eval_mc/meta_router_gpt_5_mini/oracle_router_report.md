# Oracle Router Report

## Objetivo

Este análisis estima el techo máximo de una política que pudiera elegir, por cada pregunta y condición, el sistema correcto de menor costo entre S0, S1, S2 y S3-MC.

## Archivos generados

- `outputs\eval_mc\meta_router_gpt_5_mini\oracle_router_summary.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\oracle_router_by_condition.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\oracle_router_selection_distribution.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\oracle_router_pattern_summary.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\oracle_router_report.md`

## Resumen general

| condition   | system          |   n |   accuracy |   avg_tokens |   accuracy_per_1000_tokens |
|:------------|:----------------|----:|-----------:|-------------:|---------------------------:|
| overall     | s0              | 300 |     0.7200 |     992.8700 |                     0.7252 |
| overall     | s1              | 300 |     0.7900 |    1353.8833 |                     0.5835 |
| overall     | s2              | 300 |     0.7667 |    1025.8033 |                     0.7474 |
| overall     | s3_mc           | 300 |     0.8200 |    2237.4933 |                     0.3665 |
| overall     | oracle_min_cost | 300 |     0.9200 |     950.5797 |                     0.9678 |

## Oracle por condición

| condition   |   n | best_single_system   |   best_single_accuracy |   best_single_avg_tokens |   oracle_accuracy |   oracle_avg_min_cost_tokens |   oracle_accuracy_per_1000_tokens |   oracle_gain_vs_best_single |   oracle_gain_vs_s1 |   oracle_gain_vs_s2 |   oracle_gain_vs_s3_mc |   none_correct_rate |   all_s1_s2_s3_agree_rate |   all_s0_s1_s2_s3_agree_rate |
|:------------|----:|:---------------------|-----------------------:|-------------------------:|------------------:|-----------------------------:|----------------------------------:|-----------------------------:|--------------------:|--------------------:|-----------------------:|--------------------:|--------------------------:|-----------------------------:|
| clean       | 100 | s3_mc                |                 0.8700 |                2314.7700 |            0.9400 |                    1016.0851 |                            0.9251 |                       0.0700 |              0.1300 |              0.1300 |                 0.0700 |              0.0600 |                    0.8000 |                       0.6800 |
| noisy       | 100 | s3_mc                |                 0.8000 |                2247.4600 |            0.9000 |                     933.4556 |                            0.9642 |                       0.1000 |              0.1500 |              0.1600 |                 0.1000 |              0.1000 |                    0.7200 |                       0.6500 |
| adversarial | 100 | s1                   |                 0.8100 |                1351.2300 |            0.9200 |                     900.4022 |                            1.0218 |                       0.1100 |              0.1100 |              0.1700 |                 0.1300 |              0.0800 |                    0.7100 |                       0.6500 |
| overall     | 300 | s3_mc                |                 0.8200 |                2237.4933 |            0.9200 |                     950.5797 |                            0.9678 |                       0.1000 |              0.1300 |              0.1533 |                 0.1000 |              0.0800 |                    0.7433 |                       0.6600 |

## Distribución de selección del Oracle

| condition   |   none_correct |   s0 |   s1 |   s2 |   s3_mc |
|:------------|---------------:|-----:|-----:|-----:|--------:|
| adversarial |              8 |   28 |   11 |   48 |       5 |
| clean       |              6 |   47 |    6 |   33 |       8 |
| noisy       |             10 |   31 |   11 |   41 |       7 |
| overall     |             24 |  106 |   28 |  122 |      20 |

## Patrones de correctitud

| condition   | pattern                   |   n |   rate |
|:------------|:--------------------------|----:|-------:|
| clean       | none_correct              |   6 | 0.0600 |
| clean       | all_four_correct          |  63 | 0.6300 |
| clean       | only_s0_correct           |   2 | 0.0200 |
| clean       | only_s1_correct           |   1 | 0.0100 |
| clean       | only_s2_correct           |   1 | 0.0100 |
| clean       | only_s3_mc_correct        |   5 | 0.0500 |
| clean       | s1_correct_s2_wrong       |   4 | 0.0400 |
| clean       | s2_correct_s1_wrong       |   4 | 0.0400 |
| clean       | s3_mc_correct_s1_s2_wrong |   7 | 0.0700 |
| noisy       | none_correct              |  10 | 0.1000 |
| noisy       | all_four_correct          |  57 | 0.5700 |
| noisy       | only_s0_correct           |   1 | 0.0100 |
| noisy       | only_s1_correct           |   1 | 0.0100 |
| noisy       | only_s2_correct           |   1 | 0.0100 |
| noisy       | only_s3_mc_correct        |   3 | 0.0300 |
| noisy       | s1_correct_s2_wrong       |  10 | 0.1000 |
| noisy       | s2_correct_s1_wrong       |   9 | 0.0900 |
| noisy       | s3_mc_correct_s1_s2_wrong |   5 | 0.0500 |
| adversarial | none_correct              |   8 | 0.0800 |
| adversarial | all_four_correct          |  58 | 0.5800 |
| adversarial | only_s0_correct           |   2 | 0.0200 |
| adversarial | only_s1_correct           |   2 | 0.0200 |
| adversarial | only_s2_correct           |   2 | 0.0200 |
| adversarial | only_s3_mc_correct        |   2 | 0.0200 |
| adversarial | s1_correct_s2_wrong       |  13 | 0.1300 |
| adversarial | s2_correct_s1_wrong       |   7 | 0.0700 |
| adversarial | s3_mc_correct_s1_s2_wrong |   2 | 0.0200 |
| overall     | none_correct              |  24 | 0.0800 |
| overall     | all_four_correct          | 178 | 0.5933 |
| overall     | only_s0_correct           |   5 | 0.0167 |
| overall     | only_s1_correct           |   4 | 0.0133 |
| overall     | only_s2_correct           |   4 | 0.0133 |
| overall     | only_s3_mc_correct        |  10 | 0.0333 |
| overall     | s1_correct_s2_wrong       |  27 | 0.0900 |
| overall     | s2_correct_s1_wrong       |  20 | 0.0667 |
| overall     | s3_mc_correct_s1_s2_wrong |  14 | 0.0467 |

## Lectura recomendada

- Mejor sistema individual global: `s3_mc` con accuracy `0.8200`.
- Oracle global: accuracy `0.9200`.
- Ganancia potencial del Oracle sobre el mejor sistema individual: `0.1000`.
- Si esta ganancia es relevante, se justifica implementar un S5 rule-based para aproximar esa selección.

## Próximo paso

Implementar:

```text
evaluation/meta_router/run_s5_rule_based_router.py
```

Ese script debe usar señales disponibles antes de mirar el gold, por ejemplo:

- acuerdos entre S1/S2/S3-MC;
- ruta de S2 (`direct` o `retrieve`);
- scores de retrieval de S2;
- active retrieval de S3-MC;
- condición clean/noisy/adversarial.

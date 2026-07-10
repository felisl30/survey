# Meta-Router Question Table Report

## Archivos generados

- `outputs\eval_mc\meta_router_gpt_5_mini\meta_router_question_table.csv`
- `outputs\eval_mc\meta_router_gpt_5_mini\meta_router_question_table_report.md`

## Shape

- Filas: `300`
- Columnas: `117`

## Filas por condición

| condition   |   count |
|:------------|--------:|
| adversarial |     100 |
| clean       |     100 |
| noisy       |     100 |

## Accuracy por sistema y condición

| condition   | system   |   n |   accuracy |   avg_tokens |
|:------------|:---------|----:|-----------:|-------------:|
| clean       | s0       | 100 |       0.72 |       992.87 |
| clean       | s1       | 100 |       0.81 |      1359.28 |
| clean       | s2       | 100 |       0.81 |      1098.36 |
| clean       | s3_mc    | 100 |       0.87 |      2314.77 |
| noisy       | s0       | 100 |       0.72 |       992.87 |
| noisy       | s1       | 100 |       0.75 |      1351.14 |
| noisy       | s2       | 100 |       0.74 |       994.22 |
| noisy       | s3_mc    | 100 |       0.8  |      2247.46 |
| adversarial | s0       | 100 |       0.72 |       992.87 |
| adversarial | s1       | 100 |       0.81 |      1351.23 |
| adversarial | s2       | 100 |       0.75 |       984.83 |
| adversarial | s3_mc    | 100 |       0.79 |      2150.25 |

## Oracle mínimo costo

| condition   |   oracle_accuracy |   oracle_avg_min_cost_tokens |   none_correct_rate |   only_s1_correct |   only_s2_correct |   only_s3_mc_correct |
|:------------|------------------:|-----------------------------:|--------------------:|------------------:|------------------:|---------------------:|
| clean       |              0.94 |                     1016.09  |                0.06 |                 1 |                 1 |                    5 |
| noisy       |              0.9  |                      933.456 |                0.1  |                 1 |                 1 |                    3 |
| adversarial |              0.92 |                      900.402 |                0.08 |                 2 |                 2 |                    2 |

## Señales útiles disponibles para S5

- `condition`
- `s2_route`
- `s2_retrieved`
- `s2_top1_score`
- `s2_top1_top2_gap`
- `s2_top5_mean_score`
- `s3_mc_active_retrieval`
- `s3_mc_confidence`
- `agreement_s1_s2`
- `agreement_s1_s3_mc`
- `agreement_s2_s3_mc`
- `oracle_min_cost_system`

## Preview

```text
              id condition gold_answer s0_answer  s0_correct_bool s1_answer  s1_correct_bool s2_answer  s2_correct_bool s2_route  s2_retrieved s3_mc_answer  s3_mc_correct_bool  s3_mc_active_retrieval oracle_min_cost_system oracle_correct_systems
musique_mc__0000     clean           A         D            False         A             True         A             True   direct         False            A                True                    True                     s1            s1,s2,s3_mc
musique_mc__0001     clean           D         D             True         D             True         D             True retrieve          True            D                True                   False                  s3_mc         s3_mc,s0,s2,s1
musique_mc__0002     clean           C         B            False         C             True         C             True retrieve          True            C                True                    True                     s2            s2,s1,s3_mc
musique_mc__0003     clean           B         B             True         A            False         D            False retrieve          True            D               False                    True                     s0                     s0
musique_mc__0004     clean           C         C             True         C             True         C             True retrieve          True            C                True                    True                     s0         s0,s2,s1,s3_mc
musique_mc__0005     clean           D         D             True         D             True         D             True retrieve          True            D                True                    True                     s0         s0,s2,s1,s3_mc
musique_mc__0006     clean           D         D             True         D             True         D             True   direct         False            D                True                   False                     s0         s0,s3_mc,s2,s1
musique_mc__0007     clean           B         B             True         B             True         B             True   direct         False            B                True                   False                     s2         s2,s0,s3_mc,s1
musique_mc__0008     clean           D         B            False         A            False         D             True retrieve          True            D                True                    True                     s2               s2,s3_mc
musique_mc__0009     clean           B         B             True         B             True         B             True   direct         False            B                True                    True                     s0         s0,s2,s1,s3_mc
musique_mc__0010     clean           B         A            False         A            False         A            False retrieve          True            A               False                    True                    NaN                       
musique_mc__0011     clean           D         D             True         D             True         D             True retrieve          True            D                True                    True                     s0         s0,s2,s1,s3_mc
musique_mc__0012     clean           B         A            False         C            False         A            False retrieve          True            C               False                    True                    NaN                       
musique_mc__0013     clean           C         A            False         A            False         C             True   direct         False            A               False                    True                     s2                     s2
musique_mc__0014     clean           A         B            False         B            False         B            False   direct         False            A                True                    True                  s3_mc                  s3_mc
```

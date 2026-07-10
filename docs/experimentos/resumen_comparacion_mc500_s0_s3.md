# Comparación final MC-500 — MuSiQue, HotpotQA y 2Wiki

## Estado general

La comparación final quedó completa y consistente. Se evaluaron tres datasets de 500 preguntas multiple-choice cada uno:

- MuSiQue-MC-500
- HotpotQA-MC-500
- 2Wiki-MC-500

En todos los casos se usó el mismo modelo (`gpt-5-mini`) y se compararon cuatro estrategias:

- S0: baseline directo sin recuperación.
- S1: RAG clásico con top-k fijo.
- S2: RAG adaptativo.
- S3: estrategia FLARE-like con recuperación activa.

Todos los sistemas tienen `n=500`, `valid_format_rate=1.0` y outputs completos.

## Tabla principal

| Dataset | Sistema | Accuracy | Delta vs S0 | Tokens prom. | Ratio tokens vs S0 | Latencia prom. |
|---|---:|---:|---:|---:|---:|---:|
| MuSiQue-MC-500 | S0 | 0.696 | 0.000 | 992.910 | 1.000 | 8.695 |
| MuSiQue-MC-500 | S1 | 0.782 | 0.086 | 1462.380 | 1.473 | 5.671 |
| MuSiQue-MC-500 | S2 | 0.754 | 0.058 | 1065.098 | 1.073 | 6.836 |
| MuSiQue-MC-500 | S3 | 0.808 | 0.112 | 2295.708 | 2.312 | 15.179 |
| HotpotQA-MC-500 | S0 | 0.800 | 0.000 | 909.978 | 1.000 | 7.074 |
| HotpotQA-MC-500 | S1 | 0.924 | 0.124 | 1364.164 | 1.499 | 4.772 |
| HotpotQA-MC-500 | S2 | 0.882 | 0.082 | 1121.872 | 1.233 | 5.224 |
| HotpotQA-MC-500 | S3 | 0.942 | 0.142 | 2285.420 | 2.512 | 14.003 |
| 2Wiki-MC-500 | S0 | 0.810 | 0.000 | 1036.246 | 1.000 | 8.110 |
| 2Wiki-MC-500 | S1 | 0.892 | 0.082 | 1368.606 | 1.321 | 6.681 |
| 2Wiki-MC-500 | S2 | 0.850 | 0.040 | 1154.302 | 1.114 | 6.224 |
| 2Wiki-MC-500 | S3 | 0.898 | 0.088 | 2460.528 | 2.374 | 18.015 |

## Lectura principal

El resultado más importante es que la recuperación mejora de forma consistente sobre el baseline directo S0 en los tres datasets.

S3 obtiene la mejor accuracy en todos los datasets:

- MuSiQue-MC-500: 0.808, mejora de +11.2 puntos frente a S0.
- HotpotQA-MC-500: 0.942, mejora de +14.2 puntos frente a S0.
- 2Wiki-MC-500: 0.898, mejora de +8.8 puntos frente a S0.

Sin embargo, S3 también es la estrategia más costosa en tokens. Su costo relativo frente a S0 es:

- MuSiQue: 2.31×
- HotpotQA: 2.51×
- 2Wiki: 2.37×

Por lo tanto, S3 maximiza accuracy, pero no siempre ofrece la mejor relación accuracy/costo.

S1 aparece como una alternativa muy competitiva. En los tres datasets mejora claramente sobre S0 y queda cerca de S3, especialmente en 2Wiki, donde S1 obtiene 0.892 y S3 obtiene 0.898. La diferencia de accuracy es de solo 0.6 puntos, pero S3 usa muchos más tokens.

S2 muestra un comportamiento intermedio: mejora sobre S0 en todos los datasets, pero menos que S1 y S3. Su punto fuerte es el costo: usa bastante menos tokens que S1/S3, especialmente frente a S3. Esto lo vuelve útil como estrategia conservadora cuando se busca mejorar el baseline sin aumentar demasiado el presupuesto de inferencia.

## Conclusión para el informe

La comparación multi-dataset refuerza que las mejoras observadas no son exclusivas de MuSiQue. Al extender la evaluación a HotpotQA-MC-500 y 2Wiki-MC-500, las estrategias con recuperación siguen superando al baseline directo. La recuperación aporta ganancias absolutas de accuracy en todos los datasets, aunque con distintos perfiles de costo.

S3 obtiene el mejor desempeño absoluto, pero su ventaja frente a S1 puede ser pequeña en relación con el aumento de tokens. En consecuencia, la elección de estrategia no debería basarse únicamente en accuracy, sino también en eficiencia. S1 representa una opción robusta y simple para mejorar accuracy con costo moderado, mientras que S2 ofrece una alternativa más económica y S3 actúa como estrategia de máxima performance.

## Archivos finales relevantes

- `outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_s0_s3_comparison_final.md`
- `outputs/eval_mc/cross_dataset/mc500_pivot_accuracy_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_pivot_tokens_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_best_by_dataset_final.csv`
- `outputs/eval_mc/cross_dataset/mc500_comparison_final_notes.md`

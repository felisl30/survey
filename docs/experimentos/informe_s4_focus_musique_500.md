# Informe S4 focalizado sobre MuSiQue-500

## Objetivo

Se evaluó S4 como auditor factual focalizado, no como sistema masivo de respuesta. El objetivo fue analizar casos conflictivos donde S3 MC FLARE-like falló, pero al menos otro sistema más barato acertó. Estos casos son especialmente interesantes porque muestran situaciones en las que una estrategia más compleja puede sobreconfiar en evidencia insuficiente, irrelevante o ambigua.

## Configuración

- Dataset: MuSiQue-500 multiple choice.
- Modelo fuente: gpt-5-mini.
- Sistema auditado: S3 MC FLARE-like.
- Cantidad de casos auditados: 25.
- Tipo de caso: `s3_wrong_others_correct`.
- Variante S4: FIRE-like rule-based auditor.
- Archivo de entrada: `outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_input.csv`.
- Archivo de salida: `outputs/eval_mc/musique_mc_rag_500/posthoc/s4_focus_raw_gpt_5_mini_limit25_rules.csv`.

## Resultado agregado

S4 completó la corrida sin errores sobre los 25 casos focalizados. En todos los casos decidió abstenerse, indicando que no había evidencia suficiente para verificar de manera confiable la respuesta inicial propuesta por S3.

Métricas principales:

| Métrica | Valor |
|---|---:|
| Casos auditados | 25 |
| Run error rate | 0.000 |
| Abstention rate | 1.000 |
| Correction rate | 1.000 |
| Claims promedio | 2.000 |
| Claims soportados promedio | 0.120 |
| Claims refutados promedio | 0.120 |
| Claims NEI promedio | 1.760 |
| Rondas de verificación promedio | 3.800 |
| Rondas de recuperación promedio | 1.800 |
| Chunks recuperados promedio | 3.440 |
| Latencia promedio | 0.401 s |
| Tokens promedio | 0.000 |

## Interpretación

El resultado muestra que S4 se comporta como un auditor conservador. En lugar de aceptar respuestas dudosas de S3, detecta que la evidencia recuperada no alcanza para confirmar la respuesta inicial y decide abstenerse. Esto es especialmente relevante porque los casos auditados fueron seleccionados precisamente donde S3 falló y otros sistemas más baratos acertaron.

Por lo tanto, S4 no debe presentarse como una mejora directa de accuracy en esta etapa, sino como una capa de verificación factual y análisis de errores. Su aporte está en identificar respuestas no suficientemente sustentadas y aportar trazabilidad sobre por qué una respuesta debería ser revisada.

## Conclusión

S4 complementa a S0-S3 y S5 desde una perspectiva distinta: no busca maximizar accuracy directamente, sino controlar el riesgo factual. En MuSiQue-500, sobre casos conflictivos de S3, S4 mostró un comportamiento prudente: ante evidencia insuficiente, se abstuvo sistemáticamente. Esto refuerza la idea de usar S4 como mecanismo de auditoría sobre casos difíciles o de alta incertidumbre.

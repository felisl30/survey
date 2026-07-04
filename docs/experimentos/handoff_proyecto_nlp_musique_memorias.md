# Handoff del proyecto — Memorias adaptativas y robustez en MuSiQue-MC

Este documento resume el estado actual del proyecto, qué se hizo hasta ahora, qué resultados se obtuvieron, qué archivos/directorios son importantes y qué pasos podrían seguir antes de cerrar el informe final.

Está pensado para pasárselo a otra instancia de chat dentro del mismo proyecto y continuar sin perder contexto.

---

## 1. Contexto general del proyecto

El proyecto evalúa distintas estrategias de uso de memoria externa / recuperación de evidencia para modelos de lenguaje en un benchmark de preguntas multiple-choice derivado de MuSiQue.

La idea central es comparar sistemas que van desde no usar memoria externa hasta usar recuperación fija, recuperación adaptativa, recuperación activa y auditoría factual.

El benchmark principal trabajado fue:

```text
MuSiQue-MC 100 preguntas
```

Ubicación principal:

```text
~/Documents/natural_language_processing/trabajo_cientifico
```

Entorno usado:

```text
(tp_cientifico)
```

---

## 2. Sistemas evaluados

Se trabajó con cinco sistemas conceptuales:

| Sistema | Nombre conceptual | Descripción |
|---|---|---|
| S0 | Direct LLM baseline | El modelo responde sin recuperación externa. |
| S1 | Fixed top-k RAG | Recupera siempre top-k documentos y responde con esa evidencia. |
| S2 | Adaptive-RAG | Decide si recuperar o responder directo según señales del retriever. |
| S3-MC | FLARE-like MC | Genera hipótesis inicial, decide si recuperar y regenera/corrige para multiple-choice. |
| S4 | FIRE-like factual auditor | Audita una respuesta inicial mediante extracción/verificación de claims. No se lo trata como sistema final de respuesta, sino como auditor factual. |

---

## 3. Dataset base MuSiQue-MC

El dataset base usado para evaluación principal está en:

```text
data/eval_mc/musique_mc_rag/
```

Archivos clave:

```text
data/eval_mc/musique_mc_rag/questions.csv
data/eval_mc/musique_mc_rag/corpus.csv
data/eval_mc/musique_mc_rag/qrels.csv
data/eval_mc/musique_mc_rag/build_summary.json
```

Tamaños relevantes:

```text
questions.csv: 100 preguntas
corpus.csv:    1200 documentos/contextos
qrels.csv:     1200 relaciones pregunta-documento relevante
```

Índice base:

```text
indexes/eval_mc/musique_mc_rag/
```

Archivos del índice:

```text
indexes/eval_mc/musique_mc_rag/chunks.csv
indexes/eval_mc/musique_mc_rag/embeddings.npy
indexes/eval_mc/musique_mc_rag/metadata.json
```

---

## 4. Resultados previos con GPT-5 mini

En una primera etapa se corrió MuSiQue-MC con GPT-5 mini.

Resultados principales:

| Sistema | Accuracy | Retrieve rate | Tokens promedio |
|---|---:|---:|---:|
| S0 | 0.69 | 0.00 | 914.26 |
| S1 | 0.78 | 1.00 | 1362.82 |
| S2 | 0.81 | 0.60 | 1094.62 |
| S3-MC | 0.85 | 0.82 | 2251.68 |

Interpretación:

- S0 sirve como baseline sin memoria.
- S1 mejora al agregar evidencia externa fija.
- S2 mejora más que S1 usando menos recuperación que S1.
- S3-MC fue el mejor en accuracy, aunque con mayor costo en tokens.
- S3-MC corrigió muchos errores de la hipótesis inicial, con pocas regresiones.

---

## 5. Grilla por tamaño de modelo

Después se evaluó cómo impacta el tamaño del modelo en S0/S1/S2/S3-MC.

Modelos usados:

```text
gpt-5.4-nano
gpt-5.4-mini
gpt-5.4
```

Script principal:

```text
scripts/run_musique_model_grid_s0_s3.sh
```

Output principal:

```text
outputs/eval_mc/model_grid_musique/
```

Resultados de accuracy:

| Modelo | S0 | S1 | S2 | S3-MC |
|---|---:|---:|---:|---:|
| gpt-5.4-nano | 0.32 | 0.41 | 0.38 | 0.49 |
| gpt-5.4-mini | 0.48 | 0.57 | 0.59 | 0.61 |
| gpt-5.4 | 0.55 | 0.64 | 0.68 | 0.76 |

Deltas vs S0:

| Modelo | S1 - S0 | S2 - S0 | S3-MC - S0 |
|---|---:|---:|---:|
| gpt-5.4-nano | +0.09 | +0.06 | +0.17 |
| gpt-5.4-mini | +0.09 | +0.11 | +0.13 |
| gpt-5.4 | +0.09 | +0.13 | +0.21 |

Interpretación:

- Las estrategias de memoria ayudan en todos los tamaños.
- S3-MC tiende a ser el sistema más fuerte, especialmente con el modelo más grande.
- S2 muestra buen balance entre mejora de accuracy y ahorro de tokens.
- La ganancia de memoria externa no desaparece con modelos más grandes; en algunos casos se amplifica.

---

## 6. Experimento de robustez con distractores

Luego se agregó un bloque extra para evaluar robustez de la memoria externa frente a ruido documental.

La pregunta experimental fue:

```text
¿Qué pasa con S1, S2 y S3-MC cuando el corpus recuperable contiene distractores?
```

Se generaron tres condiciones:

| Condición | Descripción |
|---|---|
| clean | Corpus original, sin distractores. |
| noisy | Corpus original + 12 distractores aleatorios por pregunta. |
| adversarial | Corpus original + 12 distractores semánticamente similares por pregunta. |

Dataset robusto:

```text
data/eval_mc/robustness_musique/
```

Archivos generados:

```text
data/eval_mc/robustness_musique/questions.csv
data/eval_mc/robustness_musique/qrels.csv
data/eval_mc/robustness_musique/corpus_clean.csv
data/eval_mc/robustness_musique/corpus_noisy.csv
data/eval_mc/robustness_musique/corpus_adversarial.csv
data/eval_mc/robustness_musique/build_summary.json
```

Tamaños:

| Condición | Filas corpus | Gold contexts | Distractores |
|---|---:|---:|---:|
| clean | 1200 | 1200 | 0 |
| noisy | 2400 | 1200 | 1200 |
| adversarial | 2400 | 1200 | 1200 |

Índices robustos:

```text
indexes/eval_mc/robustness_musique_clean/
indexes/eval_mc/robustness_musique_noisy/
indexes/eval_mc/robustness_musique_adversarial/
```

Cada índice contiene:

```text
chunks.csv
embeddings.npy
metadata.json
```

---

## 7. Retrieval en clean/noisy/adversarial

Antes de gastar API, se evaluó retrieval puro.

Resultados:

| Condición | hit@1 | hit@5 | mrr@10 |
|---|---:|---:|---:|
| clean | 0.72 | 0.87 | 0.784 |
| noisy | 0.44 | 0.82 | 0.604 |
| adversarial | 0.49 | 0.77 | 0.613 |

Interpretación:

- El retrieval empeora al agregar distractores.
- La caída más fuerte aparece en hit@1.
- Adversarial degrada más hit@5 que noisy.
- El experimento robusto es válido porque efectivamente tensiona la recuperación.

---

## 8. Corrida completa de robustez S0/S1/S2/S3-MC

Modelo usado:

```text
gpt-5.4-mini
```

Output principal:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
```

Script usado:

```text
scripts/run_musique_robustness_s0_s3.sh
```

Resumen generado con:

```text
evaluation/summarize_musique_robustness_s0_s3.py
```

Reporte:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_s0_s3_report.txt
```

Resultados:

| Sistema | Clean | Noisy | Adversarial |
|---|---:|---:|---:|
| S1 | 0.58 | 0.57 | 0.57 |
| S2 | 0.54 | 0.47 | 0.52 |
| S3-MC | 0.55 | 0.52 | 0.52 |

Baseline:

| Sistema | Accuracy |
|---|---:|
| S0 | 0.48 |

Todas las corridas tuvieron:

```text
n = 100
valid_format_rate = 1.0
run_error_rate = 0.0
```

Interpretación:

- S1 fue sorprendentemente estable frente al ruido.
- S2 fue eficiente, pero sensible al routing.
- S3-MC tuvo valor en casos puntuales, pero no dominó globalmente en robustez.
- La degradación del retrieval no se traduce linealmente en caída de accuracy.

---

## 9. Análisis profundo de robustez

Script:

```text
evaluation/analyze_musique_robustness_deep.py
```

Reporte:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_report.txt
```

Archivos generados:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_system_summary.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_condition_deltas.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_question_matrix.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_patterns_summary.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_interesting_cases.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_deep_report.txt
```

Resultados importantes:

| Sistema | Condición | Accuracy | Tokens promedio | Retrieve/active rate |
|---|---|---:|---:|---:|
| S1 | clean | 0.58 | 904.12 | siempre recupera |
| S1 | noisy | 0.57 | 881.00 | siempre recupera |
| S1 | adversarial | 0.57 | 881.53 | siempre recupera |
| S2 | clean | 0.54 | 599.29 | retrieve 0.60 |
| S2 | noisy | 0.47 | 402.11 | retrieve 0.25 |
| S2 | adversarial | 0.52 | 412.68 | retrieve 0.25 |
| S3-MC | clean | 0.55 | 1133.44 | active retrieval 0.87 |
| S3-MC | noisy | 0.52 | 982.09 | active retrieval 0.71 |
| S3-MC | adversarial | 0.52 | 1049.43 | active retrieval 0.75 |

Interpretación clave:

- S2 pierde accuracy en noisy porque recupera mucho menos.
- S2 se vuelve demasiado conservador al cambiar el corpus.
- S3-MC reduce su active retrieval en noisy/adversarial.
- S1 se mantiene estable porque siempre recibe top-k evidencia.

---

## 10. Ejemplos cualitativos de robustez

Script:

```text
evaluation/export_musique_robustness_qualitative_examples.py
```

Reporte generado:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/robustness_qualitative_examples.md
```

Ejemplos seleccionados para informe:

### `musique_mc__0000`

Patrón:

```text
S1 estable, S2 falla
```

S1 y S3-MC aciertan en clean/noisy/adversarial. S2 falla en las tres.

### `musique_mc__0019`

Patrón:

```text
S2/S3-MC regresan en noisy, pero se recuperan en adversarial
```

Muestra que noisy y adversarial no son simplemente niveles lineales de dificultad.

### `musique_mc__0029`

Patrón:

```text
S3-MC rescata adversarial
```

S3-MC es correcto en noisy/adversarial donde S1/S2 fallan.

### `musique_mc__0036`

Patrón:

```text
S1 vence en adversarial
```

S3-MC acierta en clean/noisy, pero falla en adversarial. S1 termina siendo el único correcto.

### `musique_mc__0020`

Patrón:

```text
Todos fallan en adversarial
```

Todos aciertan en clean, pero todos fallan en adversarial. Buen candidato para auditoría S4.

---

## 11. Informe de cierre de robustez

Se preparó un informe específico:

```text
docs/experimentos/informe_robustez_musique.md
```

También se recomendó guardarlo en:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/informe_robustez_musique.md
```

Contenido del informe:

- Motivación.
- Construcción del benchmark robusto.
- Evaluación de retrieval.
- Resultados S0/S1/S2/S3-MC.
- Análisis por sistema.
- Patrones por pregunta.
- Ejemplos cualitativos.
- Conclusión del bloque.

Conclusión central:

```text
No existe una estrategia de memoria universalmente óptima. S1 es estable,
S2 es eficiente pero sensible al routing, y S3-MC rescata casos puntuales aunque
también puede sufrir regresiones frente a distractores.
```

---

## 12. S4 como auditor factual focalizado

Después de cerrar robustez, se probó S4 sobre un subset pequeño y focalizado.

Objetivo:

```text
Evaluar si S4 puede detectar respuestas de S3-MC afectadas por ruido/distractores
sin rechazar respuestas correctas.
```

Input focalizado:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/input/s4_robustness_focus_core5.csv
```

Script usado para construirlo:

```text
evaluation/build_s4_robustness_focus_input.py
```

Subset core5:

| ID | Condición | Tipo de caso | S3-MC correcto | Esperado de S4 |
|---|---|---|---:|---|
| `musique_mc__0020` | adversarial | adversarial_all_rag_wrong | No | detect_error |
| `musique_mc__0036` | adversarial | s3_mc_regresses_adversarial | No | detect_error |
| `musique_mc__0029` | adversarial | adversarial_only_s3_mc_correct | Sí | preserve_correct |
| `musique_mc__0019` | noisy | s3_mc_regresses_noisy | No | detect_error |
| `musique_mc__0022` | adversarial | s3_mc_regresses_adversarial | No | detect_error |

Distribución esperada:

```text
detect_error:      4
preserve_correct: 1
```

---

## 13. S4 dry-run

Se validó primero que S4 pudiera leer el input.

Comando conceptual:

```text
python s4_model_code/run_s4_fire_like.py ... --dry-run --no-index
```

Resultado:

```text
5 filas generadas
90 columnas
run_error_present = False
dry_run = True
```

Interpretación:

- El pipeline S4 lee bien el input.
- El dry-run no sirve para evaluar factualidad porque no usa índice ni API.
- Era esperable que marcara muchos `not_enough_info`.

---

## 14. S4 rules + índice real

Se corrió S4 con índice real y todas las estrategias en `rules`.

Script:

```text
scripts/run_s4_robustness_focus_rules.sh
```

Resumen:

```text
evaluation/summarize_s4_robustness_focus.py
```

Reporte:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_rules_report.txt
```

Resultados:

```text
n: 5
expected_match_rate: 0.8
s4_suspicious_rate: 1.0
abstention_rate: 1.0
expected_error_detected: 4
false_rejection_of_correct_s3: 1
```

Interpretación:

- S4 rules detectó los 4 errores esperados.
- También rechazó el único caso correcto.
- Funcionó como auditor conservador, no como corrector final.

---

## 15. S4 LLM-verifier + índice real

Después se corrió S4 cambiando solo el verificador:

```text
verification-strategy: llm
```

Manteniendo:

```text
claim-strategy: rules
query-strategy: rules
repair-strategy: rules
retrieval: índice real
model: gpt-5.4-mini
```

Reporte:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_llm_verify_report.txt
```

Resultados:

```text
n: 5
expected_match_rate: 0.8
s4_suspicious_rate: 1.0
abstention_rate: 1.0
expected_error_detected: 4
false_rejection_of_correct_s3: 1
```

Comparación:

| Variante | Errores detectados | Falsos rechazos | Abstention rate |
|---|---:|---:|---:|
| S4 rules | 4/4 | 1/1 | 1.0 |
| S4 LLM-verifier | 4/4 | 1/1 | 1.0 |

Diferencia importante:

- Rules devolvía casi todo como `not_enough_info`.
- LLM-verifier produjo veredictos más ricos a nivel claim:
  - `refuted`
  - `supported`
  - `not_enough_info`

Ejemplos:

```text
musique_mc__0020: refuted|not_enough_info
musique_mc__0022: not_enough_info|refuted|supported
musique_mc__0019: not_enough_info|refuted
```

Interpretación:

- El LLM-verifier no mejoró la decisión global.
- Sí mejoró la trazabilidad factual claim-level.
- S4 sigue siendo conservador.

---

## 16. Informe de cierre S4

Se preparó un informe específico:

```text
docs/experimentos/informe_s4_robustez_focus.md
```

También se recomendó guardarlo en:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/informe_s4_robustez_focus.md
```

Conclusión central:

```text
S4 sirve como auditor conservador de soporte factual. Detecta respuestas
problemáticas o no soportadas, pero puede rechazar respuestas correctas si la
evidencia explícita recuperada no alcanza. No debe venderse como sistema final
para mejorar accuracy.
```

---

## 17. Archivos y scripts clave del proyecto

### Dataset base

```text
data/eval_mc/musique_mc_rag/questions.csv
data/eval_mc/musique_mc_rag/corpus.csv
data/eval_mc/musique_mc_rag/qrels.csv
data/eval_mc/musique_mc_rag/build_summary.json
```

### Dataset robusto

```text
data/eval_mc/robustness_musique/questions.csv
data/eval_mc/robustness_musique/qrels.csv
data/eval_mc/robustness_musique/corpus_clean.csv
data/eval_mc/robustness_musique/corpus_noisy.csv
data/eval_mc/robustness_musique/corpus_adversarial.csv
data/eval_mc/robustness_musique/build_summary.json
```

### Índices

```text
indexes/eval_mc/musique_mc_rag/
indexes/eval_mc/robustness_musique_clean/
indexes/eval_mc/robustness_musique_noisy/
indexes/eval_mc/robustness_musique_adversarial/
```

### Scripts S0-S3

```text
scripts/run_musique_model_grid_s0_s3.sh
scripts/run_musique_robustness_s0_s3.sh
evaluation/run_s1_mc_rag.py
evaluation/run_s2_mc_real_adaptive.py
evaluation/run_s3_mc_flare_like.py
```

### Scripts de dataset/index/retrieval

```text
evaluation/build_mc_rag_dataset.py
evaluation/build_mc_rag_index.py
evaluation/evaluate_mc_rag_retrieval.py
evaluation/build_mc_robustness_dataset.py
```

### Scripts de análisis

```text
evaluation/summarize_musique_robustness_s0_s3.py
evaluation/analyze_musique_robustness_deep.py
evaluation/export_musique_robustness_qualitative_examples.py
```

### Scripts S4

```text
s4_model_code/run_s4_fire_like.py
evaluation/build_s4_robustness_focus_input.py
scripts/run_s4_robustness_focus_rules.sh
evaluation/summarize_s4_robustness_focus.py
```

### Informes preparados

```text
docs/experimentos/informe_robustez_musique.md
docs/experimentos/informe_s4_robustez_focus.md
```

### Outputs principales

```text
outputs/eval_mc/model_grid_musique/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/
```

---

## 18. Estructura conceptual del informe final

Una estructura sugerida para el informe final sería:

```text
1. Introducción
2. Motivación: memoria externa y razonamiento multi-hop
3. Dataset MuSiQue-MC
4. Sistemas comparados: S0, S1, S2, S3-MC, S4
5. Pipeline experimental
6. Resultados principales en MuSiQue-MC
7. Grilla por tamaño de modelo
8. Experimento de robustez con distractores
9. Análisis cualitativo de robustez
10. S4 como auditor factual
11. Discusión
12. Limitaciones
13. Conclusiones
```

---

## 19. Mensaje conceptual para defender el proyecto

La tesis experimental que se puede defender es:

```text
La memoria externa mejora el rendimiento de modelos de lenguaje en preguntas
multi-hop, pero su utilidad depende de cómo se decide recuperar, cómo se usa la
evidencia y qué tan contaminado está el corpus recuperable.
```

Y más específicamente:

```text
S1 muestra robustez por recuperar siempre evidencia.
S2 muestra eficiencia, pero su política adaptativa puede volverse frágil.
S3-MC puede corregir hipótesis y rescatar casos difíciles, pero también puede
ser afectado por distractores.
S4 aporta trazabilidad factual y detección de respuestas no soportadas, aunque
su conservadurismo puede producir falsos rechazos.
```

---

## 20. Próximos pasos posibles antes de cerrar el informe

No parece necesario correr más experimentos grandes.

Pasos recomendados:

1. Consolidar resultados en tablas finales.
2. Elegir 3 a 5 gráficos definitivos.
3. Integrar los informes parciales en el informe final.
4. Revisar consistencia de nombres: S3-MC, S4/FIRE-like, Adaptive-RAG.
5. Preparar una sección breve de limitaciones.
6. Preparar una sección de trabajo futuro.

Posibles gráficos finales:

```text
accuracy_by_model_system.png
retrieval_rate_by_model_system.png
tokens_by_model_system.png
accuracy_clean_noisy_adversarial.png
tokens_clean_noisy_adversarial.png
s4_rules_vs_llm_verify_summary.png
```

Limitaciones importantes a mencionar:

```text
MuSiQue-MC tiene solo 100 preguntas en esta versión experimental.
El benchmark robusto fue construido artificialmente con distractores.
S4 se evaluó en subset focalizado pequeño, no como benchmark masivo.
La métrica de S4 no debe confundirse con accuracy de respuesta final.
Los resultados dependen del retriever, top-k y política de routing.
```

Trabajo futuro:

```text
Evaluar S4 en datasets más naturalmente fact-checking como FEVER o SciFact.
Ajustar la política adaptativa de S2 para no sub-recuperar en corpus ruidosos.
Diseñar una versión de S3-MC más robusta ante distractores.
Probar S4 como señal de abstención o revisión humana, no como reemplazo de respuesta.
Evaluar con más preguntas y más modelos.
```

---

## 21. Estado actual recomendado

El proyecto está en una etapa buena para pasar a redacción final.

Ya se cuenta con:

```text
benchmark principal
grilla por tamaño de modelo
experimento de robustez
análisis cuantitativo
análisis cualitativo
S4 focalizado como auditor
informes parciales en Markdown
```

La recomendación es no abrir demasiados experimentos nuevos salvo que sean necesarios para completar una figura o una tabla faltante.

El foco debería pasar a:

```text
organizar resultados
redactar discusión
preparar tablas/figuras finales
cerrar conclusiones
```

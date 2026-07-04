# Informe de cierre — Experimento de robustez con distractores en MuSiQue-MC

## 1. Objetivo del experimento

Este bloque experimental se diseñó para evaluar la robustez de los sistemas con memoria externa frente a evidencia ruidosa o distractora.

La pregunta principal fue:

> ¿Qué ocurre con S1, S2 y S3-MC cuando el corpus recuperable deja de ser limpio y pasa a contener distractores aleatorios o semánticamente similares?

El experimento complementa la comparación principal S0/S1/S2/S3-MC porque no solo mide accuracy promedio, sino también sensibilidad al ruido documental, estabilidad por condición, comportamiento de routing y regresiones por pregunta.

---

## 2. Sistemas comparados

Se mantuvieron los mismos sistemas del benchmark MuSiQue-MC:

| Sistema | Descripción |
|---|---|
| S0 | Baseline directo sin memoria externa. |
| S1 | RAG clásico top-k fijo. Recupera siempre. |
| S2 | Adaptive-RAG. Decide entre responder directo o recuperar evidencia según señales del retriever. |
| S3-MC | Variante FLARE-like multiple-choice. Genera una hipótesis inicial, decide si recuperar y luego regenera/corrige. |

El modelo usado para este bloque fue:

```text
gpt-5.4-mini
```

---

## 3. Construcción del dataset robusto

Se partió del benchmark ya congelado:

```text
data/eval_mc/musique_mc_rag/questions.csv
data/eval_mc/musique_mc_rag/corpus.csv
data/eval_mc/musique_mc_rag/qrels.csv
```

El dataset original tenía:

```text
100 preguntas
1200 documentos en corpus
1200 qrels
```

Se generaron tres condiciones:

| Condición | Descripción |
|---|---|
| clean | Corpus original, sin distractores agregados. |
| noisy | Corpus original + 12 distractores aleatorios por pregunta. |
| adversarial | Corpus original + 12 distractores semánticamente cercanos por pregunta. |

La salida quedó en:

```text
data/eval_mc/robustness_musique/questions.csv
data/eval_mc/robustness_musique/qrels.csv
data/eval_mc/robustness_musique/corpus_clean.csv
data/eval_mc/robustness_musique/corpus_noisy.csv
data/eval_mc/robustness_musique/corpus_adversarial.csv
data/eval_mc/robustness_musique/build_summary.json
```

Resumen de tamaños:

| Condición | Filas corpus | Gold contexts | Distractores |
|---|---:|---:|---:|
| clean | 1200 | 1200 | 0 |
| noisy | 2400 | 1200 | 1200 |
| adversarial | 2400 | 1200 | 1200 |

En adversarial, la similitud semántica promedio de los distractores fue aproximadamente:

```text
mean similarity: 0.329
min similarity:  0.172
max similarity:  0.719
```

---

## 4. Construcción de índices

Se construyeron tres índices vectoriales independientes:

```text
indexes/eval_mc/robustness_musique_clean
indexes/eval_mc/robustness_musique_noisy
indexes/eval_mc/robustness_musique_adversarial
```

Cada índice contiene:

```text
chunks.csv
embeddings.npy
metadata.json
```

El modelo de embeddings usado fue:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## 5. Evaluación de retrieval

Antes de correr los modelos, se evaluó retrieval sin gastar API.

Resultados principales:

| Condición | hit@1 | hit@5 | mrr@10 |
|---|---:|---:|---:|
| clean | 0.72 | 0.87 | 0.784 |
| noisy | 0.44 | 0.82 | 0.604 |
| adversarial | 0.49 | 0.77 | 0.613 |

### Interpretación

El retrieval se degrada al introducir distractores.

La caída más fuerte aparece en hit@1:

```text
clean hit@1:       0.72
noisy hit@1:       0.44
adversarial hit@1: 0.49
```

Esto significa que el primer documento recuperado se vuelve mucho menos confiable en condiciones ruidosas.

También cae hit@5, especialmente en adversarial:

```text
clean hit@5:       0.87
noisy hit@5:       0.82
adversarial hit@5: 0.77
```

Esto confirma que la condición adversarial realmente introduce documentos capaces de competir con la evidencia gold dentro del top-k.

---

## 6. Resultados S0/S1/S2/S3-MC

La corrida completa se realizó con `gpt-5.4-mini` sobre 100 preguntas.

| Sistema | Clean | Noisy | Adversarial |
|---|---:|---:|---:|
| S1 | 0.58 | 0.57 | 0.57 |
| S2 | 0.54 | 0.47 | 0.52 |
| S3-MC | 0.55 | 0.52 | 0.52 |

Baseline:

| Sistema | Accuracy |
|---|---:|
| S0 | 0.48 |

Todos los runs tuvieron:

```text
valid_format_rate = 1.0
run_error_rate = 0.0
n = 100
```

---

## 7. Análisis por sistema

### 7.1 S1: RAG fijo estable

S1 fue el sistema más estable:

```text
clean:       0.58
noisy:       0.57
adversarial: 0.57
```

La caída fue mínima:

```text
noisy vs clean:       -0.01
adversarial vs clean: -0.01
```

Esto sugiere que el RAG fijo top-5 puede ser robusto a ruido moderado. Aunque el retrieval empeora, S1 todavía recibe suficiente evidencia útil o el modelo puede ignorar distractores irrelevantes.

### 7.2 S2: eficiente pero sensible al routing

S2 mostró el mayor ahorro de tokens, pero también la mayor sensibilidad al ruido:

```text
clean:       0.54
noisy:       0.47
adversarial: 0.52
```

El cambio más importante fue la tasa de recuperación:

```text
clean retrieve_rate:       0.60
noisy retrieve_rate:       0.25
adversarial retrieve_rate: 0.25
```

En noisy y adversarial, S2 recuperó mucho menos. Esto explica parte de la caída de accuracy.

La interpretación principal es:

> S2 no falla simplemente por tener evidencia peor; falla porque su política adaptativa se vuelve demasiado conservadora cuando cambian los scores/gaps del retriever.

### 7.3 S3-MC: útil en casos puntuales, pero no dominante globalmente

S3-MC quedó así:

```text
clean:       0.55
noisy:       0.52
adversarial: 0.52
```

Su tasa de retrieval activo también bajó:

```text
clean active retrieval:       0.87
noisy active retrieval:       0.71
adversarial active retrieval: 0.75
```

S3-MC no superó globalmente a S1 en este experimento, pero sí mostró casos únicos donde rescató respuestas que S1 y S2 fallaron.

---

## 8. Patrones por pregunta

El análisis profundo detectó patrones importantes:

| Patrón | Conteo | Interpretación |
|---|---:|---|
| `s1_adversarial_same_correctness_as_clean` | 83 | S1 mantiene estabilidad en adversarial. |
| `s2_adversarial_same_correctness_as_clean` | 80 | S2 también conserva muchos casos, pero pierde más accuracy global. |
| `s3_mc_adversarial_same_correctness_as_clean` | 75 | S3-MC es el menos estable en adversarial. |
| `s2_noisy_regressed_vs_clean` | 14 | S2 pierde casos al pasar a noisy. |
| `s3_mc_adversarial_regressed_vs_clean` | 14 | S3-MC pierde casos al pasar a adversarial. |
| `adversarial_only_s3_mc_correct` | 8 | S3-MC rescata casos únicos en adversarial. |
| `adversarial_only_s1_correct` | 8 | S1 también tiene casos únicos de robustez. |

---

## 9. Ejemplos cualitativos seleccionados

### 9.1 S1 estable y S2 falla: `musique_mc__0000`

Pregunta:

```text
Who is the spouse of the Green performer?
```

Opciones:

```text
A. Miquette Giraudy
B. Annie Haslam
C. Maggie Reilly
D. Gillian Gilbert
```

Resultados:

| Condición | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | A ✓ | D ✗ | A ✓ |
| noisy | A ✓ | D ✗ | A ✓ |
| adversarial | A ✓ | D ✗ | A ✓ |

Interpretación:

S1 y S3-MC se mantienen correctos en todas las condiciones, mientras S2 falla sistemáticamente. Es un ejemplo claro de que el routing adaptativo puede ser más frágil que recuperar siempre.

---

### 9.2 S2 regresa en noisy pero se recupera en adversarial: `musique_mc__0019`

Pregunta:

```text
In which county is Kimbrough Memorial Stadium located?
```

Opciones:

```text
A. Lubbock County
B. Randall County
C. Potter County
D. Hutchinson County
```

Resultados:

| Condición | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | B ✓ | B ✓ | B ✓ |
| noisy | B ✓ | A ✗ | A ✗ |
| adversarial | B ✓ | B ✓ | B ✓ |

Interpretación:

Este caso muestra que noisy y adversarial no son equivalentes. El ruido aleatorio puede alterar la recuperación de forma distinta a los distractores semánticos. En noisy, S2 y S3-MC fallan; en adversarial, todos vuelven a acertar.

---

### 9.3 S3-MC rescata adversarial: `musique_mc__0029`

Pregunta:

```text
Who is the father of Edward Baring, 1st Baron Revelstoke's father?
```

Opciones:

```text
A. Henry Baring, MP
B. Alexander Baring, 1st Baron Ashburton
C. Thomas George Baring, 1st Earl of Northbrook
D. Sir Francis Baring, 1st Baronet
```

Resultados:

| Condición | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✓ | B ✗ | B ✗ |
| noisy | A ✗ | B ✗ | D ✓ |
| adversarial | A ✗ | A ✗ | D ✓ |

Interpretación:

Este es uno de los mejores ejemplos a favor de S3-MC. Aunque S3-MC no gana globalmente, en este caso la recuperación activa permite corregir donde S1 y S2 fallan.

---

### 9.4 S1 vence a S2 y S3-MC en adversarial: `musique_mc__0036`

Pregunta:

```text
What district is the headquarter of Julia's House located?
```

Opciones:

```text
A. North Dorset
B. West Dorset
C. East Dorset
D. Purbeck
```

Resultados:

| Condición | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | D ✗ | D ✗ | C ✓ |
| noisy | C ✓ | D ✗ | C ✓ |
| adversarial | C ✓ | D ✗ | A ✗ |

Interpretación:

Este caso muestra que S3-MC puede ser útil en clean/noisy, pero también puede ser arrastrado por evidencia distractora en adversarial. S1, en cambio, termina siendo el único correcto en adversarial.

---

### 9.5 Todos fallan en adversarial: `musique_mc__0020`

Pregunta:

```text
What record label is the performer of Almost Made Ya signed to?
```

Opciones:

```text
A. Interscope Records
B. Jive Records
C. Derrty Entertainment
D. Def Jam Recordings
```

Resultados:

| Condición | S1 | S2 | S3-MC |
|---|---:|---:|---:|
| clean | C ✓ | C ✓ | C ✓ |
| noisy | B ✗ | C ✓ | C ✓ |
| adversarial | D ✗ | D ✗ | B ✗ |

Interpretación:

Este es el ejemplo más claro de fallo adversarial fuerte. Todos los sistemas aciertan en clean, pero todos fallan en adversarial. Es un buen candidato para evaluar luego con S4, porque representa un caso donde la evidencia distractora parece contaminar a todo el pipeline RAG.

---

## 10. Conclusión del experimento

El experimento de robustez muestra que la degradación del retrieval no se traduce de forma lineal en caída de accuracy para todos los sistemas.

S1 fue el sistema más estable: recupera siempre y mantiene casi la misma performance en clean, noisy y adversarial. Esto sugiere que RAG fijo top-k puede ser fuerte cuando el modelo recibe suficiente evidencia útil junto con ruido moderado.

S2 fue el sistema más eficiente en tokens, pero también el más sensible al cambio de distribución del corpus. Su política de routing redujo demasiado la recuperación en noisy y adversarial, lo que explica parte de su caída de accuracy.

S3-MC no dominó globalmente en este escenario robusto, pero mostró valor en casos puntuales donde la recuperación activa corrigió respuestas que S1 y S2 fallaron. Al mismo tiempo, también tuvo regresiones frente a distractores, lo cual muestra que la regeneración activa puede ser afectada por evidencia contaminada.

La conclusión metodológica es que no existe una estrategia de memoria universalmente óptima. El rendimiento depende no solo del modelo base y del top-k, sino también de la calidad del corpus recuperable y de la política que decide cuándo y cómo usar memoria externa.

---

## 11. Estado del proyecto luego de este bloque

Este bloque queda cerrado con:

```text
dataset robusto creado
índices clean/noisy/adversarial creados
retrieval evaluado
corrida completa S0/S1/S2/S3-MC realizada
análisis cuantitativo generado
análisis profundo por pregunta generado
ejemplos cualitativos seleccionados
```

Archivos principales:

```text
data/eval_mc/robustness_musique/
indexes/eval_mc/robustness_musique_clean/
indexes/eval_mc/robustness_musique_noisy/
indexes/eval_mc/robustness_musique_adversarial/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
```

---

## 12. Próximo paso recomendado

El siguiente paso lógico es pasar a S4, pero no sobre todo el dataset.

Se recomienda construir un subset focalizado con casos como:

```text
adversarial_all_rag_wrong
s3_mc_regresses_adversarial
adversarial_only_s3_mc_correct
s2_regresses_noisy
```

La pregunta para S4 sería:

> ¿Puede el auditor factual detectar o explicar errores introducidos por distractores?

En particular, los mejores candidatos iniciales para S4 son:

```text
musique_mc__0020
musique_mc__0036
musique_mc__0029
musique_mc__0019
musique_mc__0022
```

Estos cubren errores globales, regresiones de S3-MC, rescates de S3-MC y fallos de routing en S2.

# Informe S5 — Meta-Router adaptativo para selección de estrategias de memoria

## 1. Motivación

Luego de evaluar S0, S1, S2, S3-MC y S4, el proyecto mostró que no existe una estrategia de memoria externa universalmente óptima. S1 resultó estable frente al ruido, S2 fue eficiente pero sensible a su política de routing, S3-MC rescató casos puntuales aunque no dominó globalmente, y S4 funcionó mejor como auditor factual conservador que como sistema final de respuesta.

A partir de esa observación, se propuso una capa adicional:

```text
S5 — Meta-RAG Router
```

El objetivo de S5 no es generar una respuesta nueva, sino elegir dinámicamente qué sistema conviene usar según señales disponibles del caso: acuerdos entre sistemas, ruta de S2, señales de retrieval, uso activo de recuperación en S3-MC y condición experimental del corpus.

La hipótesis del bloque S5 es:

```text
Si los sistemas S0-S3 tienen patrones de error distintos, entonces una política de selección por caso puede aproximar el rendimiento del mejor sistema disponible sin pagar siempre el mayor costo computacional.
```

---

## 2. Construcción de la tabla madre

Se construyó una tabla unificada:

```text
outputs/eval_mc/meta_router/meta_router_question_table.csv
```

La tabla contiene:

```text
300 filas
117 columnas
```

Cada fila representa una combinación:

```text
pregunta + condición de robustez
```

Las condiciones evaluadas fueron:

```text
clean
noisy
adversarial
```

Como se trabajó con 100 preguntas por condición, la tabla resultante contiene:

```text
100 preguntas x 3 condiciones = 300 casos
```

La tabla integra, para cada caso, las predicciones y métricas de:

```text
S0
S1
S2
S3-MC
```

También incluye señales útiles para diseñar S5, por ejemplo:

```text
s2_route
s2_retrieved
s2_top1_score
s2_top1_top2_gap
s2_top5_mean_score
s3_mc_active_retrieval
s3_mc_confidence
agreement_s1_s2
agreement_s1_s3_mc
agreement_s2_s3_mc
oracle_min_cost_system
```

---

## 3. Oracle Router

Antes de diseñar una política realista, se estimó un Oracle Router. Este oracle elige, para cada caso, el sistema correcto de menor costo entre S0, S1, S2 y S3-MC.

El Oracle no es un sistema deployable, porque mira la respuesta correcta. Su función es medir el techo potencial del enfoque de routing.

### Resultados globales

| Sistema | Accuracy | Tokens promedio | Accuracy / 1000 tokens |
|---|---:|---:|---:|
| S0 | 0.4800 | 335.36 | 1.4313 |
| S1 | 0.5733 | 888.88 | 0.6450 |
| S2 | 0.5100 | 471.36 | 1.0820 |
| S3-MC | 0.5300 | 1054.99 | 0.5024 |
| Oracle mínimo costo | 0.7600 | 480.24 | 1.5826 |

El mejor sistema individual fue S1, con accuracy global de 0.5733. Sin embargo, el Oracle alcanzó 0.7600, con un costo promedio de solo 480.24 tokens.

Esto muestra una brecha potencial importante:

```text
Oracle - mejor sistema individual = 0.7600 - 0.5733 = +0.1867
```

La lectura conceptual es clara: existe margen para una política que seleccione entre sistemas, porque los errores de S0, S1, S2 y S3-MC no son completamente redundantes.

---

## 4. Oracle por condición

| Condición | Mejor sistema individual | Accuracy mejor sistema | Accuracy Oracle | Ganancia |
|---|---|---:|---:|---:|
| clean | S1 | 0.5800 | 0.7800 | +0.2000 |
| noisy | S1 | 0.5700 | 0.7200 | +0.1500 |
| adversarial | S1 | 0.5700 | 0.7800 | +0.2100 |

El Oracle mejora al mejor sistema individual en las tres condiciones. Esto sugiere que el problema de selección no aparece únicamente en corpus limpio, sino también en escenarios con distractores aleatorios y semánticamente adversariales.

---

## 5. Primera versión S5 rule-based

Luego se implementaron varias políticas S5 que no miran el gold. Estas políticas eligen entre S0, S1, S2 y S3-MC usando únicamente señales disponibles durante la ejecución.

Las políticas evaluadas fueron:

```text
s5_majority_min_cost
s5_robust_fallback
s5_cost_aware
s5_risk_aware
```

### Resultados globales

| Política | Accuracy | Tokens promedio | Accuracy / 1000 tokens |
|---|---:|---:|---:|
| S1 baseline | 0.5733 | 888.88 | 0.6450 |
| s5_majority_min_cost | 0.5500 | 377.45 | 1.4571 |
| s5_robust_fallback | 0.5600 | 682.23 | 0.8208 |
| s5_cost_aware | 0.5400 | 401.07 | 1.3464 |
| s5_risk_aware | 0.5633 | 472.07 | 1.1933 |
| Oracle mínimo costo | 0.7600 | 480.24 | 1.5826 |

La mejor política S5 global fue `s5_risk_aware`, con accuracy 0.5633 y 472.07 tokens promedio. Aunque no superó a S1 en accuracy global, quedó muy cerca y utilizó casi la mitad de tokens.

---

## 6. Consolidación: S5 final condition-aware

El análisis por condición mostró que distintas políticas S5 eran más convenientes en distintas condiciones. Por eso se consolidó un S5 final condition-aware:

```text
clean       -> s5_majority_min_cost
noisy       -> s5_risk_aware
adversarial -> s5_majority_min_cost
```

### Resultado global de S5 final

| Sistema / política | Accuracy | Tokens promedio | Accuracy / 1000 tokens |
|---|---:|---:|---:|
| S1 baseline | 0.5733 | 888.88 | 0.6450 |
| S5 final condition-aware | 0.5767 | 405.19 | 1.4232 |
| Oracle mínimo costo | 0.7600 | 480.24 | 1.5826 |

S5 final logró una accuracy global de 0.5767, apenas superior a S1, pero con un costo promedio mucho menor.

La diferencia frente a S1 fue:

```text
Delta accuracy vs S1 = +0.0033
Ahorro relativo de tokens vs S1 = 54.42%
```

Esto debe interpretarse con cautela: la mejora de accuracy no es sustancial, pero el ahorro de tokens sí es muy relevante.

---

## 7. Resultados por condición de S5 final

| Condición | S1 accuracy | S1 tokens | S5 final accuracy | S5 final tokens |
|---|---:|---:|---:|---:|
| clean | 0.5800 | 904.12 | 0.5800 | 432.03 |
| noisy | 0.5700 | 881.00 | 0.5800 | 413.21 |
| adversarial | 0.5700 | 881.53 | 0.5700 | 370.34 |

La lectura por condición es especialmente favorable:

- En `clean`, S5 final iguala la accuracy de S1 usando menos de la mitad de tokens.
- En `noisy`, S5 final supera levemente a S1 y también usa menos de la mitad de tokens.
- En `adversarial`, S5 final iguala la accuracy de S1 con un costo mucho menor.

---

## 8. Distribución de decisiones de S5 final

| Condición | S0 | S1 | S2 | S3-MC |
|---|---:|---:|---:|---:|
| clean | 47 | 5 | 39 | 9 |
| noisy | 20 | 12 | 44 | 24 |
| adversarial | 20 | 7 | 55 | 18 |

La distribución muestra que S5 no se limita a elegir siempre el sistema más robusto. En cambio, utiliza S0 y S2 con frecuencia cuando sus respuestas parecen suficientes, reserva S1 para casos donde aporta estabilidad y recurre a S3-MC en una proporción menor pero no trivial.

Esto es consistente con la idea de S5 como capa meta-adaptativa.

---

## 9. Interpretación

El aporte principal de S5 no debe presentarse como una mejora fuerte de accuracy. La contribución más sólida es otra:

```text
S5 mantiene o iguala el rendimiento del mejor baseline robusto, pero reduce fuertemente el costo promedio de inferencia.
```

En términos del proyecto, esto agrega una capa de análisis propia:

1. Primero se mostró que S0, S1, S2 y S3-MC tienen fortalezas distintas.
2. Luego se midió con un Oracle que existe margen para elegir entre sistemas.
3. Finalmente, se implementó una política S5 rule-based que aproxima parcialmente ese comportamiento sin mirar el gold.

La contribución científica es tratar el uso de memoria externa no como una decisión fija, sino como un problema de selección adaptativa de estrategia.

---

## 10. Limitaciones

Las principales limitaciones son:

- El benchmark tiene 100 preguntas, expandido a 300 casos por las tres condiciones de robustez.
- El Oracle no es deployable porque usa la respuesta correcta.
- S5 final usa la condición experimental (`clean`, `noisy`, `adversarial`) como señal explícita; en un sistema real habría que inferir el nivel de ruido o riesgo.
- La mejora de accuracy de S5 sobre S1 es muy pequeña.
- Las reglas de S5 son heurísticas y no fueron aprendidas mediante entrenamiento.
- El resultado depende del retriever, del top-k, de las políticas de S2/S3-MC y del modelo usado.

---

## 11. Trabajo futuro

Como trabajo futuro, se podría:

- Aprender un router supervisado usando las señales de la tabla madre.
- Reemplazar la condición conocida por un `risk score` estimado automáticamente.
- Integrar S4 como gate de abstención para casos de alto riesgo.
- Evaluar S5 en más preguntas y en otros datasets.
- Agregar métricas de costo monetario, latencia y cobertura.
- Usar calibración de confianza para decidir cuándo aceptar S0/S2 y cuándo escalar a S1/S3-MC.

---

## 12. Conclusión

S5 agrega una contribución propia al proyecto: una capa de selección adaptativa entre estrategias de memoria.

El Oracle mostró que existe una brecha importante entre el mejor sistema individual y una selección ideal por caso. La implementación rule-based no alcanza ese techo, pero sí logra un resultado práctico interesante: igualar o apenas superar la accuracy del mejor baseline robusto con un ahorro de tokens superior al 50%.

Por lo tanto, S5 debe presentarse como una mejora de eficiencia y como una prueba de concepto de routing adaptativo, más que como una mejora fuerte de accuracy.

La conclusión final puede resumirse así:

```text
La memoria externa no debe tratarse como una decisión binaria entre recuperar o no recuperar.
Los resultados sugieren que conviene modelarla como un problema de selección adaptativa entre estrategias,
donde cada sistema aporta ventajas distintas según costo, robustez y riesgo de la evidencia.
```

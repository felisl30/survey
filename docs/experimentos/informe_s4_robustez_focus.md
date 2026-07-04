# Informe de cierre — S4 como auditor factual en el experimento de robustez

## 1. Objetivo

Este bloque evaluó S4 como auditor factual FIRE-like sobre un subconjunto focalizado del experimento de robustez MuSiQue-MC.

La pregunta principal fue:

> ¿Puede S4 detectar respuestas de S3-MC afectadas por ruido o distractores adversariales sin rechazar respuestas correctas?

S4 no fue tratado como un sistema final de respuesta multiple-choice. Su función fue auditar una respuesta inicial producida por S3-MC y decidir si estaba suficientemente soportada por la evidencia disponible.

---

## 2. Subset focalizado

Se construyó un input de 5 casos (`core5`) a partir del análisis cualitativo de robustez.

Casos incluidos:

| ID | Condición | Tipo de caso | S3-MC correcto | Esperado de S4 |
|---|---|---|---:|---|
| `musique_mc__0020` | adversarial | `adversarial_all_rag_wrong` | No | detectar error |
| `musique_mc__0036` | adversarial | `s3_mc_regresses_adversarial` | No | detectar error |
| `musique_mc__0029` | adversarial | `adversarial_only_s3_mc_correct` | Sí | preservar correcto |
| `musique_mc__0022` | adversarial | `s3_mc_regresses_adversarial` | No | detectar error |
| `musique_mc__0019` | noisy | `s3_mc_regresses_noisy` | No | detectar error |

La distribución esperada fue:

```text
detect_error:      4
preserve_correct: 1
```

---

## 3. Configuraciones evaluadas

Se evaluaron dos variantes de S4 sobre el mismo subset:

### 3.1 S4 rules

Configuración:

```text
claim-strategy: rules
verification-strategy: rules
query-strategy: rules
repair-strategy: rules
retrieval: índice real
```

### 3.2 S4 LLM-verifier

Configuración:

```text
claim-strategy: rules
verification-strategy: llm
query-strategy: rules
repair-strategy: rules
retrieval: índice real
model: gpt-5.4-mini
```

La segunda variante cambió únicamente el verificador de claims, manteniendo fijo el resto del pipeline. Esto permite aislar el efecto de reemplazar la verificación léxica/reglas por una verificación con LLM.

---

## 4. Resultados agregados

### 4.1 Comparación global

| Variante | n | expected_match_rate | suspicious_rate | abstention_rate | errores detectados | falsos rechazos |
|---|---:|---:|---:|---:|---:|---:|
| S4 rules | 5 | 0.80 | 1.00 | 1.00 | 4 | 1 |
| S4 LLM-verifier | 5 | 0.80 | 1.00 | 1.00 | 4 | 1 |

Ambas configuraciones produjeron la misma categoría global:

```text
expected_error_detected:          4
false_rejection_of_correct_s3:    1
```

---

## 5. Resultado de S4 rules

S4 con reglas detectó los cuatro casos incorrectos, pero también rechazó el único caso correcto.

Resumen:

```text
n: 5
expected_match_rate: 0.8
s4_suspicious_rate: 1.0
abstention_rate: 1.0
expected_error_detected: 4
false_rejection_of_correct_s3: 1
```

Todos los casos fueron marcados como sospechosos y terminaron en abstención.

La salida dominante fue:

```text
No hay información suficiente en la evidencia recuperada para verificar la respuesta inicial.
```

Interpretación:

S4 rules funcionó como auditor extremadamente conservador. Fue útil para marcar errores, pero no logró preservar el caso correcto `musique_mc__0029`.

---

## 6. Resultado de S4 LLM-verifier

S4 con verificación LLM mantuvo la misma categoría global:

```text
n: 5
expected_match_rate: 0.8
s4_suspicious_rate: 1.0
abstention_rate: 1.0
expected_error_detected: 4
false_rejection_of_correct_s3: 1
```

Sin embargo, cambió la calidad de la verificación claim-level.

Promedios:

```text
avg_supported_claims: 0.2
avg_refuted_claims:   0.6
avg_nei_claims:       1.4
avg_chunks_retrieved: 10.8
```

A diferencia de S4 rules, el verificador LLM sí encontró algunos claims explícitamente refutados o soportados.

Ejemplos:

| ID | Veredictos S4 LLM | Interpretación |
|---|---|---|
| `musique_mc__0020` | `refuted|not_enough_info` | Detecta contradicción parcial de la respuesta inicial. |
| `musique_mc__0022` | `not_enough_info|refuted|supported` | Mezcla soporte, refutación e insuficiencia; refleja trazabilidad más rica. |
| `musique_mc__0019` | `not_enough_info|refuted` | Detecta al menos una afirmación contradicha. |
| `musique_mc__0029` | `not_enough_info` | Sigue rechazando el único caso correcto. |

---

## 7. Caso clave: `musique_mc__0029`

Este caso era el test más importante para medir preservación.

Pregunta:

```text
Who is the father of Edward Baring, 1st Baron Revelstoke's father?
```

S3-MC respondió correctamente en adversarial:

```text
gold_answer: D
s4_mc_final_choice: D
```

Pero S4 lo clasificó como:

```text
false_rejection_of_correct_s3
s4_final_decision: abstained
s4_claim_verdicts: not_enough_info
```

Interpretación:

S4 no contradijo la respuesta correcta; simplemente no consiguió suficiente evidencia explícita para verificarla. Esto confirma que S4 es conservador incluso con verificación LLM.

---

## 8. Conclusión principal

El resultado confirma que S4 es útil como auditor de soporte factual, no como sistema final para maximizar accuracy.

S4 detectó todos los errores seleccionados de S3-MC:

```text
error detection on selected errors: 4/4
```

Pero también rechazó el único caso correcto del subset:

```text
correct preservation on selected correct case: 0/1
```

Por lo tanto, el aporte de S4 debe formularse así:

> S4 permite marcar respuestas potencialmente no soportadas o contradichas por la evidencia, aportando trazabilidad factual y señales de riesgo. Sin embargo, su naturaleza conservadora puede producir falsos rechazos sobre respuestas correctas, especialmente en preguntas multi-hop donde la evidencia requerida es fragmentaria o difícil de recuperar.

---

## 9. Diferencia entre rules y LLM-verifier

Aunque las métricas globales fueron iguales, la variante LLM-verifier aportó más información:

| Aspecto | S4 rules | S4 LLM-verifier |
|---|---|---|
| Detecta los 4 errores | Sí | Sí |
| Preserva el caso correcto | No | No |
| Produce claims refutados | No | Sí |
| Produce claims soportados | No | Sí |
| Trazabilidad factual | Baja/media | Mayor |
| Riesgo de falso rechazo | Alto | Alto |

La mejora del LLM-verifier no está en la decisión final, sino en el diagnóstico claim-level.

---

## 10. Cómo reportarlo en el informe

Una formulación adecuada sería:

```text
En un subset focalizado de cinco casos de robustez, S4 detectó los cuatro casos
en que S3-MC producía una respuesta incorrecta bajo ruido o distractores
adversariales. Tanto la variante basada en reglas como la variante con
verificador LLM marcaron el 100% de los errores seleccionados como sospechosos.
Sin embargo, ambas configuraciones rechazaron también el único caso correcto
incluido en el subset. Esto confirma que S4 funciona mejor como auditor
conservador de soporte factual que como mecanismo directo de mejora de accuracy.
La variante con verificación LLM no redujo el falso rechazo, pero sí produjo
diagnósticos más informativos a nivel de claims, incluyendo veredictos refuted y
supported.
```

---

## 11. Estado del bloque S4

Este bloque queda cerrado con:

```text
input focalizado core5 construido
dry-run S4 validado
S4 rules + índice real ejecutado
S4 LLM-verifier + índice real ejecutado
comparación rules vs LLM-verifier generada
resultado interpretado como auditor conservador
```

Archivos principales:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/input/s4_robustness_focus_core5.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_rules_report.txt
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_llm_verify_report.txt
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_rules_combined.csv
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/core5_llm_verify_combined.csv
```

---

## 12. Recomendación

No se recomienda expandir S4 a todo el dataset MuSiQue-MC en este momento.

El resultado ya es suficiente para defender S4 como:

```text
módulo de auditoría
módulo de trazabilidad factual
detector conservador de respuestas no soportadas
```

No debería presentarse como:

```text
sistema final de respuesta
método para mejorar directamente accuracy
reemplazo de S1/S2/S3-MC
```

El próximo paso del proyecto debería ser consolidar el informe general con los bloques:

```text
S0/S1/S2/S3-MC en MuSiQue-MC
grilla por tamaño de modelo
robustez clean/noisy/adversarial
S4 como auditor focalizado
```

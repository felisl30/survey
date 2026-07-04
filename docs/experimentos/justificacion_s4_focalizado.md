# Justificación metodológica — Por qué evaluar S4 sobre casos focalizados

## 1. Contexto

En el proyecto se compararon varias arquitecturas de pregunta-respuesta sobre MuSiQue-MC:

- **S0:** baseline directo sin memoria externa.
- **S1:** RAG clásico con recuperación fija.
- **S2:** Adaptive-RAG, que decide si responder directo o recuperar evidencia.
- **S3-MC:** active retrieval / FLARE-like adaptado a multiple-choice.
- **S4:** auditor factual FIRE-like basado en extracción y verificación de claims.

S0, S1, S2 y S3-MC son sistemas de respuesta final: reciben una pregunta y devuelven una opción A/B/C/D.

S4 cumple otro rol: no está pensado principalmente para elegir una opción multiple-choice desde cero, sino para auditar una respuesta ya generada. Su objetivo es tomar una respuesta candidata, extraer claims verificables, buscar evidencia adicional y decidir si esa respuesta está apoyada, contradicha o insuficientemente respaldada por la evidencia.

Por eso, S4 no debe evaluarse exactamente igual que S0-S3. Requiere un análisis más focalizado y personalizado.

---

## 2. Diferencia entre evaluar un respondedor y evaluar un auditor

Un sistema como S1, S2 o S3-MC se puede evaluar de manera directa con accuracy:

```text
pregunta -> respuesta A/B/C/D -> comparación con gold_answer
```

En cambio, S4 opera sobre otra unidad de análisis:

```text
pregunta + respuesta generada + evidencia -> juicio factual sobre la respuesta
```

Esto cambia la pregunta experimental.

Para S1/S2/S3-MC preguntamos:

> ¿El sistema respondió correctamente?

Para S4 preguntamos:

> ¿El auditor puede detectar cuándo una respuesta está mal, cuándo está bien soportada, y cuándo la evidencia no alcanza?

Por eso, las métricas relevantes para S4 no son solamente accuracy, sino también:

```text
error_detection_rate
false_pass_rate
correct_preservation_rate
false_rejection_rate
supported/refuted/not_enough_info por claim
cantidad de claims extraídos
cantidad de evidencia recuperada
decisión final del auditor
```

---

## 3. Por qué no conviene correr S4 sobre todos los casos

Correr S4 sobre las 100 preguntas completas no sería la mejor primera evaluación por cuatro motivos.

### 3.1 S4 es más costoso

S4 no hace una única llamada simple. Su pipeline puede incluir:

```text
extracción de claims
retrieval para cada claim
verificación claim-level
decisión final
posible reparación o abstención
```

Eso lo vuelve más caro en tokens, latencia y complejidad que S0-S3.

Como S4 no es el sistema principal de respuesta final, conviene usarlo donde más aporta valor: en casos donde hay riesgo factual, contradicción, regresión o evidencia contaminada.

### 3.2 Muchos casos correctos no son informativos

Si S3-MC responde bien y la evidencia es clara, S4 probablemente solo confirmará algo que ya sabemos.

Eso no es inútil, pero aporta poca información científica.

Los casos más interesantes son aquellos donde:

```text
S3-MC falla.
S3-MC cambia una respuesta correcta por una incorrecta.
Todos los sistemas RAG fallan en adversarial.
S3-MC es el único que acierta.
S1 acierta pero S2/S3 fallan.
S2 falla por routing.
```

En esos casos, S4 puede mostrar si realmente detecta problemas de soporte factual o si simplemente rechaza por exceso de conservadurismo.

### 3.3 S4 ya mostró comportamiento conservador

En evaluaciones previas, S4 funcionó como auditor conservador: detectó muchos errores de S3-MC, pero también rechazó muchas respuestas correctas.

Eso significa que el análisis de S4 debe mirar dos tipos de error:

```text
false_pass:
    S4 deja pasar una respuesta incorrecta.

false_rejection:
    S4 marca como problemática una respuesta correcta.
```

Si se lo corre sobre todo el dataset sin distinguir tipos de caso, estos fenómenos pueden quedar mezclados y ser difíciles de interpretar.

### 3.4 El experimento de robustez generó patrones específicos

El experimento clean/noisy/adversarial produjo patrones por pregunta que son ideales para auditar con S4.

Ejemplos de patrones útiles:

```text
adversarial_all_rag_wrong
s3_mc_regresses_adversarial
adversarial_only_s3_mc_correct
s2_regresses_noisy
adversarial_only_s1_correct
```

Estos patrones no son equivalentes. Cada uno permite estudiar una capacidad distinta de S4.

---

## 4. Qué aporta un análisis focalizado de S4

Un subset focalizado permite que S4 responda preguntas más precisas.

### 4.1 Casos `adversarial_all_rag_wrong`

Estos son casos donde S1, S2 y S3-MC fallan en la condición adversarial.

Pregunta experimental:

> ¿Puede S4 detectar que la respuesta final no está bien apoyada por la evidencia?

Este tipo de caso evalúa la capacidad de S4 para detectar contaminación o insuficiencia factual cuando todo el pipeline RAG fue engañado.

---

### 4.2 Casos `s3_mc_regresses_adversarial`

Estos son casos donde S3-MC acierta en clean o noisy, pero falla en adversarial.

Pregunta experimental:

> ¿Puede S4 detectar que la regeneración activa de S3-MC fue perjudicada por evidencia distractora?

Este patrón es especialmente importante porque S3-MC no solo responde: también decide si recuperar y luego regenera. Si la evidencia adversarial lo arrastra hacia una respuesta incorrecta, S4 debería idealmente identificar que la justificación no está bien soportada.

---

### 4.3 Casos `adversarial_only_s3_mc_correct`

Estos son casos donde S3-MC es el único sistema que acierta en adversarial.

Pregunta experimental:

> ¿Puede S4 preservar una corrección difícil lograda por S3-MC?

Este patrón evalúa el problema contrario: no queremos que S4 rechace todo. Un buen auditor debe detectar errores, pero también preservar respuestas correctas cuando están bien justificadas.

Estos casos son importantes para medir `false_rejection_rate`.

---

### 4.4 Casos `s2_regresses_noisy`

Estos son casos donde S2 acierta en clean pero falla en noisy.

Pregunta experimental:

> ¿Puede S4 ayudar a explicar errores provocados por routing adaptativo o sub-recuperación?

S2 no siempre falla por mala generación; a veces falla porque decide no recuperar o porque recupera menos de lo necesario. S4 puede ayudar a distinguir si el error viene de falta de evidencia, evidencia incorrecta o decisión de ruta.

---

## 5. Por qué este análisis merece ser personalizado

S4 merece un análisis personalizado porque su salida es más rica que una letra A/B/C/D.

Mientras S1/S2/S3-MC se comparan principalmente por accuracy, S4 produce información sobre:

```text
qué claims extrajo
qué evidencia recuperó
qué claims fueron supported
qué claims fueron refuted
qué claims quedaron not_enough_info
si la respuesta final debe preservarse, marcarse como sospechosa o rechazarse
```

Esto lo convierte en una herramienta de diagnóstico, no solo en otro respondedor.

Por eso, evaluarlo únicamente como si fuera S0-S3 perdería parte de su valor. Lo importante no es solo si cambia la respuesta, sino si explica por qué una respuesta debe ser confiable o no.

---

## 6. Relación con el informe

En el informe, esta decisión se puede justificar así:

> Dado que S4 cumple el rol de auditor factual y no de respondedor primario, no lo evaluamos inicialmente sobre todo el benchmark como un sistema más de accuracy. En cambio, lo aplicamos sobre un subconjunto focalizado de casos seleccionados por patrones de error y robustez. Esto permite estudiar de forma más precisa si S4 detecta respuestas incorrectas inducidas por distractores, si preserva respuestas correctas difíciles y si puede explicar fallos de routing o regeneración en S2/S3-MC.

Esta formulación deja claro que el subset focalizado no es una limitación accidental, sino una decisión metodológica.

---

## 7. Subset recomendado para S4

El primer subset recomendado debería incluir aproximadamente entre 20 y 40 casos, balanceados entre patrones.

Propuesta:

```text
5-8 casos adversarial_all_rag_wrong
5-8 casos s3_mc_regresses_adversarial
5-8 casos adversarial_only_s3_mc_correct
5-8 casos s2_regresses_noisy
5-8 casos adversarial_only_s1_correct o noisy_only_s1_correct
```

También se pueden incluir manualmente casos cualitativos ya seleccionados:

```text
musique_mc__0020
musique_mc__0036
musique_mc__0029
musique_mc__0019
musique_mc__0022
```

Estos casos cubren:

```text
fallo global en adversarial
regresión de S3-MC
rescate único de S3-MC
fallo de S2 en noisy
caso donde S3-MC acierta en clean pero falla con distractores
```

---

## 8. Métricas esperadas para S4 focalizado

Para el subset focalizado, las métricas centrales deberían ser:

```text
n_cases
n_s3_correct
n_s3_wrong
s4_suspicious_rate
error_detection_rate
false_pass_rate
correct_preservation_rate
false_rejection_rate
avg_claims_per_case
avg_supported_claims
avg_refuted_claims
avg_nei_claims
avg_chunks_retrieved
```

Además, conviene reportar resultados por tipo de caso:

```text
adversarial_all_rag_wrong
s3_mc_regresses_adversarial
adversarial_only_s3_mc_correct
s2_regresses_noisy
```

Esto permite saber si S4 funciona mejor detectando errores que preservando aciertos, o viceversa.

---

## 9. Conclusión

S4 debe evaluarse sobre casos focalizados porque su rol experimental es diferente al de S0-S3.

S0-S3 responden preguntas. S4 audita respuestas.

La evaluación focalizada permite estudiar si S4 aporta trazabilidad factual en los puntos donde el pipeline realmente lo necesita: errores inducidos por distractores, regresiones de active retrieval, fallos de routing adaptativo y respuestas correctas difíciles que no deberían ser rechazadas.

Por eso, el análisis personalizado de S4 no es un atajo ni una evaluación parcial injustificada. Es la forma metodológicamente adecuada de medir un sistema cuyo objetivo principal no es maximizar accuracy directa, sino controlar la confiabilidad factual de respuestas generadas por otros sistemas.

# Cómo reproducir el proyecto NLP MuSiQue / Memorias Adaptativas

Este documento deja un procedimiento práctico para que otra persona pueda preparar el entorno, verificar archivos clave y volver a correr los resultados principales del proyecto.

## 1. Resumen del proyecto

El proyecto evalúa estrategias de memoria externa y recuperación de evidencia sobre un benchmark de preguntas multiple-choice derivado de MuSiQue.

Sistemas principales:

| Sistema | Descripción |
|---|---|
| S0 | Baseline directo: responde sin retrieval. |
| S1 | RAG clásico: recupera siempre top-k evidencia. |
| S2 | Adaptive-RAG: decide si recuperar o responder directo. |
| S3-MC | FLARE-like para multiple choice: hipótesis inicial, recuperación activa y corrección. |
| S4 | FIRE-like factual auditor: audita soporte factual de respuestas. |
| S5 | Meta-Router adaptativo: selecciona entre S0, S1, S2 y S3-MC según señales del caso. |

Directorio esperado del proyecto:

```bash
~/Documents/natural_language_processing/trabajo_cientifico
```

Entorno esperado:

```bash
tp_cientifico
```

## 2. Preparar el entorno desde cero

Desde la raíz del proyecto:

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
python3 -m venv tp_cientifico
source tp_cientifico/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configurar la API key:

```bash
cp .env.example .env
nano .env
```

Dentro de `.env` debe quedar algo así:

```bash
OPENAI_API_KEY="TU_API_KEY"
```

Importante: no subir `.env` a Git.

## 3. Verificación rápida del entorno

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

python --version
python - <<'PY'
import pandas as pd
import numpy as np
import sklearn
import openai
print("OK: imports principales funcionando")
PY
```

## 4. Verificar datasets e índices necesarios

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico

for f in \
  data/eval_mc/musique_mc_rag/questions.csv \
  data/eval_mc/musique_mc_rag/corpus.csv \
  data/eval_mc/musique_mc_rag/qrels.csv \
  data/eval_mc/musique_mc_rag/build_summary.json \
  indexes/eval_mc/musique_mc_rag/chunks.csv \
  indexes/eval_mc/musique_mc_rag/embeddings.npy \
  indexes/eval_mc/musique_mc_rag/metadata.json \
  data/eval_mc/robustness_musique/questions.csv \
  data/eval_mc/robustness_musique/qrels.csv \
  data/eval_mc/robustness_musique/corpus_clean.csv \
  data/eval_mc/robustness_musique/corpus_noisy.csv \
  data/eval_mc/robustness_musique/corpus_adversarial.csv \
  data/eval_mc/robustness_musique/build_summary.json \
  indexes/eval_mc/robustness_musique_clean/chunks.csv \
  indexes/eval_mc/robustness_musique_clean/embeddings.npy \
  indexes/eval_mc/robustness_musique_clean/metadata.json \
  indexes/eval_mc/robustness_musique_noisy/chunks.csv \
  indexes/eval_mc/robustness_musique_noisy/embeddings.npy \
  indexes/eval_mc/robustness_musique_noisy/metadata.json \
  indexes/eval_mc/robustness_musique_adversarial/chunks.csv \
  indexes/eval_mc/robustness_musique_adversarial/embeddings.npy \
  indexes/eval_mc/robustness_musique_adversarial/metadata.json
  do
    test -f "$f" && echo "OK: $f" || echo "FALTA: $f"
  done
```

Si faltan índices, hay que reconstruirlos antes de correr RAG/S2/S3/S4.

## 5. Modo recomendado: reproducir análisis sin gastar API

Este modo usa outputs ya generados. Es ideal para verificar que el proyecto quedó consistente sin volver a llamar modelos.

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

bash scripts/reproducir_resultados_principales.sh
```

El script hace:

1. Verificación de entorno.
2. Verificación de archivos clave.
3. Compilación básica de scripts Python.
4. Regeneración de análisis y reportes a partir de outputs existentes.
5. Regeneración del bloque S5 Meta-Router.

## 6. Modo completo: volver a correr experimentos con API

Este modo puede consumir créditos/API.

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

RUN_API=1 bash scripts/reproducir_resultados_principales.sh
```

Con `RUN_API=1`, el script intenta volver a correr:

- grilla MuSiQue S0-S3;
- robustez MuSiQue clean/noisy/adversarial;
- S4 auditor factual focalizado;
- análisis posteriores;
- S5 Meta-Router.

## 7. Comandos individuales importantes

### 7.1 Grilla de modelos S0-S3

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

bash scripts/run_musique_model_grid_s0_s3.sh
python evaluation/summarize_musique_model_grid_s0_s3.py
python evaluation/plot_model_grid_results.py
```

Outputs esperados:

```text
outputs/eval_mc/model_grid_musique/
outputs/eval_mc/model_grid_musique/analysis/
outputs/eval_mc/model_grid_musique/analysis/figures/
```

### 7.2 Robustez S0-S3

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

bash scripts/run_musique_robustness_s0_s3.sh
python evaluation/summarize_musique_robustness_s0_s3.py
python evaluation/analyze_musique_robustness_deep.py
python evaluation/export_musique_robustness_qualitative_examples.py
```

Outputs esperados:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/analysis/
```

### 7.3 S4 auditor factual focalizado

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

python evaluation/build_s4_robustness_focus_input.py
bash scripts/run_s4_robustness_focus_rules.sh
python evaluation/summarize_s4_robustness_focus.py
```

Outputs esperados:

```text
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/
outputs/eval_mc/robustness_musique/gpt_5_4_mini/s4/analysis/
```

### 7.4 S5 Meta-Router

S5 usa los outputs previos de S0-S3 sobre robustez. Si esos outputs ya existen, normalmente no debería gastar API.

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

python evaluation/meta_router/build_meta_router_table.py
python evaluation/meta_router/analyze_oracle_router.py
python evaluation/meta_router/run_s5_rule_based_router.py
python evaluation/meta_router/run_s5_final_router.py
```

Outputs esperados:

```text
outputs/eval_mc/meta_router/meta_router_question_table.csv
outputs/eval_mc/meta_router/oracle_router_report.md
outputs/eval_mc/meta_router/s5_rule_based_report.md
outputs/eval_mc/meta_router/s5_final_router_report.md
docs/experimentos/informe_s5_meta_router.md
```

## 8. Ver informes finales

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico

cat docs/experimentos/informe_robustez_musique.md
cat docs/experimentos/informe_s4_robustez_focus.md
cat docs/experimentos/informe_s5_meta_router.md
cat outputs/eval_mc/meta_router/s5_final_router_report.md
```

## 9. Verificar qué va a subir a Git

Antes de hacer commit:

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico

git status --short

git check-ignore -v \
  docs/handoff/COMO_REPRODUCIR_PROYECTO.md \
  scripts/reproducir_resultados_principales.sh \
  || true
```

Si `git check-ignore` no imprime nada para esos dos archivos, no están ignorados y se pueden subir normalmente.

Agregar guía y script:

```bash
git add docs/handoff/COMO_REPRODUCIR_PROYECTO.md scripts/reproducir_resultados_principales.sh
git status --short
git commit -m "Agrego guía de reproducción del proyecto"
git push origin main
```

## 10. Verificar si datasets e índices están versionados

Para que otros puedan correr el proyecto después de clonar el repo, no alcanza con subir la guía y el script: también tienen que estar disponibles los datasets e índices, o debe existir un procedimiento para reconstruirlos.

Ver si Git ya los está trackeando:

```bash
cd ~/Documents/natural_language_processing/trabajo_cientifico

git ls-files data/eval_mc/musique_mc_rag | head
git ls-files data/eval_mc/robustness_musique | head
git ls-files indexes/eval_mc/musique_mc_rag | head
git ls-files indexes/eval_mc/robustness_musique_clean | head
git ls-files indexes/eval_mc/robustness_musique_noisy | head
git ls-files indexes/eval_mc/robustness_musique_adversarial | head
```

Ver si hay archivos demasiado grandes para GitHub:

```bash
find \
  data/eval_mc/musique_mc_rag \
  data/eval_mc/robustness_musique \
  indexes/eval_mc/musique_mc_rag \
  indexes/eval_mc/robustness_musique_clean \
  indexes/eval_mc/robustness_musique_noisy \
  indexes/eval_mc/robustness_musique_adversarial \
  -type f -size +95M -print
```

Si no aparece nada grande y quieren subir esos datos/índices al repo:

```bash
git add \
  data/eval_mc/musique_mc_rag \
  data/eval_mc/robustness_musique \
  indexes/eval_mc/musique_mc_rag \
  indexes/eval_mc/robustness_musique_clean \
  indexes/eval_mc/robustness_musique_noisy \
  indexes/eval_mc/robustness_musique_adversarial

git status --short
git commit -m "Agrego datasets e indices de evaluacion MuSiQue"
git push origin main
```

Si aparecen archivos mayores a 95 MB, conviene no subirlos a Git normal. En ese caso usar Git LFS, Drive, Hugging Face Hub o regenerarlos con scripts.

## 11. Archivos que no deben subirse

No subir:

```text
.env
tp_cientifico/
__pycache__/
*.pyc
logs/*.pid
```

El archivo `.env.example` sí puede subirse, siempre que no tenga una API key real.


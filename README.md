# NLPSURVEY

Repositorio de experimentos para el informe **Estrategias RAG adaptativas para razonamiento multi-hop**.

El informe editable esta en Overleaf y el PDF final versionado esta en `docs/informe/informe_final.pdf`.

Para saber exactamente que se entrega y que archivo respalda cada resultado, usar `docs/ENTREGA_FINAL.md` como inventario canonico. Este README solo describe la estructura general del repo y como orientarse.

## Alcance experimental

### Sistemas

| Sistema | Rol en el informe | Implementacion principal |
| --- | --- | --- |
| S0 | LLM directo, sin recuperacion | `modelos/s0/run_s0_direct.py`, `modelos/s0/parse_s0_outputs.py`, `modelos/s0/evaluate_s0.py`, `direct_llm.py` |
| S1 | RAG fijo con top-k constante | `modelos/s1/run_s1_mc_rag.py` |
| S2 | RAG adaptativo con decision previa de recuperar | `modelos/s2/run_s2_mc_real_adaptive.py` |
| S3 | Recuperacion activa tipo FLARE | `modelos/s3/run_s3_mc_flare_like.py` |
| S4 | Auditor factual exploratorio/post-hoc | `modelos/s4/`, `docs/experimentos/informe_s4_focus_musique_500.md` |
| S5 | Meta-router post-hoc/oracle sobre S0-S3 | `modelos/s5/meta_router/` |

S4 no se reporta como sistema competitivo principal de accuracy porque fue evaluado como auditor focalizado. En el informe queda como modulo exploratorio: aporta evidencia sobre verificacion factual, pero no reemplaza la tabla S0-S3.

### Modelos base

Las corridas finales del informe usan modelos compactos/costo-eficientes disponibles al momento de la corrida:

- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-4.1-mini`

El objetivo no fue comparar contra el ultimo frontier model, sino medir trade-offs de accuracy, costo y recuperacion en modelos de costo moderado.

### Datasets

| Dataset | Uso |
| --- | --- |
| `data/eval_mc/musique_mc_rag_500/` | Dataset principal MuSiQue-MC-500 |
| `data/eval_mc/hotpotqa_mc_rag_500/` | Validacion multi-dataset HotpotQA-MC-500 |
| `data/eval_mc/2wiki_mc_rag_500/` | Validacion multi-dataset 2Wiki-MC-500 |
| `data/eval_mc/robustness_musique/` | Robustez MuSiQue con corpus limpio, ruidoso y adversarial |

## Estructura

| Ruta | Contenido |
| --- | --- |
| `modelos/` | Implementaciones finales S0-S5. Ver `modelos/README.md`. |
| `data/eval_mc/` | Datasets finales usados para las corridas reportadas. |
| `outputs/eval_mc/` | Tablas, reportes y salidas citadas por el informe. |
| `logs/` | Logs finales de corridas largas. |
| `scripts/` | Scripts de reconstruccion de datasets, indices y corridas agregadas. |
| `evaluation/` | Utilidades conservadas para construir datasets/indices y agregar resultados finales. |
| `docs/` | Inventario de entrega, PDF final y notas interpretativas versionadas. |

## Archivos de raiz

| Archivo | Rol |
| --- | --- |
| `.env.example` | Plantilla de variables de entorno. No contiene credenciales. |
| `.gitignore` | Excluye caches, copias locales de Overleaf, zips y material intermedio. |
| `direct_llm.py` | Wrapper comun para llamadas a la API usado por varios sistemas. |
| `project_paths.py` | Rutas compartidas por scripts legacy y finales. |
| `requirements.txt` | Dependencias Python necesarias para reproducir/analizar. |

## Reproduccion rapida

Crear entorno e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las corridas completas consumen API y no deben lanzarse por accidente. Para reconstruir los artefactos agregados al informe, usar los scripts versionados:

```powershell
# Construccion/indices multi-dataset
bash scripts/step2_build_hotpotqa_mc500.sh
bash scripts/step3_build_hotpotqa_rag_index.sh
bash scripts/step6_build_2wiki_mc500.sh
bash scripts/step7_build_2wiki_rag_index.sh

# Corridas S0-S3 multi-dataset y agregacion
bash scripts/step5_full_hotpotqa_s0_s3.sh
bash scripts/step9_full_2wiki_s0_s3.sh
bash scripts/step11_run_musique_s0_and_final_aggregate.sh

# Robustez MuSiQue S0-S3
powershell -ExecutionPolicy Bypass -File scripts/run_musique_robustness_s0_s3.ps1
```

## Que no forma parte de la entrega

Los zips/exportaciones locales de Overleaf, carpetas temporales y smokes quedan fuera del entregable. Pueden servir como respaldo local, pero no deben citarse como fuente de resultados finales.

El informe fuente final esta cargado en Overleaf. En este repo se entrega solo el PDF final en `docs/informe/informe_final.pdf`.

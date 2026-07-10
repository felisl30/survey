# Modelos evaluados

Esta carpeta agrupa las implementaciones finales por sistema. Los scripts comparten utilidades de la raiz del repo, especialmente `direct_llm.py`, `project_paths.py` y los parsers/evaluadores de S0 cuando la salida mantiene el mismo formato multiple-choice.

| Sistema | Archivo principal | Dependencias compartidas | Rol en el informe |
| --- | --- | --- | --- |
| S0 | `s0/run_s0_direct.py` | `direct_llm.py`, `s0/parse_s0_outputs.py`, `s0/evaluate_s0.py` | Baseline sin recuperacion. |
| S1 | `s1/run_s1_mc_rag.py` | indice RAG, `s0/parse_s0_outputs.py`, `s0/evaluate_s0.py` | RAG fijo top-k. |
| S2 | `s2/run_s2_mc_real_adaptive.py` | indice RAG, `s0/parse_s0_outputs.py`, `s0/evaluate_s0.py` | RAG adaptativo con decision previa de recuperar. |
| S3 | `s3/run_s3_mc_flare_like.py` | indice RAG, `direct_llm.py`, `s0/parse_s0_outputs.py`, `s0/evaluate_s0.py` | Recuperacion activa tipo FLARE. |
| S4 | `s4/run_s4_fire_like.py` | modulos internos de `s4/`, outputs focalizados | Auditor factual exploratorio/post-hoc. |
| S5 | `s5/meta_router/run_s5_final_router.py` | outputs S0-S3, scripts de `s5/meta_router/` | Meta-router post-hoc/oracle. |

S1, S2 y S3 no tienen parsers propios porque escriben `raw_output` compatible con el parseo/evaluacion multiple-choice de S0. Eso evita duplicar logica de metricas y mantiene comparables las corridas del informe.

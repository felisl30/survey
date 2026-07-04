#!/usr/bin/env bash
set -uE -o pipefail

cd "$(dirname "$0")/.." || exit 1

log() {
  echo
  echo "== $* =="
}

warn() {
  echo "AVISO: $*"
}

run_optional() {
  echo
  echo "+ $*"
  "$@"
  local code=$?
  if [ "$code" -ne 0 ]; then
    warn "falló el comando opcional: $*"
    warn "se continúa para permitir diagnóstico parcial"
  fi
}

run_required() {
  echo
  echo "+ $*"
  "$@"
}

log "Proyecto NLP MuSiQue: reproducción de resultados principales"
echo "Directorio: $(pwd)"

if [ -d "tp_cientifico" ]; then
  # shellcheck disable=SC1091
  source tp_cientifico/bin/activate
else
  echo "ERROR: no existe el entorno tp_cientifico."
  echo "Crear con:"
  echo "  python3 -m venv tp_cientifico"
  echo "  source tp_cientifico/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  warn "no existe .env. Si se corren experimentos con API, crear .env desde .env.example."
fi

log "Python"
python --version
which python || true

log "Chequeo de imports"
run_required python - <<'PY'
import pandas as pd
import numpy as np
import sklearn
import openai
print("OK: imports principales funcionando")
PY

log "Chequeo de API key"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "OK: OPENAI_API_KEY está definida"
else
  warn "OPENAI_API_KEY no está definida. RUN_API=1 probablemente fallará."
fi

log "Chequeo de archivos clave"
missing=0
for f in \
  requirements.txt \
  .env.example \
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
    if [ -f "$f" ]; then
      echo "OK: $f"
    else
      echo "FALTA: $f"
      missing=1
    fi
  done

if [ "$missing" -eq 1 ]; then
  warn "faltan archivos clave. Algunas reproducciones pueden fallar."
  if [ "${STRICT:-0}" = "1" ]; then
    echo "STRICT=1 activo: se corta la ejecución."
    exit 1
  fi
fi

log "Verificación de sintaxis de scripts principales"
run_required python -m py_compile \
  evaluation/run_s1_mc_rag.py \
  evaluation/run_s2_mc_real_adaptive.py \
  evaluation/run_s3_mc_flare_like.py \
  s4_model_code/run_s4_fire_like.py \
  evaluation/meta_router/build_meta_router_table.py \
  evaluation/meta_router/analyze_oracle_router.py \
  evaluation/meta_router/run_s5_rule_based_router.py \
  evaluation/meta_router/run_s5_final_router.py

echo "OK: scripts principales compilan"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
  log "CHECK_ONLY=1: chequeo terminado sin correr análisis"
  exit 0
fi

if [ "${RUN_API:-0}" = "1" ]; then
  log "RUN_API=1: se vuelven a correr experimentos que pueden consumir API"

  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: RUN_API=1 requiere OPENAI_API_KEY definida en .env o en el entorno."
    exit 1
  fi

  chmod +x scripts/run_musique_model_grid_s0_s3.sh scripts/run_musique_robustness_s0_s3.sh scripts/run_s4_robustness_focus_rules.sh

  log "Grilla MuSiQue S0-S3"
  run_required bash scripts/run_musique_model_grid_s0_s3.sh

  log "Robustez MuSiQue S0-S3"
  run_required bash scripts/run_musique_robustness_s0_s3.sh

  log "S4 auditor factual focalizado"
  run_optional python evaluation/build_s4_robustness_focus_input.py
  run_required bash scripts/run_s4_robustness_focus_rules.sh
else
  log "RUN_API no activado"
  echo "Se saltean corridas que pueden consumir API."
  echo "Para correrlas:"
  echo "  RUN_API=1 bash scripts/reproducir_resultados_principales.sh"
fi

log "Análisis de grilla de modelos"
run_optional python evaluation/summarize_musique_model_grid_s0_s3.py
run_optional python evaluation/plot_model_grid_results.py

log "Análisis de robustez"
run_optional python evaluation/summarize_musique_robustness_s0_s3.py
run_optional python evaluation/analyze_musique_robustness_deep.py
run_optional python evaluation/export_musique_robustness_qualitative_examples.py

log "Análisis S4"
run_optional python evaluation/summarize_s4_robustness_focus.py

log "S5 Meta-Router"
run_optional python evaluation/meta_router/build_meta_router_table.py
run_optional python evaluation/meta_router/analyze_oracle_router.py
run_optional python evaluation/meta_router/run_s5_rule_based_router.py
run_optional python evaluation/meta_router/run_s5_final_router.py

log "Reportes principales esperados"
for f in \
  docs/experimentos/informe_robustez_musique.md \
  docs/experimentos/informe_s4_robustez_focus.md \
  docs/experimentos/informe_s5_meta_router.md \
  outputs/eval_mc/meta_router/s5_final_router_report.md
  do
    if [ -f "$f" ]; then
      echo "OK: $f"
    else
      echo "FALTA: $f"
    fi
  done

log "Estado de Git"
git status --short || true

log "Fin"
echo "Reproducción terminada. Revisar avisos anteriores si algún bloque opcional falló."

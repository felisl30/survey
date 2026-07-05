#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/natural_language_processing/trabajo_cientifico
source tp_cientifico/bin/activate

set -a
source .env
set +a

mkdir -p logs

LOG="logs/musique_500_full_$(date +%Y%m%d_%H%M%S).log"

echo "Log: $LOG"

pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File ./evaluation/run_musique_500_model_grid.ps1 \
  -Models "gpt-5-nano,gpt-5-mini,gpt-4.1-mini" \
  -Systems "s0,s1,s2,s3" \
  2>&1 | tee "$LOG"

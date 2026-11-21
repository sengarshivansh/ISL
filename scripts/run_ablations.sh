#!/usr/bin/env bash
# Runs every ablation in configs/ablations on top of configs/isign_base.yaml, then
# builds the results table. Re-running skips runs that already have results.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for f in configs/ablations/*.yaml; do
  name=$(basename "$f" .yaml)
  if [ -f "runs/$name/results.json" ]; then echo "skip $name (done)"; continue; fi
  $PY - "$f" <<'PY'
import sys, yaml
from islt.train import load_config, deep_update, train
cfg = load_config("configs/isign_base.yaml")
deep_update(cfg, yaml.safe_load(open(sys.argv[1])))
train(cfg)
PY
done
$PY -m islt.report runs --out paper/results_table.tex

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/sebi_compliance_agent/src"

mkdir -p learned_data learned_runs/run1

python - <<'PYCODE'
from pathlib import Path
from learned_pipeline.data_gen import generate_splits

generate_splits(
    out_dir=Path("learned_data"),
    n_train=40,
    n_val=10,
    n_test=10,
    min_chars=5000,
    max_chars=12000,
    max_depth_refs=5,
    seed=42,
)
print("data generation ok")
PYCODE

/usr/bin/time -f "TRAIN_TIME %e sec" python -m learned_pipeline.train \
  --data-root learned_data \
  --out-dir learned_runs/run1 \
  --epochs 12 \
  --batch-size 64 \
  --emb-dim 256 \
  --hidden-dim 192 \
  --dropout 0.2 \
  --lr 1e-3 \
  --patience 4
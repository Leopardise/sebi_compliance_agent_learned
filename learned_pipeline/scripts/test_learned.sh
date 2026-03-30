#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/sebi_compliance_agent/src"

/usr/bin/time -f "TEST_TIME %e sec" python -m learned_pipeline.test \
  --data-root learned_data \
  --checkpoint learned_runs/run1/model.pt \
  --split test \
  --out-json learned_runs/run1/test_metrics.json
#!/usr/bin/env bash
# Reproduce every result in reports/ from the two raw CSVs.
#
# Usage:
#   ./run_all.sh                       # expects CSVs in ~/Downloads
#   SIMULA_DATA_DIR=/path ./run_all.sh # or point it somewhere else
#
# Total runtime is ~15 minutes on an M-series Mac; step 03 dominates because the
# hashed-logistic baseline trains on a 1M x 2^20 sparse matrix.
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  # Pick an interpreter that actually has wheels for the scientific stack.
  # Bare `python3` on a recent macOS can be 3.14, which had no lightgbm/pandas
  # wheels at time of writing and falls back to a source build that fails.
  BOOTSTRAP=""
  for cand in python3.12 python3.11 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then BOOTSTRAP="$cand"; break; fi
  done
  if [ -z "$BOOTSTRAP" ]; then
    echo "No suitable python3 found (need 3.11-3.13)."; exit 1
  fi
  echo "Creating virtualenv with $BOOTSTRAP ($("$BOOTSTRAP" --version))..."
  "$BOOTSTRAP" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

# libomp is required by LightGBM on macOS: brew install libomp
"$PY" -c "import lightgbm" 2>/dev/null || {
  echo "LightGBM failed to import. On macOS run: brew install libomp"; exit 1;
}

run() {
  echo ""
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
  "$PY" "$1"
}

run scripts/01_eda.py
run scripts/02_signal_audit.py
run scripts/03_train.py          # must precede 04 and 07 (writes artifacts/)
run scripts/04_ranking.py
run scripts/05_coldstart.py
run scripts/06_drift.py
run scripts/07_serving_bench.py
run scripts/08_sample_output.py

echo ""
echo "Done. Reports written to reports/:"
ls -1 reports/*.md

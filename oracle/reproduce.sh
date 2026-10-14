#!/usr/bin/env bash
# One-command reproduction of the Paper-2 oracle (Gate-0 REPRO requirement).
# Creates a pinned venv, runs all oracle versions, compares to goldens with numeric tolerance.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1          # stabilize float reduction order across BLAS builds

PYBIN="${PYBIN:-python3}"
if [ ! -d .venv ]; then
  "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "== environment =="
python --version
python -c "import numpy; print('numpy', numpy.__version__)"

echo "== running oracle v2 (recovery + confound + interval) =="
python common_mode_oracle_v2.py

echo "== running oracle v3 (misspecification + interval-failure + rho-id) =="
python common_mode_oracle_v3.py

echo "== checking goldens (numeric tolerance) =="
python check_goldens.py

echo "ALL ORACLE CHECKS PASSED"

#!/usr/bin/env bash
# Detached launcher for the corrected ML-Useful full run.
set -euo pipefail
cd /home/nanyeon99/project/macroforecast
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
LOG="scripts/replication/ml_useful_2022/run_full.log"
nohup python3 scripts/replication/ml_useful_2022/run_full.py /tmp/ml_useful_data/2018-01.csv \
  > "$LOG" 2>&1 < /dev/null &
echo $! > scripts/replication/ml_useful_2022/run_full.pid
echo "PID=$(cat scripts/replication/ml_useful_2022/run_full.pid)"
echo "LOG=$LOG"

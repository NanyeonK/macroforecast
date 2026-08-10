#!/bin/bash
# Full monthly Table 5 (B3): 5 targets x 5 horizons = 25 cells.
# SAFE bounded parallelism: MAXJOBS=2 concurrent x n_cores=6 = 12 threads on 16 cores.
# NEVER raise these without checking load — 5x8 oversubscribed the box on 2026-07-25.
cd /home/nanyeon99/project/macroforecast || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/home/nanyeon99/project/macroforecast
CORES=6
MAXJOBS=2
LOG=/tmp/t5_orchestrator.log
echo "ORCHESTRATOR START $(date)" > "$LOG"

run_cell() {
  local t=$1 h=$2
  if [ -s "/tmp/table5_${t}_h${h}.json" ]; then
    echo "SKIP  $t h$h (json exists) $(date +%m-%d_%H:%M)" >> "$LOG"
    return 0
  fi
  echo "START $t h$h $(date +%m-%d_%H:%M)" >> "$LOG"
  python3 /tmp/run_table5.py "$t" "$h" "$CORES" > "/tmp/t5_${t}_h${h}.log" 2>&1
  echo "END   $t h$h rc=$? $(date +%m-%d_%H:%M)" >> "$LOG"
}

for t in IP UR INF SPREAD HOUST; do
  for h in 1 3 9 12 24; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
    run_cell "$t" "$h" &
  done
done
wait
echo "ALL_CELLS_DONE $(date)" >> "$LOG"

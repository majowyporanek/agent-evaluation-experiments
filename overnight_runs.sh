#!/bin/bash
# Overnight runs: 3 seed replications of the GSM8K calibration pilot.
# Each run: n=50 examples, Qwen2.5-0.5B-Instruct. ~60-90 min per run on CPU.

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

OUT_DIR=overnight_results
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/overnight.log"

echo "=== OVERNIGHT RUN STARTED $(date) ===" | tee -a "$LOG"
echo "Host: $(hostname), Python: $(python --version)" | tee -a "$LOG"
echo "Free memory:" | tee -a "$LOG"
free -h | tee -a "$LOG"
echo "" | tee -a "$LOG"

for seed in 0 1 2; do
    echo "--- seed=$seed START $(date) ---" | tee -a "$LOG"
    python -m experiments.run_pilot \
        --n 50 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --seed "$seed" \
        --out "$OUT_DIR/results_seed${seed}.csv" \
        2>&1 | tee -a "$LOG"
    echo "--- seed=$seed DONE $(date) ---" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== ALL RUNS COMPLETE $(date) ===" | tee -a "$LOG"

for csv in "$OUT_DIR"/results_seed*.csv; do
    echo "" | tee -a "$LOG"
    echo "### analysis of $csv ###" | tee -a "$LOG"
    python -m experiments.analyze --csv "$csv" --out "${csv%.csv}_scatter.png" 2>&1 | tee -a "$LOG"
done

echo "=== ANALYSIS COMPLETE $(date) ===" | tee -a "$LOG"

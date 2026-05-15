#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-glm-4.7-flash:latest}"
BACKEND="${BACKEND:-local}"
ENDPOINT="${ENDPOINT:-chat}"
STRUCTURED_OUTPUT="${STRUCTURED_OUTPUT:-auto}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_API_KEY_ENV="${OLLAMA_API_KEY_ENV:-OLLAMA_API_KEY}"
NUM_RUNS="${NUM_RUNS:-20}"
ICL_SIZE="${ICL_SIZE:-64}"
TEST_SIZE_PER_RUN="${TEST_SIZE_PER_RUN:-64}"
SEED="${SEED:-20260514}"

DATASETS=(
  random_demo_0.0
  random_demo_0.1
  random_demo_0.25
  random_demo_0.5
  sequential_demo_0.0
  sequential_demo_0.1
  sequential_demo_0.25
  sequential_demo_0.5
)

for dataset in "${DATASETS[@]}"; do
  python3 "$SCRIPT_DIR/run_ollama_symmetry_experiment.py" \
    --data-path "$REPO_ROOT/symmetry_demo_fraction_datasets/$dataset/symmetry_friendship_atoms.csv" \
    --output-dir "$SCRIPT_DIR/results/$dataset" \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --endpoint "$ENDPOINT" \
    --structured-output "$STRUCTURED_OUTPUT" \
    --ollama-url "$OLLAMA_URL" \
    --ollama-api-key-env "$OLLAMA_API_KEY_ENV" \
    --seed "$SEED" \
    --num-runs "$NUM_RUNS" \
    --icl-size "$ICL_SIZE" \
    --test-size-per-run "$TEST_SIZE_PER_RUN" \
    --temperature 0.0 \
    --sampling balanced
done

python3 "$SCRIPT_DIR/summarize_ollama_vs_tabpfn.py" \
  --llm-results-root "$SCRIPT_DIR/results" \
  --tabpfn-results-root "$REPO_ROOT/tfm_results_tabpfn" \
  --output-csv "$SCRIPT_DIR/ollama_vs_tabpfn_summary.csv"

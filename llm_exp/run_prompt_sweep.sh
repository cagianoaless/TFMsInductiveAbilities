#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-qwen3:8b}"
BACKEND="${BACKEND:-local}"
ENDPOINT="${ENDPOINT:-chat}"
STRUCTURED_OUTPUT="${STRUCTURED_OUTPUT:-auto}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_API_KEY_ENV="${OLLAMA_API_KEY_ENV:-OLLAMA_API_KEY}"
NUM_RUNS="${NUM_RUNS:-20}"
ICL_SIZE="${ICL_SIZE:-64}"
TEST_SIZE_PER_RUN="${TEST_SIZE_PER_RUN:-64}"
SEED="${SEED:-20260514}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/prompt_sweep}"

MODEL_SAFE="$(printf '%s' "$MODEL" | sed 's#[/:]#_#g')"

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

PROMPTS=(
  neutral
  rule_induction
  balanced_rule_induction
  reverse_pair_probe
  explicit_symmetry
)

for prompt in "${PROMPTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    python3 "$SCRIPT_DIR/run_ollama_symmetry_experiment.py" \
      --data-path "$REPO_ROOT/symmetry_demo_fraction_datasets/$dataset/symmetry_friendship_atoms.csv" \
      --output-dir "$RESULTS_ROOT/$MODEL_SAFE/$prompt/$dataset" \
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
      --sampling balanced \
      --prompt-template "$prompt"
  done
done

python3 "$SCRIPT_DIR/summarize_prompt_sweep.py" \
  --results-root "$RESULTS_ROOT/$MODEL_SAFE" \
  --output-csv "$RESULTS_ROOT/${MODEL_SAFE}_prompt_comparison.csv" \
  --aggregate-csv "$RESULTS_ROOT/${MODEL_SAFE}_prompt_comparison_by_prompt.csv" \
  --baseline-prompt neutral

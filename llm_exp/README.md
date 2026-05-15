# Ollama Symmetry Experiment

This folder evaluates local Ollama LLMs on the same symmetry datasets used for
the TabPFN comparison.

The runner does not send the whole dataset in one prompt. For each run it:

1. reads `symmetry_friendship_atoms.csv`;
2. samples an in-context-learning batch from `split == train`;
3. samples a test batch from `split == test`;
4. asks the Ollama model for JSON predictions;
5. repeats this over multiple random runs;
6. writes `metrics.csv`, `per_run_metrics.csv`, and `predictions.csv`.

By default, the prompt does **not** tell the model that the relation is
symmetric. This is intentional: the goal is to test whether the model can infer
the rule from examples. Use `--reveal-symmetry` only as a sanity-check upper
bound.

## Requirements

Ollama must be running locally:

```bash
ollama serve
```

The model must be available:

```bash
ollama pull glm-4.7-flash:latest
```

Python dependencies are only `pandas`.

## Run One Dataset

From the `experiment_2` folder:

```bash
python3 llm_exp/run_ollama_symmetry_experiment.py \
  --data-path symmetry_demo_fraction_datasets/random_demo_0.5/symmetry_friendship_atoms.csv \
  --output-dir llm_exp/results/random_demo_0.5 \
  --model glm-4.7-flash:latest \
  --num-runs 20 \
  --icl-size 64 \
  --test-size-per-run 64 \
  --temperature 0.0
```

## Run All Symmetry Datasets

```bash
bash llm_exp/run_all_ollama_symmetry.sh
```

You can override defaults without editing the script:

```bash
MODEL=glm-4.7-flash:latest NUM_RUNS=50 ICL_SIZE=96 TEST_SIZE_PER_RUN=64 bash llm_exp/run_all_ollama_symmetry.sh
```

## Compare With TabPFN

If TabPFN results are in `tfm_results_tabpfn`, the all-run script automatically
creates:

```text
llm_exp/ollama_vs_tabpfn_summary.csv
```

You can also run the summarizer manually:

```bash
python3 llm_exp/summarize_ollama_vs_tabpfn.py \
  --llm-results-root llm_exp/results \
  --tabpfn-results-root tfm_results_tabpfn \
  --output-csv llm_exp/ollama_vs_tabpfn_summary.csv
```

## Important Interpretation

This is not exactly the same learning protocol as TabPFN. TabPFN consumes all
training rows in one fit/predict call. The LLM consumes several sampled ICL
batches because the full table is too large for a local prompt. Therefore, the
main comparison should be interpreted as:

- TabPFN: full-table in-context tabular inference;
- Ollama LLM: sampled textual in-context inference;
- random-ID datasets: stricter test of symbolic symmetry;
- sequential-ID datasets: vulnerable to numeric/block leakage.

# Ollama Symmetry Experiment

This folder evaluates Ollama LLMs on the same symmetry datasets used for the
TabPFN comparison. It supports local Ollama models and Ollama Cloud models.

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

## Local Requirements

Ollama must be running locally:

```bash
ollama serve
```

The model must be available:

```bash
ollama pull glm-4.7-flash:latest
```

Python dependencies are only `pandas`.

## Ollama Cloud

There are two supported cloud modes.

The first mode uses your local Ollama installation as a proxy. Sign in once:

```bash
ollama signin
```

Then pull and use a cloud model through the normal local endpoint:

```bash
ollama pull gpt-oss:120b-cloud
MODEL=gpt-oss:120b-cloud bash llm_exp/run_all_ollama_symmetry.sh
```

The second mode calls Ollama Cloud directly via `https://ollama.com/api`.
Create an API key, export it, and run with `BACKEND=cloud`:

```bash
export OLLAMA_API_KEY=your_api_key
BACKEND=cloud MODEL=gpt-oss:120b bash llm_exp/run_all_ollama_symmetry.sh
```

In direct cloud mode the script calls `https://ollama.com/api/chat` and adds:

```text
Authorization: Bearer $OLLAMA_API_KEY
```

Do not commit API keys.

By default the runner uses `ENDPOINT=auto`:

- local normal models, such as `qwen3:8b`, use `/api/generate`;
- direct cloud models and local `*-cloud` models use `/api/chat`.

If you explicitly want the prompt-completion endpoint, set:

```bash
ENDPOINT=generate bash llm_exp/run_all_ollama_symmetry.sh
```

If you explicitly want the chat endpoint, set:

```bash
ENDPOINT=chat bash llm_exp/run_all_ollama_symmetry.sh
```

The runner also avoids `format=json` automatically for cloud models because
Ollama Cloud may not support structured outputs. The prompt still asks the
model to return JSON, and the parser extracts JSON from the text response.

## Run One Dataset

From the `experiment_2` folder:

```bash
python3 llm_exp/run_ollama_symmetry_experiment.py \
  --data-path symmetry_demo_fraction_datasets/random_demo_0.5/symmetry_friendship_atoms.csv \
  --output-dir llm_exp/results/random_demo_0.5 \
  --model glm-4.7-flash:latest \
  --backend local \
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

## Test Different Prompts

Available prompt templates are:

```text
neutral
rule_induction
balanced_rule_induction
reverse_pair_probe
explicit_symmetry
```

The most scientific baseline is `neutral`, because it does not reveal the rule.
`explicit_symmetry` is an upper-bound sanity check, not a fair rule-inference
test.

Run one prompt across all datasets:

```bash
PROMPT_TEMPLATE=balanced_rule_induction MODEL=qwen3:8b bash llm_exp/run_all_ollama_symmetry.sh
```

Run the full prompt sweep:

```bash
MODEL=qwen3:8b bash llm_exp/run_prompt_sweep.sh
```

This writes:

```text
llm_exp/prompt_sweep/qwen3_8b_prompt_comparison.csv
llm_exp/prompt_sweep/qwen3_8b_prompt_comparison_by_prompt.csv
```

The first file compares every `(dataset, prompt)` pair. The second averages
metrics by prompt and sorts prompts by mean accuracy and F1.

## Debug Invalid Outputs

If a run prints `accuracy=nan valid=0/64`, the Ollama call succeeded but the
model response could not be converted into labels. The runner now writes:

```text
invalid_responses.jsonl
```

inside the output directory whenever parsing fails or is partial. Inspect that
file to see the raw model response and parser error.

The parser accepts normal JSON, fenced JSON, id-to-label dictionaries, string
ids such as `"id=0"`, and simple line outputs such as `id=0 label=1`.

Direct Ollama Cloud example:

```bash
export OLLAMA_API_KEY=your_api_key
BACKEND=cloud MODEL=gpt-oss:120b NUM_RUNS=50 ICL_SIZE=96 TEST_SIZE_PER_RUN=64 bash llm_exp/run_all_ollama_symmetry.sh
```

## Compare With TabPFN

If TabPFN results are in `results/symmetry/demo_fraction`, the all-run script automatically
creates:

```text
llm_exp/ollama_vs_tabpfn_summary.csv
```

You can also run the summarizer manually:

```bash
python3 llm_exp/summarize_ollama_vs_tabpfn.py \
  --llm-results-root llm_exp/results \
  --tabpfn-results-root results/symmetry/demo_fraction \
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

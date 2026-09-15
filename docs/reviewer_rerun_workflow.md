# Reviewer-rerun workflow

The authoritative plan and live checklist are in `dev_log_rerun.md`. This page
contains the executable command sequence. Run commands from the repository root
on branch `reviewer-rerun`.

## Safety boundaries

- `snapshot`, `build`, `estimate`, `assemble`, `retry`, `validate`, and `render`
  are data stages and do not submit an OpenAI job.
- `preflight` makes one API request and requires `--confirm-api-call`.
- `submit` creates a paid Batch and requires the exact confirmation string
  `SUBMIT_PAID_BATCH`.
- No training command should be run before the storage and model-target gates.
- Never mix `src/baseline_evaluation.py` output with `src/evaluate_v2.py`
  output in a corrected results table.

The Batch/Responses implementation follows the official OpenAI API references:

- <https://developers.openai.com/api/reference/resources/batches>
- <https://developers.openai.com/api/reference/cli/resources/responses/methods/create>

The MLX-LM YAML fields follow the upstream example:

- <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/lora_config.yaml>

## 1. Preparation checks

Start from a clean clone of `reviewer-rerun`. The bootstrap is pinned to the
same `uv` and Python versions used to prepare and validate this branch. Install
`uv` outside the project virtual environment; a fresh clone does not yet have a
`.venv` directory. The versioned installer form follows the official Astral
installation documentation: <https://docs.astral.sh/uv/getting-started/installation/>.

```bash
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
./scripts/bootstrap_m4.sh
```

The expected `uv` output begins with `uv 0.9.18`; the script installs Python
3.13.2, synchronizes only the locked `rerun` dependency group, checks the lock,
runs all preparation tests and configuration validations, executes the three
no-download training dry-runs, and prints non-secret environment metadata.

The script fails before synchronization when the machine is not an Apple
M4-family Mac, the branch is not `reviewer-rerun`, the working tree is dirty,
or the `uv` version differs. It does not persist an environment report during
bootstrap, so it does not overwrite the committed originating-machine report
or dirty an otherwise clean clone. It makes no OpenAI request, downloads no
model weights, and starts no training or evaluation.

If local policy does not allow piping an installer into a shell, download and
inspect that exact versioned installer first or use another approved Astral
installation method while preserving `uv 0.9.18`.

## 2. Snapshot a 30-source data pilot

The command defaults to the frozen GSM8K revision
`740312add88f781978c0658806c59bc2815b9866` and validates that the complete
upstream train split still contains 7,473 rows before selecting the pilot:

```bash
.venv/bin/python -m src.openai_conversion_v2 snapshot \
  --output data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --limit 30
```

## 3. Build and inspect the pilot Batch input

```bash
.venv/bin/python -m src.openai_conversion_v2 build \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl

.venv/bin/python -m src.openai_conversion_v2 estimate \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl
```

Inspect the JSONL and manifest before any API request.

## 4. One-request teacher preflight

This is the first command that incurs API usage:

```bash
OPENAI_API_KEY=... .venv/bin/python -m src.openai_conversion_v2 preflight \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --output data/reviewer_rerun/batch_outputs/preflight.json \
  --confirm-api-call
```

Confirm the requested model, returned model, structured schema, three variants,
and output quality before proceeding.

## 5. Submit, inspect, and download the pilot Batch

```bash
OPENAI_API_KEY=... .venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --confirm SUBMIT_PAID_BATCH

OPENAI_API_KEY=... .venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json

OPENAI_API_KEY=... .venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs
```

`status` performs one retrieval and returns; it does not poll indefinitely.

## 6. Assemble and render the pilot

Replace `<batch-id>` with the ID recorded in the manifest:

```bash
.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --batch-output data/reviewer_rerun/batch_outputs/<batch-id>.output.jsonl \
  --canonical-output data/reviewer_rerun/canonical/pilot.jsonl \
  --audit-output data/reviewer_rerun/audits/pilot_assembly.jsonl

.venv/bin/python -m src.openai_conversion_v2 render \
  --input data/reviewer_rerun/canonical/pilot.jsonl \
  --output-root data/reviewer_rerun/pilot \
  --target-count 90 \
  --min-solution-chars 0
```

The render command writes a pairing report and fails on any mismatch between
the Socratic and non-Socratic arms.

## 7. Full data run

Do not start this stage until the pilot, storage review, model check, and API
volume review are complete. The full snapshot expects all 7,473 GSM8K training
rows, and the Batch builder creates 7,473 requests for 22,419 candidates.

The final renderer targets 21,250 accepted records after shared validation and
deduplication. If fewer survive, retry failed/rejected sources before rendering.

The offline full-volume input is created with:

```bash
.venv/bin/python -m src.openai_conversion_v2 snapshot \
  --output data/reviewer_rerun/source/gsm8k_train.jsonl

.venv/bin/python -m src.openai_conversion_v2 build \
  --source data/reviewer_rerun/source/gsm8k_train.jsonl \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl

.venv/bin/python -m src.openai_conversion_v2 estimate \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl
```

## 8. Training smoke runs

After installing the locked MLX stack and creating the paired data:

```bash
.venv/bin/python -m src.run_training run \
  --config configs/reviewer_rerun/qwen3_0.6b_socratic.yaml \
  --experiment-id qwen3-0.6b-socratic-smoke \
  --run-dir runs/reviewer_rerun/training/qwen3_0.6b_socratic_smoke \
  --smoke-iters 20 \
  --minimum-free-gib 5
```

Repeat for the non-Socratic Qwen configuration and the Llama configuration.
Do not pass `--skip-model-preflight` for an official smoke or full run.
The wrapper verifies each configured repository SHA, downloads that exact
snapshot, passes its local path to MLX-LM, and validates all seven target-module
suffixes before training.

## 9. Corrected evaluation

Use pinned dataset revisions for official runs. Example greedy command:

```bash
.venv/bin/python -m src.evaluate_v2 \
  --experiment-id qwen3-0.6b-base \
  --model mlx-community/Qwen3-0.6B-bf16 \
  --benchmark gsm8k \
  --mode greedy \
  --samples 1 \
  --seed 42 \
  --run-dir runs/reviewer_rerun/evaluation/qwen3_0.6b_base/gsm8k/greedy
```

For the timing-gated SC@5 protocol:

```bash
.venv/bin/python -m src.evaluate_v2 \
  --experiment-id qwen3-0.6b-base \
  --model mlx-community/Qwen3-0.6B-bf16 \
  --benchmark gsm8k \
  --mode self_consistency \
  --samples 5 \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --seed 42 \
  --run-dir runs/reviewer_rerun/evaluation/qwen3_0.6b_base/gsm8k/sc5
```

Use `--limit 20` only for a smoke run. Full reporting rejects partial benchmark
counts by default. The evaluator registry already pins GSM8K, MultiArith, and
SVAMP to the revisions in `configs/reviewer_rerun/protocol.yaml`; use
`--dataset-revision` only to reproduce an explicitly documented alternative.
The three frozen model identifiers are also resolved against their exact commit
SHAs and materialized locally before MLX-LM loads them. An unregistered remote
model requires an explicit 40-character `--model-revision`; local model paths
are content-hashed instead. The resulting identity is stored in the evaluation
manifest and each raw prediction row.

## 10. Reports

```bash
make rerun-reports
```

This creates the canonical accuracy, Wilson interval, resource, and
reproducibility outputs under `results/reviewer_rerun/`.

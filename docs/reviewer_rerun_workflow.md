# Reviewer-rerun workflow

The complete one-pass research-assistant runbook is
[`../RA_REVIEWER_RERUN_PROTOCOL.md`](../RA_REVIEWER_RERUN_PROTOCOL.md). The
authoritative scientific settings are in `configs/reviewer_rerun/protocol.yaml`,
and the rationale/live history is in `dev_log_rerun.md`. This page is a compact
command reference. Run commands from the repository root on branch
`reviewer-rerun`.

## Safety boundaries

- `snapshot`, `build`, `estimate`, `assemble`, `merge`, `retry`,
  `retry-rejected`, `validate`, and `render` are data stages and do not submit
  an OpenAI job.
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

Do not create `.env` for the bootstrap or offline preparation stages. The
reviewer template is `.env.reviewer_rerun.example`, and the corrected scripts
do not load it automatically. Only after the one-request API preflight has been
approved, copy it to the ignored `.env`, set its permissions to `600`, fill in
the required key locally, and export it into the current shell as documented
inside the template. Never capture the key in a log or evidence packet.

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
.venv/bin/python -m src.openai_conversion_v2 preflight \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --output data/reviewer_rerun/batch_outputs/preflight.json \
  --confirm-api-call
```

Confirm the requested model, returned model, structured schema, three variants,
and output quality before proceeding.

## 5. Submit, inspect, and download the pilot Batch

```bash
.venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --confirm SUBMIT_PAID_BATCH

.venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json

.venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs
```

`status` performs one retrieval and returns; it does not poll indefinitely.

## 6. Assemble and render the pilot

Use the Batch manifest so both the submitted input and downloaded output are
located and hash-checked without copying a dynamic Batch ID:

```bash
.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --batch-manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --canonical-output data/reviewer_rerun/canonical/pilot.jsonl \
  --audit-output data/reviewer_rerun/audits/pilot_assembly.jsonl

.venv/bin/python -m src.openai_conversion_v2 render \
  --input data/reviewer_rerun/canonical/pilot.jsonl \
  --output-root data/reviewer_rerun/pilot \
  --all-accepted
```

The render command writes a pairing report and fails on any mismatch between
the Socratic and non-Socratic arms.

## 7. Full data run

Do not start this stage until the pilot, storage review, model check, and API
volume review are complete. The full snapshot expects all 7,473 GSM8K training
rows, and the Batch builder creates 7,473 requests for 22,419 candidates.

The final renderer targets 21,250 accepted records after shared validation and
deduplication. If fewer survive, retry failed/rejected sources before rendering.
The exact transport-retry, canonical-merge, filtered-source regeneration, and
whole-source replacement commands are in sections 9-10 of the one-pass RA
protocol.

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

After installing the locked MLX stack and creating the paired data, plan or run
all three mandatory smoke jobs through the guarded matrix script:

```bash
./scripts/run_training_matrix.sh --smoke --mandatory --plan
caffeinate -s ./scripts/run_training_matrix.sh --smoke --mandatory --execute
```

Do not pass `--skip-model-preflight` for an official smoke or full run.
The wrapper verifies each configured repository SHA, downloads that exact
snapshot, passes its local path to MLX-LM, and validates all seven target-module
suffixes before training.

## 9. Corrected evaluation

Use pinned dataset revisions for official runs. Plan and execute the entire
mandatory greedy matrix with:

```bash
./scripts/run_evaluation_matrix.sh --full --greedy --mandatory --plan
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --full --greedy --mandatory --execute
```

For the timing-gated SC@5 protocol:

```bash
./scripts/run_evaluation_matrix.sh --full --sc5 --mandatory --plan
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --full --sc5 --mandatory --execute
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
reproducibility outputs under `results/reviewer_rerun/`. It fails unless all
three mandatory training runs and all 15 unique full greedy evaluations are
present. Use `make rerun-reports-partial` only for progress inspection.

# Reviewer rerun entrypoint

This page is the starting point for the post-review experiments. The submitted
paper and the original experiment workflow remain preserved in `README.md` and
`HANDOVER.md`; they are historical records and are not the runbook for the new
results.

All reviewer experiments must be run from the `reviewer-rerun` branch. New
artifacts belong under `data/reviewer_rerun/`, `runs/reviewer_rerun/`, and
`results/reviewer_rerun/` so that submitted results cannot be silently mixed
with corrected results.

## Documentation map

Read these files in order:

1. [`configs/reviewer_rerun/protocol.yaml`](configs/reviewer_rerun/protocol.yaml)
   is the scientific authority for frozen models, revisions, dataset splits,
   prompts, seeds, training settings, and reporting requirements.
2. [`docs/reviewer_rerun_handover.md`](docs/reviewer_rerun_handover.md) explains
   what the research assistant receives, what is still missing, the stop/go
   gates, and which artifacts must be returned.
3. [`docs/reviewer_rerun_workflow.md`](docs/reviewer_rerun_workflow.md) contains
   the executable workflow from a fresh M4 bootstrap through data generation,
   training, evaluation, and reporting.
4. [`dev_log_rerun.md`](dev_log_rerun.md) contains the full preparation plan,
   design decisions, safeguards, and chronological execution log.
5. [`results/reviewer_rerun/reproducibility/readiness_report.md`](results/reviewer_rerun/reproducibility/readiness_report.md)
   records the verified preparation state and remaining gates.
6. [`results/reviewer_rerun/reproducibility/reproducibility.md`](results/reviewer_rerun/reproducibility/reproducibility.md)
   is the generated reproducibility-note scaffold. It is incomplete until the
   official runs have populated their manifests.
7. [`docs/submission_provenance.md`](docs/submission_provenance.md) identifies
   the submitted manuscript and the Git state from which the reviewer lane was
   created.

If prose and executable configuration ever disagree, stop. Do not resolve the
disagreement by guessing: compare the protocol hash, inspect the Git history,
and record an explicit correction before running an experiment.

## Fresh-M4 starting point

Clone the dedicated branch, install the pinned `uv` release, and run the
guarded bootstrap:

```bash
git clone --branch reviewer-rerun --single-branch <repository-url> socratiq
cd socratiq
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
./scripts/bootstrap_m4.sh
```

The bootstrap must be run from a clean checkout on an Apple M4-family Mac. It
installs the pinned Python environment, checks the lock, runs the preparation
suite, validates all reviewer training configurations, and exercises the
no-download training dry-runs. It does not make an OpenAI API request, download
model weights, train a model, or start an evaluation.

After it succeeds, record the machine details and exact code identity before
any further action:

```bash
git status --short --branch
git rev-parse HEAD
shasum -a 256 configs/reviewer_rerun/protocol.yaml uv.lock
uv --version
.venv/bin/python --version
system_profiler SPHardwareDataType
sw_vers
df -h .
```

Send that output back for review. The bootstrap succeeding is a setup result,
not permission to cross the next cost or resource gate.

## Current state

As of 15 September 2026:

- the submitted state is preserved by tag `submitted-draft-baseline`;
- the isolated reviewer pipeline, frozen protocol, M4 bootstrap, and 56
  preparation tests are committed on `reviewer-rerun`;
- immutable metadata revisions are frozen for Qwen3-0.6B, Qwen3-1.7B,
  Llama-3.2-1B, GSM8K, MultiArith, and SVAMP;
- an offline 30-source teacher-request pilot and full-volume request input have
  been built and hashed on the preparation machine, but their raw JSONL files
  are intentionally excluded from Git;
- no paid teacher request, Batch submission, model-weight download, training
  run, or corrected benchmark evaluation has been performed; and
- the final RA command matrix for every official training and evaluation run
  still needs to be approved after the M4 bootstrap, pilot, model smoke tests,
  and timing measurements.

The current branch is therefore ready for a fresh-M4 environment check, not
for an unattended full experiment run.

## Legacy boundary

Do not use the following legacy paths to generate corrected reviewer results:

- `src/openai_conversion.py`;
- `src/baseline_evaluation.py`;
- the root `lora-config-run-*.yaml`, `qwen-1.7-lora-config-run-*.yaml`, or
  `llama-lora-config-run-*.yaml` files;
- `evaluation_summary.csv`; or
- results quoted in the historical `README.md` or `HANDOVER.md`.

Those files remain useful for reproducing and interpreting the submitted
draft. Reviewer runs use `src/openai_conversion_v2.py`, `src/evaluate_v2.py`,
`src/run_training.py`, and `configs/reviewer_rerun/`.

## Non-negotiable safety gates

- A present API key is not approval to spend. The one-request teacher preflight
  needs `--confirm-api-call`; each paid Batch needs the exact confirmation
  string documented in the workflow.
- Do not launch a paid full Batch until the one-request and 30-source pilots
  have been inspected and their measured cost and quality accepted.
- Do not start model downloads or training until M4 free storage, model load,
  LoRA target modules, short adapter save/reload, and estimated duration have
  been verified.
- Never tune filtering, prompts, checkpoints, or evaluation rules after seeing
  which arm produces the preferred conclusion.
- Never report a run without its raw predictions, manifest, code commit,
  protocol hash, model revision, seed, timing, memory, and adapter identity.

The exact next commands after the bootstrap are deliberately staged in
[`docs/reviewer_rerun_workflow.md`](docs/reviewer_rerun_workflow.md). Stop at
each stated gate and return the evidence requested by the handover.

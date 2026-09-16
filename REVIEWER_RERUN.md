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

1. [`RA_REVIEWER_RERUN_PROTOCOL.md`](RA_REVIEWER_RERUN_PROTOCOL.md) is the
   complete one-pass operator protocol to give the research assistant together
   with handoff tag `reviewer-rerun-ra-handoff-v2`.
2. [`configs/reviewer_rerun/protocol.yaml`](configs/reviewer_rerun/protocol.yaml)
   is the scientific authority for frozen models, revisions, dataset splits,
   prompts, seeds, training settings, and reporting requirements.
3. [`.env.reviewer_rerun.example`](.env.reviewer_rerun.example) is the separate
   secret and cache-location template for the experiment machine. Do not use
   the legacy `.env.example` for corrected runs.
4. [`docs/reviewer_rerun_handover.md`](docs/reviewer_rerun_handover.md) explains
   what the research assistant receives, what is still missing, the stop/go
   gates, and which artifacts must be returned.
5. [`docs/reviewer_rerun_workflow.md`](docs/reviewer_rerun_workflow.md) contains
   the executable workflow from a fresh M4 bootstrap through data generation,
   training, evaluation, and reporting.
6. [`dev_log_rerun.md`](dev_log_rerun.md) contains the full preparation plan,
   design decisions, safeguards, and chronological execution log.
7. [`docs/reviewer_rerun_generation_v3.md`](docs/reviewer_rerun_generation_v3.md)
   records the active v3 generation contract, exact offline hashes, and the
   preserved failed-v2 boundary.
8. [`results/reviewer_rerun/reproducibility/readiness_report_v3.md`](results/reviewer_rerun/reproducibility/readiness_report_v3.md)
   records the current verified preparation state and remaining gates. The
   older `readiness_report.md` remains the pre-pilot historical snapshot.
9. [`results/reviewer_rerun/reproducibility/reproducibility.md`](results/reviewer_rerun/reproducibility/reproducibility.md)
   is the generated reproducibility-note scaffold. It is incomplete until the
   official runs have populated their manifests.
10. [`docs/submission_provenance.md`](docs/submission_provenance.md) identifies
   the submitted manuscript and the Git state from which the reviewer lane was
   created.

If prose and executable configuration ever disagree, stop. Do not resolve the
disagreement by guessing: compare the protocol hash, inspect the Git history,
and record an explicit correction before running an experiment.

## Fresh-M4 starting point

Clone the dedicated branch, install the pinned `uv` release, and run the
guarded bootstrap:

```bash
git clone --branch reviewer-rerun --single-branch \
  https://github.com/jakeogrady/socratiq.git socratiq
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

Capture that output in the evidence record. Under the one-pass RA protocol, the
RA may continue when the M4/storage checks pass and the next paid/resource stage
is already authorized; otherwise they must stop at the failed gate.

Do not create or populate `.env` for the bootstrap. Immediately before the
first explicitly approved API stage, copy `.env.reviewer_rerun.example` to
`.env`, restrict its permissions, fill in the required key locally, and source
it as documented inside the template. The corrected scripts do not
automatically read dotenv files.

## Current state

As of 16 September 2026:

- the submitted state is preserved by tag `submitted-draft-baseline`;
- the isolated reviewer pipeline, protocol 1.1, M4 bootstrap, and 77
  preparation tests are committed on `reviewer-rerun`;
- immutable metadata revisions are frozen for Qwen3-0.6B, Qwen3-1.7B,
  Llama-3.2-1B, GSM8K, MultiArith, and SVAMP;
- a paid `matched-pairs-v2` preflight and 30-request Batch completed, but its
  74/90 filter acceptance failed the 86/90 gate; the complete v2 record is
  preserved and no v2 full Batch was submitted;
- the stricter `matched-pairs-v3` preflight passed and its paid 30-request pilot
  Batch completed 30/30 requests with no API failures; a separately authorized
  two-request retry completed 2/2, and the merged pilot passes at 90/90 with
  zero filter/deduplication rejection and a passed matched-pair audit;
- the full 7,473-request v3 Batch has not been submitted and remains behind its
  own cost-authorization gate;
- no model-weight download, training run, or corrected benchmark evaluation
  has been performed; and
- the complete mandatory and optional command matrices, strict completion
  validator, retry/replacement flow, and return checklist are packaged in the
  one-pass RA protocol.

The branch is ready to hand over as a staged end-to-end protocol. It is not an
unconditional unattended job: each in-document cost, quality, storage, and
scientific gate still applies.

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

The complete sequence is in
[`RA_REVIEWER_RERUN_PROTOCOL.md`](RA_REVIEWER_RERUN_PROTOCOL.md). The RA should
capture evidence at every gate, continue through passing authorized gates, and
return the complete packet at the end or immediately after a stop condition.

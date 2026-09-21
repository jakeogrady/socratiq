# RA instructions: clean greedy and SC@5 evaluation rerun

Date: 21 September 2026

## Objective

Run a completely fresh evaluation matrix from the already trained final LoRA
adapters. Produce new greedy and SC@5 results for all eight model/condition
combinations on GSM8K, MultiArith, and SVAMP.

Do not regenerate the synthetic dataset and do not retrain any adapter. The two
corrected defects are in answer evaluation, after training:

- the primary checker now accepts the last `####`-marked numeric answer even if
  text such as `?` follows it; and
- numeric normalization no longer raises `decimal.InvalidOperation` for
  integers at or above `10^28`.

Do not use or modify `runs/reviewer_rerun/evaluation/`. The clean runner writes
to new, isolated directories.

## Frozen clean-run settings

- Primary scorer: `submitted-last-marked-decimal-v1`.
- Greedy: one sample, temperature 0, top-p 1, top-k 0.
- SC@5: five samples, temperature 0.7, top-p 0.95, top-k 20.
- Maximum new tokens: 512.
- Base seed: 42; per-sample seeds are derived deterministically.
- GSM8K: four fixed training demonstrations and all 1,319 test rows.
- MultiArith: zero-shot and all 180 test rows.
- SVAMP: zero-shot and all 300 pinned test rows.
- Qwen thinking: disabled.
- Full output: `runs/reviewer_rerun/evaluation_clean_v1/`.
- Timing-smoke output: `runs/reviewer_rerun/evaluation_clean_v1_smoke/`.

The evaluator records its own SHA-256 and the scorer ID in the configuration
hash. A source-code change therefore prevents unsafe resume into an existing
run directory.

## 1. Update and verify the repository

Run from the existing M4 repository clone:

```sh
git fetch origin
git switch reviewer-rerun
git pull --ff-only origin reviewer-rerun
git status --short
git rev-parse HEAD
```

The handoff message states the expected commit. Stop if `git status --short`
shows an unexpected tracked modification. Ignored model caches, adapters, and
run outputs are expected to remain local.

Do not create or copy an `.env` file. No OpenAI API credential is required for
training or evaluation.

## 2. Run the offline readiness checks

```sh
SOCRATIQ_UV_BIN="$(command -v uv)" make rerun-check
```

This must pass before evaluation. It performs no OpenAI request, training, or
model generation.

## 3. Verify the final adapter weights

```sh
shasum -a 256 -c configs/reviewer_rerun/final_adapter_sha256.txt
```

All five entries must report `OK`. Stop if any adapter is missing or has a
different hash. Do not substitute an intermediate checkpoint.

## 4. Inspect the clean plans

```sh
./scripts/run_clean_evaluation_matrix.sh \
  --full --greedy --include-qwen-1.7b --plan

./scripts/run_clean_evaluation_matrix.sh \
  --full --sc5 --include-qwen-1.7b --plan
```

Each command must print exactly 24 `PLAN:` lines. Every full-run path must begin
with `runs/reviewer_rerun/evaluation_clean_v1/`. Stop if a plan points to the
historical `runs/reviewer_rerun/evaluation/` directory.

## 5. Run timing smokes with the final adapters

The smoke matrix uses 20 rows per benchmark and writes only to the isolated
smoke root.

```sh
caffeinate -s ./scripts/run_clean_evaluation_matrix.sh \
  --smoke --greedy --include-qwen-1.7b --execute

caffeinate -s ./scripts/run_clean_evaluation_matrix.sh \
  --smoke --sc5 --include-qwen-1.7b --execute
```

Both commands are resumable by running the identical command again. Do not
change a decoding option or source file while a matrix is in progress.

Confirm that 48 smoke manifests exist after both matrices finish:

```sh
find runs/reviewer_rerun/evaluation_clean_v1_smoke \
  -name manifest.json -type f | wc -l
```

The expected result is `48`. Review the smoke durations and free disk space
before starting the full matrix. The previous runs project approximately 6.3
hours for greedy and 31.3 hours for SC@5, but allow 40-48 hours in total because
stochastic generations and greedy tie-breaks can increase runtime.

## 6. Run the complete fresh greedy matrix

```sh
caffeinate -s ./scripts/run_clean_evaluation_matrix.sh \
  --full --greedy --include-qwen-1.7b --execute
```

This creates 24 new greedy evaluations. If interrupted, run the exact same
command again; completed example IDs are skipped only when the complete
configuration, evaluator-code hash, and scorer identity match.

## 7. Run the complete fresh SC@5 matrix

```sh
caffeinate -s ./scripts/run_clean_evaluation_matrix.sh \
  --full --sc5 --include-qwen-1.7b --execute
```

This creates 24 new SC@5 evaluations. It performs five primary generations per
benchmark item and may perform an additional greedy generation to break a tied
vote. Resume an interruption only with the identical command.

## 8. Verify completion and generate reports

There must be 48 completed full-run manifests across greedy and SC@5:

```sh
find runs/reviewer_rerun/evaluation_clean_v1 \
  -name manifest.json -type f | wc -l
```

The expected result is `48`.

Generate the new report packet:

```sh
.venv/bin/python -m src.summarize_results \
  --evaluation-root runs/reviewer_rerun/evaluation_clean_v1 \
  --training-root runs/reviewer_rerun/training \
  --output-root results/reviewer_rerun/evaluation_clean_v1 \
  --require-mandatory-matrix
```

Then independently reparse every raw response, verify every prediction hash,
and calculate paired comparisons separately for greedy and SC@5:

```sh
.venv/bin/python -m src.rescore_predictions \
  --evaluation-root runs/reviewer_rerun/evaluation_clean_v1 \
  --output-root results/reviewer_rerun/evaluation_clean_v1_scoring_audit
```

The audit command deliberately refuses to overwrite an existing output root.
If an earlier audit attempt exists, preserve it and select a new suffixed output
directory rather than deleting evidence.

Expected final counts:

- 48 evaluation runs;
- 28,784 benchmark decisions across both modes;
- 24 greedy prediction files;
- 24 SC@5 prediction files;
- 42 paired comparison rows: seven declared comparisons, three benchmarks,
  and two decoding modes.

## 9. Package the evidence for return

From the repository root:

```sh
tar -czf reviewer-rerun-clean-v1-results.tar.gz \
  runs/reviewer_rerun/evaluation_clean_v1 \
  runs/reviewer_rerun/evaluation_clean_v1_smoke \
  results/reviewer_rerun/evaluation_clean_v1 \
  results/reviewer_rerun/evaluation_clean_v1_scoring_audit \
  runs/reviewer_rerun/training/*/manifest.json \
  runs/reviewer_rerun/training/*/adapter_manifest.json

shasum -a 256 reviewer-rerun-clean-v1-results.tar.gz \
  > reviewer-rerun-clean-v1-results.tar.gz.sha256
```

Return both the archive and its `.sha256` file. Retain the unpacked M4 run
directories until the archive has been transferred, verified, and backed up.

Do not include `.env`, API credentials, Hugging Face tokens, model-cache
credentials, or unrelated user files.

## Stop conditions

Stop and report the exact command and error before continuing if any of the
following occurs:

- the checkout does not match the handoff commit;
- a final adapter checksum fails;
- a plan points to a historical evaluation directory;
- a manifest reports a scorer other than
  `submitted-last-marked-decimal-v1`;
- an evaluation configuration-hash mismatch is reported;
- `decimal.InvalidOperation` appears;
- a completed full benchmark has fewer than 1,319 GSM8K, 180 MultiArith, or
  300 SVAMP rows;
- report generation records an evaluation or training error; or
- the M4 runs critically low on free disk space.

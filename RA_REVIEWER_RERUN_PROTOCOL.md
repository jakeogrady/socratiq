# Research-assistant protocol: reviewer rerun

This is the single operational document to give the research assistant together
with repository tag `reviewer-rerun-ra-handoff-v2`. It covers the complete
mandatory reviewer experiment from a fresh Apple M4 Mac through the return of
the raw evidence packet. Supporting rationale remains in
[`dev_log_rerun.md`](dev_log_rerun.md); the submitted/legacy workflow must not
be used for these results.

The RA may work through all phases in one handoff. Passing a checkpoint means
continue to the next phase; it does not require an intermediate response to the
authors. Stop and report only when a stated gate fails, a monetary authorization
is absent, or a scientific setting would have to change.

## 1. Authorization record

Before giving this protocol to the RA, the project owner must record these
decisions in the accompanying message or lab record. Do not edit the frozen
protocol YAML to record them.

| Decision | Required record |
|---|---|
| OpenAI account/project | Account or project name, never the API key |
| One-request preflight | Authorized: yes/no |
| 30-request pilot Batch | Authorized: yes/no |
| Full 7,473-request Batch | Authorized: yes/no and maximum acceptable cost |
| Retry Batches | Maximum requests or monetary ceiling |
| Mandatory matrix | Three training and 15 greedy evaluation runs: yes/no |
| SC@5 | Authorized only if the smoke timing estimate is acceptable: yes/no |
| Optional Qwen3-1.7B | Authorized: yes/no |
| Optional OOD benchmark | Deferred until mandatory work is complete |

An available API key is not monetary authorization. If a required authorization
or cost ceiling has not been supplied, stop before that paid stage while
continuing any remaining offline work that does not depend on it.

## 2. Frozen scope

The mandatory experiment is:

| Family | Condition | Train | GSM8K | MultiArith | SVAMP |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | base | no | greedy | greedy | greedy |
| Qwen3-0.6B | Socratic | yes | greedy | greedy | greedy |
| Qwen3-0.6B | non-Socratic | yes | greedy | greedy | greedy |
| Llama-3.2-1B | base | no | greedy | greedy | greedy |
| Llama-3.2-1B | Socratic | yes | greedy | greedy | greedy |

This means three mandatory LoRA training runs and 15 mandatory full benchmark
runs. The Qwen3-0.6B Socratic and non-Socratic training configurations are
matched except for dataset and adapter paths. Llama uses the same full Socratic
synthetic dataset and evaluation protocol.

Optional work, in priority order, is:

1. SC@5 for the mandatory conditions, only after the measured smoke timing is
   accepted;
2. Qwen3-1.7B Socratic and non-Socratic training plus base/tuned evaluations;
3. a harder or out-of-distribution benchmark, which is not implemented in this
   handoff and must not delay the mandatory matrix.

The scientific authority is
[`configs/reviewer_rerun/protocol.yaml`](configs/reviewer_rerun/protocol.yaml),
whose expected SHA-256 is:

```text
a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb
```

Do not modify prompts, filters, seeds, benchmark splits, revisions, answer
extraction, LoRA settings, or checkpoint selection after seeing results.

## 3. Prohibited legacy paths

Do not use these for corrected results:

- `src/openai_conversion.py`;
- `src/baseline_evaluation.py`;
- root `lora-config-run-*.yaml`, `qwen-1.7-lora-config-run-*.yaml`, or
  `llama-lora-config-run-*.yaml` files;
- `evaluation_summary.csv`; or
- accuracy values in the historical `README.md` and `HANDOVER.md`.

The corrected commands are `src/openai_conversion_v2.py`,
`src/run_training.py`, `src/evaluate_v2.py`, `src/summarize_results.py`, and the
two matrix scripts under `scripts/`.

## 4. Clone and verify the handoff

Run all commands from the repository root. Use the tagged handoff, not an
unreviewed moving branch:

```bash
git clone --branch reviewer-rerun --single-branch \
  https://github.com/jakeogrady/socratiq.git socratiq
cd socratiq
git fetch --tags origin
git checkout reviewer-rerun
git pull --ff-only origin reviewer-rerun
git rev-parse HEAD
git rev-list -n 1 reviewer-rerun-ra-handoff-v2
git status --short --branch
```

The two commit IDs must match and the checkout must be clean. Record both IDs.
Do not continue from a detached commit, another branch, or a dirty clone.

Install the pinned `uv` and run the guarded bootstrap:

```bash
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
./scripts/bootstrap_m4.sh
```

The bootstrap requires an Apple M4-family chip, branch `reviewer-rerun`, a clean
checkout, `uv 0.9.18`, and Python 3.13.2. It runs the preparation tests,
configuration checks, and no-download training plans. It makes no OpenAI
request, downloads no model weights, and starts no training or evaluation.

Record the M4 environment in a new file; do not overwrite the originating M2
report:

```bash
.venv/bin/python -m src.run_training environment \
  --output results/reviewer_rerun/reproducibility/environment_m4_preflight.json

shasum -a 256 configs/reviewer_rerun/protocol.yaml uv.lock \
  results/reviewer_rerun/reproducibility/environment_m4_preflight.json
df -h .
```

Expected lock SHA-256:

```text
cea213bade4a75241aa328d902b99e443d8847f5b30d9fa834a3eccf42ff5517
```

Gate: the environment report must identify an M4-family Mac, the expected Git
branch/commit, Python 3.13.2, and the locked packages. The repository volume
must have at least 40 GiB free; 60 GiB is preferred. If external storage is
needed, place the clone itself on that volume and set `HF_HOME` there before
the first dataset/model download.

## 5. Secrets

Do not create `.env` during bootstrap. Immediately before the approved API
preflight:

```bash
cp .env.reviewer_rerun.example .env
chmod 600 .env
```

Fill the key locally, then load it into that terminal:

```bash
set -a
. ./.env
set +a
```

The corrected scripts do not automatically load dotenv files. Never print or
return the real key. `HF_TOKEN` is optional because all frozen MLX mirrors are
public. Never set the bootstrap bypass variables for an official run.

## 6. Reproduce the offline request artifacts

Create the pinned full source snapshot and a separate 30-source pilot:

```bash
.venv/bin/python -m src.openai_conversion_v2 snapshot \
  --output data/reviewer_rerun/source/gsm8k_train.jsonl

.venv/bin/python -m src.openai_conversion_v2 snapshot \
  --output data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --limit 30
```

Expected source counts are 7,473 full and 30 pilot. Expected normalized JSONL
hashes from preparation are:

```text
full   2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5
pilot  aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f
```

Build and estimate the pilot request file:

```bash
.venv/bin/python -m src.openai_conversion_v2 build \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl

.venv/bin/python -m src.openai_conversion_v2 estimate \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl
```

Expected pilot request input:

```text
requests: 30
candidates: 90
bytes: 105,405
SHA-256: dbb92eb9f0a8bf7d54d7f3dd004ec6dfdb7cc14781324dc39760fa233b91f958
```

Gate: stop if a count or hash differs. Preserve the generated manifest and
report the mismatch without rebuilding under a changed revision or prompt.

## 7. One-request teacher preflight

This is the first paid request and must be authorized in the handoff record:

```bash
.venv/bin/python -m src.openai_conversion_v2 preflight \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --output data/reviewer_rerun/batch_outputs/preflight.json \
  --confirm-api-call
```

Inspect the response locally. Gate requirements:

- requested and returned model are exactly `gpt-5-mini-2025-08-07`;
- exactly three canonical variations validate;
- every synthetic problem contains a direct mathematical question ending in
  `?`;
- every variation has two to six nonredundant solution steps;
- each guiding question is separate and ends with `?`;
- each reasoning field is declarative and remains coherent without the guiding
  question, and contains 60--300 characters;
- the final answer is exactly `#### N` with one positive integer;
- no explicit arithmetic equality is incorrect;
- input/output token usage is present and within the supplied cost ceiling.

If the dated model is unavailable or a different model is returned, stop. Do
not remove the dated snapshot, permit fallback, or select a substitute.

## 8. Paid 30-source pilot Batch

Only run this after the pilot Batch authorization is recorded:

```bash
.venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --confirm SUBMIT_PAID_BATCH
```

The manifest now contains the Batch ID. Do not submit the same input again.
Check status periodically; each command performs one retrieval and returns:

```bash
.venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json
```

When the manifest reports `completed`, download and assemble using the manifest
so the submitted input and downloaded output hashes are verified automatically:

```bash
.venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs

.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train_pilot.jsonl \
  --batch-manifest data/reviewer_rerun/batch_inputs/gsm8k_train_pilot.manifest.json \
  --canonical-output data/reviewer_rerun/canonical/pilot_initial.jsonl \
  --audit-output data/reviewer_rerun/audits/pilot_initial.jsonl

.venv/bin/python -m src.openai_conversion_v2 validate \
  --input data/reviewer_rerun/canonical/pilot_initial.jsonl

.venv/bin/python -m src.openai_conversion_v2 render \
  --input data/reviewer_rerun/canonical/pilot_initial.jsonl \
  --output-root data/reviewer_rerun/pilot \
  --all-accepted
```

Inspect:

- `data/reviewer_rerun/canonical/pilot_initial.manifest.json`;
- `data/reviewer_rerun/audits/pilot_initial.jsonl`;
- `data/reviewer_rerun/pilot/manifests/dataset_manifest.json`;
- `data/reviewer_rerun/pilot/manifests/pairing_report.json`;
- samples from both rendered arms.

Gate: no returned-model mismatch, pairing status `passed`, identical example ID
sequences across arms, guiding questions absent only from the non-Socratic arm,
and at least 86 of 90 candidates accepted by the frozen filters. If fewer than
86 pass, the observed acceptance rate is below that required to reach 21,250
from 22,419 full candidates; stop and report without changing the filters.

## 9. Full teacher Batch

Build and inspect the full input before the paid submission:

```bash
.venv/bin/python -m src.openai_conversion_v2 build \
  --source data/reviewer_rerun/source/gsm8k_train.jsonl \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl

.venv/bin/python -m src.openai_conversion_v2 estimate \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl
```

Expected full request input:

```text
requests: 7,473
candidates: 22,419
bytes: 25,975,624
SHA-256: ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359
```

Only submit if the full-Batch authorization and cost ceiling were supplied and
the measured preflight/pilot usage fits that ceiling:

```bash
.venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full.manifest.json \
  --confirm SUBMIT_PAID_BATCH
```

Use one-shot status checks until a terminal state:

```bash
.venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full.manifest.json
```

After `completed`:

```bash
.venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs

.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train.jsonl \
  --batch-manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full.manifest.json \
  --canonical-output data/reviewer_rerun/canonical/full_initial.jsonl \
  --audit-output data/reviewer_rerun/audits/full_initial.jsonl
```

If the assembly manifest has no `retry_custom_ids`, merge the single initial
output into the stable downstream path:

```bash
.venv/bin/python -m src.openai_conversion_v2 merge \
  --input data/reviewer_rerun/canonical/full_initial.jsonl \
  --output data/reviewer_rerun/canonical/full_merged.jsonl
```

If it lists failed or missing IDs and retries are authorized, build retry 01:

```bash
.venv/bin/python -m src.openai_conversion_v2 retry \
  --original-input data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl \
  --assembly-manifest data/reviewer_rerun/canonical/full_initial.manifest.json \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.jsonl
```

Submit, status-check, download, and assemble the retry exactly as follows:

```bash
.venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.manifest.json \
  --confirm SUBMIT_PAID_BATCH

.venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.manifest.json

.venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs

.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train.jsonl \
  --batch-manifest data/reviewer_rerun/batch_inputs/gsm8k_train_full_retry01.manifest.json \
  --canonical-output data/reviewer_rerun/canonical/full_retry01.jsonl \
  --audit-output data/reviewer_rerun/audits/full_retry01.jsonl
```

The retry assembly is scoped only to IDs actually submitted in that retry.
If another retry is necessary and authorized, use its previous retry input and
assembly manifest to build `retry02`; never resubmit a manifest that already
contains a Batch ID.

Merge every successful initial/transport-retry canonical file once:

```bash
.venv/bin/python -m src.openai_conversion_v2 merge \
  --input data/reviewer_rerun/canonical/full_initial.jsonl \
  --input data/reviewer_rerun/canonical/full_retry01.jsonl \
  --output data/reviewer_rerun/canonical/full_merged.jsonl
```

Omit nonexistent retry inputs or add later retry files with another `--input`.
The merge fails on duplicate example or source/variant IDs.

## 10. Shared filtering and matched rendering

Validate the merged generated records, then run a feasibility render using all
accepted records:

```bash
.venv/bin/python -m src.openai_conversion_v2 validate \
  --input data/reviewer_rerun/canonical/full_merged.jsonl

.venv/bin/python -m src.openai_conversion_v2 render \
  --input data/reviewer_rerun/canonical/full_merged.jsonl \
  --output-root data/reviewer_rerun/full_feasibility01 \
  --all-accepted
```

Read `full_feasibility01/manifests/dataset_manifest.json`. If
`accepted_before_selection` is at least 21,250, proceed to the final render.

If fewer than 21,250 survive and filtered-source retries are authorized, build
one request for every source group with a rejected variant:

```bash
.venv/bin/python -m src.openai_conversion_v2 retry-rejected \
  --original-input data/reviewer_rerun/batch_inputs/gsm8k_train_full.jsonl \
  --canonical-input data/reviewer_rerun/canonical/full_merged.jsonl \
  --rejection-audit data/reviewer_rerun/full_feasibility01/audits/rejections.jsonl \
  --output data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.jsonl
```

Submit/status/download/assemble that file using the same guarded commands and
names:

```bash
.venv/bin/python -m src.openai_conversion_v2 submit \
  --input data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.jsonl \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.manifest.json \
  --confirm SUBMIT_PAID_BATCH

.venv/bin/python -m src.openai_conversion_v2 status \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.manifest.json

.venv/bin/python -m src.openai_conversion_v2 download \
  --manifest data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.manifest.json \
  --output-dir data/reviewer_rerun/batch_outputs

.venv/bin/python -m src.openai_conversion_v2 assemble \
  --source data/reviewer_rerun/source/gsm8k_train.jsonl \
  --batch-manifest data/reviewer_rerun/batch_inputs/gsm8k_train_filtered_retry01.manifest.json \
  --canonical-output data/reviewer_rerun/canonical/full_filtered_retry01.jsonl \
  --audit-output data/reviewer_rerun/audits/full_filtered_retry01.jsonl
```

Wait for `completed` before download, as with every Batch. If this replacement
Batch itself has transport/response failures, apply the ordinary transport
retry procedure to its input and assembly manifest before the replacement
merge. Then replace each affected source group as a whole with its later
generation:

```bash
.venv/bin/python -m src.openai_conversion_v2 merge \
  --input data/reviewer_rerun/canonical/full_merged.jsonl \
  --input data/reviewer_rerun/canonical/full_filtered_retry01.jsonl \
  --output data/reviewer_rerun/canonical/full_merged_filtered01.jsonl \
  --replace-sources-from-later
```

Run another `--all-accepted` feasibility render on that new merged file. Each
replacement and input hash is recorded in the merge manifest. Repeat only
within the authorized retry ceiling. Never retain a mix of old and regenerated
variants from the same source.

Once at least 21,250 records survive, perform the frozen exact-size render from
the latest merged canonical file:

```bash
.venv/bin/python -m src.openai_conversion_v2 render \
  --input data/reviewer_rerun/canonical/full_merged.jsonl \
  --output-root data/reviewer_rerun \
  --target-count 21250 \
  --min-solution-chars 120 \
  --max-solution-chars 2000 \
  --ngram-size 5 \
  --jaccard-threshold 0.85 \
  --validation-fraction 0.10 \
  --seed 42
```

If a filtered replacement was used, substitute the latest
`full_merged_filteredNN.jsonl` in that final command.

Gate requirements:

- `accepted_rows` is exactly 21,250;
- pairing status is `passed` for train and validation;
- Socratic and non-Socratic split counts and example ID sequences match;
- questions, declarative reasoning, answers, source IDs, variant IDs, and split
  assignments match across arms;
- only guiding-question lines are removed from the non-Socratic completion;
- train and validation source-ID sets do not overlap;
- every output file path and SHA-256 is present in the dataset manifest.

After the dataset is complete, remove the API key from the shell:

```bash
unset OPENAI_API_KEY
```

## 11. Inspect the exact experiment plans

The matrix scripts print commands without executing by default:

```bash
make rerun-plan-matrices

./scripts/run_training_matrix.sh --smoke --mandatory --plan
./scripts/run_evaluation_matrix.sh --smoke --greedy --mandatory --plan
./scripts/run_evaluation_matrix.sh --smoke --sc5 --mandatory --plan
```

Gate: the mandatory full plan must show exactly three training commands and 15
greedy evaluation commands. Do not use `--skip-model-preflight`,
`--skip-revision-resolution`, `--enable-thinking`, `--limit`, or altered
decoding settings for official full runs.

## 12. Model, training, and evaluation smoke gate

The smoke artifacts are deliberately outside the official training/evaluation
roots so they cannot enter final tables.

Run all three mandatory 20-iteration training smokes:

```bash
caffeinate -s ./scripts/run_training_matrix.sh \
  --smoke --mandatory --execute
```

This downloads only the exact frozen model snapshots, validates all intended
LoRA targets, trains, saves final smoke adapters, hashes them, and records
resource manifests.

Run 20 examples from each of the 15 mandatory conditions/benchmarks:

```bash
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --smoke --greedy --mandatory --execute
```

Exercise SC@5 timing on the same smoke ranges:

```bash
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --smoke --sc5 --mandatory --execute
```

Generate smoke-only reports:

```bash
make rerun-smoke-reports
```

Gate requirements:

- exact model revisions resolve and load;
- all seven target-module suffixes exist in every intended layer;
- every smoke adapter saves and reloads;
- no configuration-hash mismatch occurs on evaluation resume;
- prompts and extracted answers pass manual spot checks;
- peak memory leaves a safe margin on the M4;
- projected full training/evaluation duration fits the available schedule.

If an official or smoke training process is interrupted, do not use MLX-LM's
adapter-only resume as an equivalent continuation: it does not restore the
optimizer or scheduler state. Preserve the incomplete directory under
`runs/reviewer_rerun/failed/training/`, record the cause, and restart that run
from iteration zero in its clean canonical path.

Evaluations are safely resumable. Re-run the identical matrix command; existing
per-example rows are skipped only when the configuration hash matches.

## 13. Mandatory official training

Run the three required full LoRA jobs sequentially:

```bash
caffeinate -s ./scripts/run_training_matrix.sh \
  --full --mandatory --execute
```

The runner skips a training directory only when its manifest status is exactly
`completed`. It stops on any incomplete existing directory instead of silently
resuming or overwriting it.

Expected official adapter directories:

```text
runs/reviewer_rerun/training/qwen3_0.6b_socratic/adapter
runs/reviewer_rerun/training/qwen3_0.6b_non_socratic/adapter
runs/reviewer_rerun/training/llama3.2_1b_socratic/adapter
```

Each run manifest must report status `completed`, the expected model revision,
the dataset split hashes, trainable/total parameters, elapsed seconds, peak MLX
memory, peak process memory, log hash, final adapter path/bytes/hash, and
selected `adapters.safetensors` path/bytes/hash.

## 14. Mandatory official greedy evaluation

Run all 15 required corrected evaluations:

```bash
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --full --greedy --mandatory --execute
```

The frozen evaluator uses:

- GSM8K: four shots from pinned train indices 0, 1, 2, and 3; 1,319 test rows;
- MultiArith: zero-shot; 180 test rows;
- SVAMP: zero-shot; 300 test rows;
- one greedy sample, temperature 0, maximum 512 new tokens;
- base seed 42 with deterministic per-example seed derivation;
- model chat template with Qwen thinking disabled;
- last strict marked decimal extraction from `#### <number>`; and
- exact normalized decimal comparison, with missing/malformed answers counted
  incorrect.

The base-model numbers must be generated here. Do not copy any submitted
baseline. Re-running the identical command safely completes missing rows.

## 15. Optional SC@5 and Qwen3-1.7B

Only run SC@5 when explicitly authorized after reviewing the smoke projection:

```bash
caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --full --sc5 --mandatory --execute
```

SC@5 uses five samples, temperature 0.7, top-p 0.95, top-k 20, maximum 512 new
tokens, and the frozen greedy-then-numeric/lexical tie policy.

Only run the optional Qwen3-1.7B pair when explicitly authorized and the
mandatory matrix is secure:

```bash
caffeinate -s ./scripts/run_training_matrix.sh \
  --full --include-qwen-1.7b --execute

caffeinate -s ./scripts/run_evaluation_matrix.sh \
  --full --greedy --include-qwen-1.7b --execute
```

These commands include the mandatory conditions; already completed training
runs are skipped, and completed evaluation rows are safely reused. If SC@5 is
also authorized for Qwen3-1.7B, use `--sc5 --include-qwen-1.7b`.

Do not add an OOD benchmark during this handoff. That requires a separately
frozen dataset revision, schema, prompt, expected denominator, and tests.

## 16. Final reports and hard completion gate

Progress reports may be generated at any time without claiming completion:

```bash
make rerun-reports-partial
```

After the mandatory matrix is complete, run the strict final report:

```bash
make rerun-reports
```

The strict command fails unless it finds all three completed mandatory training
manifests, all 15 unique mandatory greedy evaluations, and every full benchmark
denominator. It generates:

```text
results/reviewer_rerun/summaries/summary.csv
results/reviewer_rerun/summaries/resource_summary.csv
results/reviewer_rerun/reproducibility/reproducibility.md
results/reviewer_rerun/report_manifest.json
```

Gate: `report_manifest.json` must contain a passed matrix validation with three
mandatory training runs and 15 mandatory evaluation runs, no evaluation errors,
and hashes for every generated report.

## 17. Final evidence and return packet

Return the following without the Hugging Face model cache, `.venv`, or `.env`:

1. exact Git commit and handoff tag;
2. protocol and lock hashes;
3. `environment_m4_preflight.json`;
4. source, Batch, assembly, merge, filtering, split, and pairing manifests;
5. all rejection/error audits and paid Batch manifests;
6. the final canonical data and both rendered training arms;
7. all mandatory training directories, including logs, manifests, adapter
   configuration, and adapter weights;
8. all official evaluation directories, including raw `predictions.jsonl` and
   manifests;
9. smoke manifests/reports and any failure evidence;
10. final result/resource CSVs, reproducibility note, and report manifest;
11. the authorization/cost record, actual API usage/cost, and wall-clock notes;
12. SHA-256 hashes for every transferred archive.

Before transfer, record:

```bash
git rev-parse HEAD
git status --short --branch
shasum -a 256 configs/reviewer_rerun/protocol.yaml uv.lock
df -h .
```

Generated large artifacts are intentionally Git-ignored. Transfer them through
the project-approved secure storage mechanism, retaining their relative paths.
Never commit `.env`, credentials, raw secrets, or model caches. Do not delete
the M4 copy until the return archive has been verified at its destination.

## 18. Stop conditions

Stop and return the accumulated evidence if:

- the handoff tag, branch, protocol hash, lock hash, model revision, dataset
  revision, source count, or prepared input hash differs;
- free storage drops below the gate or the M4 runs out of memory;
- the teacher snapshot is unavailable or the returned model differs;
- a paid stage is not authorized or would exceed its ceiling;
- pairing fails or the two arms differ beyond guiding-question removal;
- fewer than 21,250 records survive within the authorized retries;
- a model target, adapter save/reload, or training run fails;
- an official training directory is incomplete;
- an evaluation reports a configuration-hash mismatch;
- a full benchmark denominator differs from 1,319, 180, or 300;
- strict final reporting does not confirm three training and 15 greedy
  evaluation runs; or
- any protocol change appears necessary after results have been inspected.

Do not substitute a model, loosen validation, reduce the official benchmark,
change the seed, choose a checkpoint based on test accuracy, or use a legacy
script to bypass a stop condition.

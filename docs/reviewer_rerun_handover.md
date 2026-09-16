# Reviewer-rerun operational handover

This handover is for the research assistant running the post-review experiments
on a separate Apple M4 Mac. It is intentionally separate from the historical
root `HANDOVER.md`, which describes the repository and artifacts associated
with the submitted draft.

Give the RA
[`../RA_REVIEWER_RERUN_PROTOCOL.md`](../RA_REVIEWER_RERUN_PROTOCOL.md) as the
single end-to-end operator document. This handover and
[`reviewer_rerun_workflow.md`](reviewer_rerun_workflow.md) remain supporting
references.

## Objective

The mandatory reviewer work is:

1. construct one canonical synthetic dataset that renders into exactly matched
   Socratic and non-Socratic training arms;
2. fine-tune Qwen3-0.6B on both arms under the same LoRA configuration;
3. fine-tune one non-Qwen model, currently frozen as Llama-3.2-1B, on the full
   Socratic arm;
4. evaluate the required base and tuned conditions on GSM8K, MultiArith, and
   SVAMP with one corrected evaluation implementation;
5. collect enough provenance and resource evidence to reproduce every reported
   number; and
6. keep all corrected results separate from the submitted results.

Qwen3-1.7B matched-arm training is optional and must not delay the mandatory
Qwen3-0.6B and Llama work. A harder out-of-distribution benchmark is also
optional and comes only after the mandatory matrix is complete.

## Source-of-truth order

Use this authority order when checking a run:

1. [`../configs/reviewer_rerun/protocol.yaml`](../configs/reviewer_rerun/protocol.yaml)
   for frozen scientific settings;
2. [`../RA_REVIEWER_RERUN_PROTOCOL.md`](../RA_REVIEWER_RERUN_PROTOCOL.md) for
   the complete operational order and exact matrix commands;
3. the tagged `reviewer-rerun` Git commit and its tracked code;
4. the run-specific manifest and raw outputs;
5. [`reviewer_rerun_workflow.md`](reviewer_rerun_workflow.md) for command
   reference and gate sequencing;
6. [`../dev_log_rerun.md`](../dev_log_rerun.md) for rationale and history.

Do not silently edit a YAML file, prompt, seed, model name, revision, output
path, or sample count to make a command run. Stop and report the exact error
and proposed correction.

## What is already prepared

The branch contains:

- a frozen protocol with immutable model and benchmark revisions;
- paired-record construction, shared filtering, deterministic deduplication,
  source-group splitting, and matched arm rendering;
- separated OpenAI snapshot, build, estimate, preflight, submit, status,
  download, scoped assembly, transport retry, filtered-source retry,
  replacement merge, validation, and render stages;
- guarded MLX-LM training with exact model materialization, target-module
  checks, resource manifests, and adapter hashing;
- a corrected resumable evaluator with fixed few-shot provenance, strict answer
  extraction, greedy and SC@5 modes, deterministic seeds, and raw predictions;
- guarded mandatory/optional training and evaluation matrix runners;
- result summarization with actual denominators, Wilson confidence intervals,
  and a hard three-training/15-evaluation completion gate;
- five reviewer LoRA configurations: mandatory Qwen3-0.6B matched arms,
  mandatory Llama-3.2-1B Socratic, and optional Qwen3-1.7B matched arms;
- a fresh-M4 bootstrap pinned to `uv 0.9.18` and Python 3.13.2; and
- 74 passing preparation tests on the branch-preparation machine.

The immutable mandatory model targets are:

| Condition | Model repository | Revision | Stored weights |
|---|---|---|---|
| Qwen3-0.6B | `mlx-community/Qwen3-0.6B-bf16` | `42096995f6402fde107068cf530136fe64b604f8` | bfloat16 |
| Llama-3.2-1B | `mlx-community/Llama-3.2-1B-Instruct-MLXTuned` | `7247cd8c176bbc558293c9b4750e9f97b5beb319` | bfloat16 |

The optional Qwen target is
`mlx-community/Qwen3-1.7B-4bit` at
`3b1b1768f8f8cf8351c712464f906e86c2b8269e`.

## What the Git clone does not contain

Normal Git history intentionally excludes large and run-specific artifacts.
A fresh clone does not contain:

- raw GSM8K source JSONL or teacher Batch input JSONL;
- paid teacher responses or the final canonical synthetic dataset;
- rendered full Socratic and non-Socratic training splits;
- downloaded model snapshots or Hugging Face caches;
- trained adapters or checkpoints;
- official raw benchmark predictions; or
- completed resource measurements and final result tables.

The tracked JSON manifests and hashes describe prepared inputs but do not
replace those raw artifacts. Rebuild or transfer an artifact only through a
documented path, verify its hash, and never copy it into a legacy output
directory.

## Environment and secrets

Use [`../.env.reviewer_rerun.example`](../.env.reviewer_rerun.example), not the
legacy root `.env.example`. The template contains only supported variables:

- `OPENAI_API_KEY` is required only for an explicitly approved teacher API
  operation;
- `HF_TOKEN` is optional because the frozen MLX model repositories are public;
  and
- `HF_HOME` is an optional, commented cache relocation for an approved storage
  plan.

The reviewer scripts do not automatically load `.env`. Do not create the local
file during bootstrap. Immediately before the approved one-request preflight:

```bash
cp .env.reviewer_rerun.example .env
chmod 600 .env
```

Fill in the required value locally, then export the file into that terminal:

```bash
set -a
. ./.env
set +a
```

An empty `OPENAI_API_KEY` causes the paid stages to fail closed. Never print the
key, include it in captured command output, commit `.env`, or return it in an
evidence packet. Do not use the bootstrap test-bypass variables during an
official run.

## First M4 action

Use a fresh, single-branch clone. Do not copy an existing `.venv` from another
Mac.

```bash
git clone --branch reviewer-rerun --single-branch \
  https://github.com/jakeogrady/socratiq.git socratiq
cd socratiq
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
./scripts/bootstrap_m4.sh
```

The script should refuse the wrong architecture, a non-M4 host, the wrong
branch, a dirty checkout, or the wrong `uv` version. A successful run may
create the ignored `.venv`, but must not create API jobs, download model
weights, train adapters, run benchmarks, or change tracked files.

Immediately after bootstrap, capture:

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

Expected protocol SHA-256:
`be66a73d43e4ceb34cd599fdbec9a4cd60dec6324d394a79292873fe9624d924`.

Expected lock SHA-256:
`cea213bade4a75241aa328d902b99e443d8847f5b30d9fa834a3eccf42ff5517`.

If this evidence satisfies the host/storage gate, continue through the
one-pass protocol. Bootstrap success alone does not authorize an API call or
model download; those authorizations must be present in the handoff record.

## Open gates after bootstrap

The following gates remain open until evidence from the M4 exists:

| Gate | Evidence required before proceeding |
|---|---|
| Host and storage | M4 identity, unified memory, macOS version, and free disk space; at least 40 GiB free, with 60 GiB preferred, or an approved external-cache plan |
| Teacher availability | One explicitly approved API preflight returning the exact requested teacher model and valid structured output |
| Data quality and cost | Inspected 30-source pilot, rejection/failure counts, measured token usage, and accepted projected full cost |
| Qwen model smoke | Exact pinned snapshot, target-module check, 20-iteration train, adapter save/reload, and small resumable evaluation |
| Llama model smoke | Equivalent exact-revision load, target-module, adapter save/reload, and evaluation evidence |
| Duration | Measured smoke throughput and time estimate for full training and all evaluations |
| Self-consistency | Decision based on measured evaluation time; the mandatory greedy matrix must not be delayed |
| Full command matrix | Prepared in the one-pass protocol; its printed plan must contain three mandatory training and 15 mandatory greedy evaluation commands |

Cross one gate at a time. Preserve command output and error logs even when a
gate fails; failures are useful planning evidence and must not be hidden by an
unrecorded workaround.

## Staged execution boundaries

The operational sequence is:

1. bootstrap the M4 and capture the environment evidence;
2. reproduce and inspect the offline 30-source request input;
3. confirm recorded authorization for the one-request paid teacher preflight;
4. inspect returned model identity, schema compliance, content quality, tokens,
   and measured cost;
5. confirm recorded authorization for the 30-source Batch pilot;
6. assemble, validate, and render the pilot into matched arms;
7. perform exact-model load and 20-iteration training/evaluation smoke tests;
8. use the measured storage, memory, and duration data to verify that the
   prepared official matrix remains feasible;
9. confirm recorded authorization for full teacher generation;
10. freeze and hash the final canonical data before any official training;
11. run the mandatory training and greedy evaluation matrix; and
12. generate reports exclusively from run manifests and raw predictions.

The exact end-to-end commands are documented in
[`../RA_REVIEWER_RERUN_PROTOCOL.md`](../RA_REVIEWER_RERUN_PROTOCOL.md). Matrix
scripts plan by default and require `--execute`; SC@5 still depends on the M4
smoke timing decision.

## Mandatory experiment identities

The minimum scientific comparison is:

| Family | Condition | Train? | GSM8K | MultiArith | SVAMP |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | base | no | evaluate | evaluate | evaluate |
| Qwen3-0.6B | Socratic | yes | evaluate | evaluate | evaluate |
| Qwen3-0.6B | non-Socratic | yes | evaluate | evaluate | evaluate |
| Llama-3.2-1B | base | no | evaluate | evaluate | evaluate |
| Llama-3.2-1B | Socratic | yes | evaluate | evaluate | evaluate |

That is three mandatory training runs and 15 mandatory greedy benchmark runs.
Qwen3-1.7B and SC@5 are later additions only if the resource and timing gates
allow them. Base numbers for the corrected table must be rerun with the same
evaluator; do not copy them from the submitted `evaluation_summary.csv`.

## Artifact and reporting discipline

Every official run needs a unique experiment directory. Do not reuse an output
directory for a changed command or configuration. Preserve at minimum:

- the exact command and UTC start/end timestamps;
- Git commit and working-tree state;
- protocol and lock hashes;
- machine model, chip, unified memory, macOS, Python, and package versions;
- exact model repository and resolved revision, or local snapshot hash;
- precision/quantization and LoRA configuration;
- trainable and total parameter counts;
- peak memory, wall-clock training time, and evaluation time;
- adapter/checkpoint path, byte size, and SHA-256;
- dataset input paths, record counts, split identities, and hashes;
- seeds and decoding settings;
- raw per-example predictions and answer-extraction outcomes; and
- final manifest, summaries, warnings, and failures.

Keep secrets out of terminal transcripts and files. Record that an API key was
available, never the key itself.

## Return packet

At each stop/go point, accumulate evidence containing:

1. the command that was run and its complete non-secret stdout/stderr;
2. `git rev-parse HEAD` and `git status --short --branch`;
3. paths and SHA-256 hashes for newly produced artifacts;
4. relevant manifests and logs;
5. disk space before and after any material download or training run;
6. elapsed time and peak memory where applicable; and
7. a short note stating whether the gate passed, failed, or needs a decision.

Return the accumulated packet after the complete run, or immediately if a stop
condition prevents further progress.

Large raw artifacts should remain outside ordinary Git history. Compact
manifests, hashes, tables, and documentation can be committed after review.
Agree on the transfer mechanism and destination before moving model snapshots,
adapters, datasets, or raw predictions between machines.

## Stop conditions

Stop and report before proceeding if any of the following occurs:

- the branch, commit, protocol hash, lock hash, model revision, or dataset
  revision differs from the expected value;
- the checkout becomes dirty for an unexplained reason;
- free storage falls below the gate;
- the requested and returned teacher model identities differ;
- the paired renderer produces unequal arms or mismatched source IDs;
- an exact model snapshot or intended LoRA target cannot be verified;
- an adapter cannot be reloaded;
- an evaluation resume reports a configuration-hash mismatch;
- a benchmark denominator differs from 1,319 GSM8K, 180 MultiArith, or 300
  SVAMP test examples; or
- a change to the protocol appears necessary after results have been viewed.

Do not substitute legacy scripts, loosen validation, switch models, reduce the
official sample, or delete evidence to get past a stop condition.

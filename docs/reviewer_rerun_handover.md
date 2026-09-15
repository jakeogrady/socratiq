# Reviewer-rerun operational handover

This handover is for the research assistant running the post-review experiments
on a separate Apple M4 Mac. It is intentionally separate from the historical
root `HANDOVER.md`, which describes the repository and artifacts associated
with the submitted draft.

Start at [`../REVIEWER_RERUN.md`](../REVIEWER_RERUN.md), then keep this page
beside [`reviewer_rerun_workflow.md`](reviewer_rerun_workflow.md) while running
the staged checks.

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
2. the current `reviewer-rerun` Git commit and its tracked code;
3. the run-specific manifest and raw outputs;
4. [`reviewer_rerun_workflow.md`](reviewer_rerun_workflow.md) for commands and
   gate sequencing;
5. [`../dev_log_rerun.md`](../dev_log_rerun.md) for rationale and history.

Do not silently edit a YAML file, prompt, seed, model name, revision, output
path, or sample count to make a command run. Stop and report the exact error
and proposed correction.

## What is already prepared

The branch contains:

- a frozen protocol with immutable model and benchmark revisions;
- paired-record construction, shared filtering, deterministic deduplication,
  source-group splitting, and matched arm rendering;
- separated OpenAI snapshot, build, estimate, preflight, submit, status,
  download, assemble, retry, validate, and render stages;
- guarded MLX-LM training with exact model materialization, target-module
  checks, resource manifests, and adapter hashing;
- a corrected resumable evaluator with fixed few-shot provenance, strict answer
  extraction, greedy and SC@5 modes, deterministic seeds, and raw predictions;
- result summarization with actual denominators and Wilson confidence
  intervals;
- five reviewer LoRA configurations: mandatory Qwen3-0.6B matched arms,
  mandatory Llama-3.2-1B Socratic, and optional Qwen3-1.7B matched arms;
- a fresh-M4 bootstrap pinned to `uv 0.9.18` and Python 3.13.2; and
- 56 passing preparation tests on the branch-preparation machine.

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

## First M4 action: bootstrap only

Use a fresh, single-branch clone. Do not copy an existing `.venv` from another
Mac.

```bash
git clone --branch reviewer-rerun --single-branch <repository-url> socratiq
cd socratiq
curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
./scripts/bootstrap_m4.sh
```

The script should refuse the wrong architecture, a non-M4 host, the wrong
branch, a dirty checkout, or the wrong `uv` version. A successful run may
create the ignored `.venv`, but must not create API jobs, download model
weights, train adapters, run benchmarks, or change tracked files.

Immediately after bootstrap, capture and return:

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
`9b2ebcff1d00309bec5355351c3b9b96de8153d513a7bb06b264adeaf3ac811a`.

Expected lock SHA-256:
`cea213bade4a75241aa328d902b99e443d8847f5b30d9fa834a3eccf42ff5517`.

Stop after returning the evidence. A clean bootstrap is the first decision
point; it is not approval for API use or model downloads.

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
| Full command matrix | Final experiment IDs, output paths, order, and resume commands approved after smoke measurements |

Cross one gate at a time. Preserve command output and error logs even when a
gate fails; failures are useful planning evidence and must not be hidden by an
unrecorded workaround.

## Staged execution boundaries

The operational sequence is:

1. bootstrap the M4 and return the environment evidence;
2. reproduce and inspect the offline 30-source request input;
3. obtain explicit approval for the one-request paid teacher preflight;
4. inspect returned model identity, schema compliance, content quality, tokens,
   and measured cost;
5. obtain separate approval for the 30-source Batch pilot;
6. assemble, validate, and render the pilot into matched arms;
7. perform exact-model load and 20-iteration training/evaluation smoke tests;
8. use the measured storage, memory, and duration data to approve the complete
   official command matrix;
9. obtain separate approval for full teacher generation;
10. freeze and hash the final canonical data before any official training;
11. run the mandatory training and greedy evaluation matrix; and
12. generate reports exclusively from run manifests and raw predictions.

The commands currently available for these stages are documented in
[`reviewer_rerun_workflow.md`](reviewer_rerun_workflow.md). The full official
matrix is intentionally not represented as a single unattended script yet:
its scheduling and self-consistency scope depend on the M4 smoke measurements.

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

At each stop/go point, return a small evidence packet containing:

1. the command that was run and its complete non-secret stdout/stderr;
2. `git rev-parse HEAD` and `git status --short --branch`;
3. paths and SHA-256 hashes for newly produced artifacts;
4. relevant manifests and logs;
5. disk space before and after any material download or training run;
6. elapsed time and peak memory where applicable; and
7. a short note stating whether the gate passed, failed, or needs a decision.

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

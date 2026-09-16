# Reviewer Rerun Development Log

## Status

- Branch: `reviewer-rerun`
- Baseline branch: `main`
- Baseline commit: `cfc17c444878291c927b3b5d5642032ff905317c`
- Work started: 2026-09-14
- Submitted-paper baseline tag: `submitted-draft-baseline`
- Full paid data generation started: no
- Full model training started: no
- Full benchmark evaluation started: no

This file is the execution plan and running audit log for the reviewer-requested experiments. The submitted repository state remains available at the baseline commit. New results must not be mixed with historical results unless they were produced by the same frozen rerun protocol.

## Objective

Prepare and execute a controlled reviewer rerun that supports three claims:

1. A matched Socratic versus non-Socratic ablation on Qwen3-0.6B.
2. Transfer of the full Socratic dataset to a small non-Qwen model.
3. Reproducible reporting of dataset provenance, prompts, LoRA settings, hardware use, timing, memory, adapter artifacts, seeds, and benchmark evaluation.

The governing experimental rule is:

> Every comparison must use one frozen dataset, a frozen training configuration, one frozen evaluator, and artifacts traceable through manifests and hashes.

## Scope

### Required

- Build a new canonical synthetic dataset from GSM8K source problems.
- Render exactly matched Socratic and non-Socratic training arms.
- Target 21,250 accepted canonical examples to retain the submitted experiment scale where validation permits.
- Train Qwen3-0.6B on both arms using identical corrected LoRA settings.
- Train one Llama-3.2-1B model on the full Socratic arm.
- Evaluate base and fine-tuned conditions on GSM8K, MultiArith, and SVAMP.
- Preserve raw per-example predictions.
- Generate resource and reproducibility reports from manifests.
- Calculate Wilson confidence intervals using actual evaluation counts.

### Preferred after required work

- Train and evaluate matched Qwen3-1.7B Socratic and non-Socratic adapters.

### Optional after all reviewer requirements

- Evaluate one harder or out-of-distribution benchmark, with GSM-Hard as the preferred first choice.

### Explicitly out of scope for preparation

- Recreating every historical hyperparameter sweep.
- Recovering lost historical adapters.
- Replacing MLX-LM as the training backend.
- Reorganizing the entire repository into a new framework.
- Launching a paid OpenAI Batch job without a pilot and volume estimate.
- Launching full training before dataset, model, and evaluation smoke tests pass.
- Deleting existing user files to obtain disk space.

## Phase 0: Preserve the submitted state

### Tasks

- [x] Record the baseline commit.
- [x] Create and switch to `reviewer-rerun`.
- [x] Create `submitted-draft-baseline` at the baseline commit.
- [x] Record the submitted PDF SHA-256 and path in a provenance note.
- [x] Confirm the only pre-existing worktree modification and keep it out of reviewer-rerun commits.
- [x] Record the initial branch and worktree state.

### Gate

- The submitted state has an immutable tag.
- The rerun branch is active.
- Pre-existing user changes are documented and preserved.

## Phase 1: Reproducible local environment and artifact layout

### Tasks

- [x] Define the supported Python version, targeting Python 3.13.
- [x] Pin direct dependencies and update the lock file.
- [x] Add test configuration and Make targets.
- [x] Add an environment-report command.
- [x] Record Python, MLX, MLX-LM, Transformers, Datasets, OpenAI SDK, and OS versions.
- [x] Create a granular artifact layout for source data, batches, canonical records, rendered arms, runs, and summaries.
- [x] Update `.gitignore` so code/configuration/manifests remain trackable while large or secret artifacts remain excluded.
- [x] Recheck free disk space before model downloads. The gate currently fails at approximately 10.9 GiB free.

### Proposed artifact layout

```text
data/reviewer_rerun/
  source/
  batch_inputs/
  batch_outputs/
  canonical/
  socratic/
  non_socratic/
  manifests/
  audits/

runs/reviewer_rerun/
  training/
  evaluation/

results/reviewer_rerun/
  summaries/
  tables/
  figures/
  reproducibility/
```

### Storage gate

Full model work requires at least approximately 40 GiB free, with approximately 60 GiB preferred. If the internal disk remains constrained, model and dataset caches must be moved to an explicitly configured external location. No cleanup is authorized implicitly.

## Phase 2: Freeze the experiment protocol

### Protocol file

Create `configs/reviewer_rerun/protocol.yaml` containing:

- protocol/schema version;
- source dataset identifier, split, revision, and expected count;
- requested variants per source;
- target accepted-example count;
- validation and filtering settings;
- exact and near-duplicate settings;
- source-group split seed;
- model identifiers;
- LoRA configurations;
- training seeds;
- benchmark identifiers, splits, revisions, and expected counts;
- prompt versions and shot counts;
- decoding settings;
- self-consistency settings;
- final-answer extraction and normalization rules;
- confidence-interval method.

### Candidate and accepted counts

- GSM8K train sources: 7,473 expected.
- Requested candidates: 3 variants per source.
- Initial candidate target: 22,419.
- Accepted canonical target: 21,250 where validation permits.
- Both rendered arms must have exactly the same accepted count and example IDs.

If fewer than 21,250 candidates survive, retry only missing/rejected requests. If more survive, select deterministically according to the frozen policy. Filtering must never be performed independently for the two arms.

### Gate

The protocol is validated and committed before full data generation or model training. Every subsequent manifest records the protocol hash.

## Phase 3: Canonical matched-dataset pipeline

### Source snapshot

Pin GSM8K and store a source manifest containing:

- dataset identifier and configuration;
- split and resolved revision;
- row count;
- normalized source IDs;
- full original questions and worked solutions;
- a SHA-256 fingerprint of the normalized snapshot.

The teacher input must contain the original worked solution, not only the final `####` value.

### Canonical record

Each accepted generated example is stored once with at least:

```json
{
  "schema_version": "1.0",
  "example_id": "gsm8k-train-000123-v02",
  "source_id": "gsm8k-train-000123",
  "variant_id": 2,
  "source_question": "...",
  "source_solution": "...",
  "synthetic_question": "...",
  "solution_steps": [
    {
      "guiding_question": "...",
      "reasoning": "..."
    }
  ],
  "final_answer": "#### 9",
  "generation": {
    "request_id": "...",
    "requested_model": "...",
    "returned_model": "...",
    "prompt_version": "...",
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

### Arm rendering

The Socratic arm contains guiding questions followed by their corresponding declarative reasoning. The non-Socratic arm omits only the separately represented guiding-question fields. It must not be independently generated or paraphrased.

Both arms must preserve:

- source problem;
- numerical and structural variation;
- synthetic problem statement;
- declarative reasoning text;
- calculations;
- final answer format;
- ordering;
- filtering decisions;
- train/validation membership.

### Resumable OpenAI workflow

Implement distinct, idempotent stages:

```text
build -> submit -> status -> download -> assemble -> retry -> validate -> render
```

Required behaviours:

- structured JSON output;
- globally unique Batch `custom_id` values;
- explicit source/variant mapping;
- request and response manifests;
- output/error file tracking;
- request counts and token usage;
- requested and returned model IDs;
- retry only failed/missing requests;
- no row-order assumptions;
- no silent teacher-model fallback;
- no paid submission in automated tests.

### Shared filtering

Apply once to canonical records before rendering:

- schema validity;
- required non-empty fields;
- stable source/variant IDs;
- parseable final-answer format;
- arithmetic/final-answer consistency where verifiable;
- complete, non-truncated reasoning;
- no malformed API output;
- no duplicate steps;
- maximum training length;
- exact duplicate removal;
- real 5-gram Jaccard near-duplicate detection.

Every rejection is written to an audit JSONL with a machine-readable reason, source ID, variant ID, and relevant score or error.

### Near-duplicate policy

- Normalize text according to a frozen function.
- Generate 5-gram sets.
- Identify candidate matches efficiently.
- Calculate exact Jaccard similarity for candidates.
- Apply the frozen threshold.
- Record the matched example ID and exact score for every rejection.

### Source-group split

- Use seed 42.
- Split by `source_id`, never individual variant.
- Keep every sibling variant in the same split.
- Preserve 21,250 total examples where feasible.
- Target approximately 90% training and 10% validation.
- Prefer source integrity over reproducing the exact historical 19,125/2,125 counts if both cannot be satisfied simultaneously.
- Record the exact resulting counts and source assignments.

### Pair validator

Fail unless the two arms have identical:

- row counts;
- example ID sequence;
- source IDs and variant IDs;
- problems;
- shared reasoning;
- calculations;
- final answers;
- ordering;
- splits.

Also require that Socratic guiding questions are present and no guiding-question field or rendered prompt remains in the non-Socratic arm.

### Pilot gate

Run the full offline/build pipeline on approximately 30 source problems. The pilot must prove structured generation handling, source-response mapping, filtering, deduplication, splitting, paired rendering, invariant checks, rejection auditing, and safe resumption. Do not submit the full paid batch before this gate passes and the estimated volume is reported.

## Phase 4: Corrected LoRA training configurations

MLX-LM remains the trainer. Add a wrapper for validation, execution, and resource capture rather than implementing training from scratch.

### Qwen3-0.6B pair

- Model: `mlx-community/Qwen3-0.6B-bf16`
- Learning rate: `8e-5`
- Iterations: `6000`
- Batch size: `4`
- Gradient accumulation: `32`
- Maximum sequence length: `2048`
- LoRA rank: `16`
- LoRA scale: `24`
- LoRA dropout: `0.05`
- Layers: all 28
- Seed: fixed and recorded

Correct intended targets:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

The Socratic and non-Socratic files may differ only in experiment name, dataset path, and adapter output path. Because the historical Qwen3-0.6B configuration used invalid MLP names, rerun both arms using the corrected intended target list. Do not compare a corrected non-Socratic run with the historical flawed Socratic run.

### Non-Qwen run

- Primary model: `mlx-community/Llama-3.2-1B-Instruct-MLXTuned`
- Dataset: full frozen Socratic arm
- Learning rate: `8e-5`
- Iterations: `6000`
- Batch size: `4`
- Gradient accumulation: `32`
- Maximum sequence length: `2048`
- LoRA rank: `16`
- LoRA scale: `24`
- LoRA dropout: `0.05`
- Layers: all 16
- Targets: the same seven projection types where verified against the model
- Seed: fixed and recorded

If this exact identifier is unavailable or incompatible, stop at model preflight and explicitly choose a Phi or Gemma fallback before any non-Qwen training begins.

### Preferred Qwen3-1.7B pair

- Model: `mlx-community/Qwen3-1.7B-4bit`
- Learning rate: `1e-4`
- Iterations: `6000`
- Batch size: `4`
- Gradient accumulation: `32`
- Maximum sequence length: `2048`
- LoRA rank: `16`
- LoRA scale: `8`
- LoRA dropout: `0.05`
- Layers: all 28
- Targets: the seven corrected projection types
- Seed: fixed and recorded

### Model/config preflight

Before iteration zero:

- resolve and record the model revision;
- enumerate matching modules;
- fail if any configured target type matches zero modules;
- verify expected adapted-layer counts;
- report trainable and total parameters;
- compare paired configurations and fail on prohibited differences.

### Resource wrapper

For each run, record:

- exact command and configuration;
- Git commit and protocol hash;
- model ID and revision;
- dataset paths, counts, and hashes;
- precision/quantization;
- target modules and layers;
- trainable and total parameters;
- hardware and software environment;
- start/end timestamps and elapsed time;
- peak MLX memory from logs;
- peak process resident memory where available;
- checkpoint and final adapter paths;
- best-checkpoint selection;
- adapter byte size and SHA-256;
- completion/failure status.

Checkpoint selection is frozen to the final adapter after the last iteration,
never test accuracy. Retain its exact weights path and hash; preserve validation
losses as diagnostics.

### Training smoke gate

Run 10-50 iterations per model family. Require successful model/tokenizer loading, target validation, nonzero trainable parameters, loss computation, checkpoint saving, adapter reloading, resource logging, and generation from the saved adapter.

## Phase 5: Corrected evaluation pipeline

Create `src/evaluate_v2.py`; retain the old evaluator as historical evidence.

### Benchmark registry

Record dataset identifiers, configurations, revisions, splits, field mappings, prompts, answer parsers, and expected counts.

Core expected counts:

- GSM8K: 1,319
- MultiArith: 180
- SVAMP: 300 in the pinned `ChilleD/SVAMP` test split (the repository also has
  700 train rows, which are not part of this benchmark evaluation)

An incomplete run must never be labelled as a complete benchmark result.

### Prompt protocol

- GSM8K: four fixed shots from the training split.
- MultiArith: zero-shot.
- SVAMP: zero-shot.
- Never use the target test split as the few-shot source.
- Require `#### <number>` as the final-answer format.
- Record exact rendered prompts for every prediction.
- Record model-specific chat-template and Qwen thinking-mode settings.
- Keep task semantics, shot count, decoding, token limit, seeds, extraction, and correctness rules identical across compared conditions.

### Answer extraction

- Use the last valid `####` marker.
- Parse only the numeric value associated with that marker.
- Normalize commas, signs, and decimal forms.
- Compare with `Decimal` so `8`, `8.0`, and `8.00` are equivalent.
- Count malformed/missing marked answers as incorrect.
- Do not fall back to an arbitrary number elsewhere in the response.

### Decoding protocol

Primary result:

```text
greedy decoding
temperature: 0
samples: 1
```

Secondary self-consistency result, subject to the timing gate:

```text
samples: 5
temperature: 0.7
top_p: 0.95
top_k: 20
maximum new tokens: frozen after truncation pilot
majority vote over normalized valid answers
```

Derive seeds from the experiment seed, benchmark, example index, and sample index. Store every sample. Define deterministic tie handling before full evaluation. If SC@5 is infeasible, change the frozen protocol for every condition before full evaluation rather than selectively after observing accuracy.

### Evaluation outputs

Store per-example JSONL including experiment/model/adapter identity, benchmark and example ID, prompt, reference answer, sample seeds, full responses, parsed answers, validity, vote, correctness, and timing.

Resume only if the model, adapter, dataset, prompt, decoder, and protocol hashes match the existing run manifest.

### Mandatory matrix

| Condition | GSM8K | MultiArith | SVAMP |
|---|---:|---:|---:|
| Qwen3-0.6B base | Full | Full | Full |
| Qwen3-0.6B Socratic | Full | Full | Full |
| Qwen3-0.6B non-Socratic | Full | Full | Full |
| Llama-3.2-1B base | Full | Full | Full |
| Llama-3.2-1B Socratic | Full | Full | Full |

Preferred extension:

| Condition | GSM8K | MultiArith | SVAMP |
|---|---:|---:|---:|
| Qwen3-1.7B base | Full | Full | Full |
| Qwen3-1.7B Socratic | Full | Full | Full |
| Qwen3-1.7B non-Socratic | Full | Full | Full |

Historical numbers produced by the old evaluator must not appear as directly comparable rows in the corrected main table. A historical baseline is either rerun with the new evaluator or clearly separated from the rerun table.

### Evaluation smoke gate

Evaluate 5-20 examples from each benchmark. Inspect prompts, verify shot provenance, exercise greedy and SC modes, test interruption/resume, compare a small sample by hand, and estimate full duration.

## Phase 6: Full execution order

1. Freeze and hash the accepted canonical dataset.
2. Render and validate the Socratic and non-Socratic arms.
3. Train corrected Qwen3-0.6B Socratic.
4. Train matched Qwen3-0.6B non-Socratic.
5. Evaluate Qwen3-0.6B base, Socratic, and non-Socratic on all three benchmarks.
6. Train Llama-3.2-1B on the full Socratic arm.
7. Evaluate Llama base and tuned conditions on all three benchmarks.
8. Generate required tables and reproducibility material.
9. If feasible, train/evaluate the Qwen3-1.7B matched pair.
10. If time remains, add GSM-Hard for every principal comparison condition.

The optional-stage decision is based on remaining time, storage, and measured duration, not on whether prior accuracy results are favourable.

## Phase 7: Reporting

Generate from raw manifests and predictions:

```text
results/reviewer_rerun/summaries/summary.csv
results/reviewer_rerun/summaries/resource_summary.csv
results/reviewer_rerun/reproducibility/reproducibility.md
results/reviewer_rerun/reproducibility/pairing_report.json
```

### Metrics

For each condition and benchmark, report:

- expected and attempted counts;
- valid parsed answers;
- correct answers;
- accuracy;
- Wilson 95% confidence interval using the observed count;
- total and mean evaluation time.

The summarizer must fail if the actual completed count differs from the
benchmark registry. This prevents a partial SVAMP run—or the 300-row test
split—from being assigned an incorrect 1,000-example confidence interval.

### Resource table

Report:

- hardware and memory;
- software versions;
- precision/quantization;
- training time;
- peak memory;
- trainable/total parameters;
- adapter path, bytes, and hash;
- final and selected checkpoints;
- evaluation duration and examples/second.

Unknown historical values remain explicitly unknown rather than being estimated as measurements.

### Reproducibility note

Include exact model IDs/revisions, data hashes/counts, filters, split method, LoRA settings, prompts, shots and shot IDs, chat-template/thinking settings, decoder settings, self-consistency settings, seed derivation, extraction/normalization rules, benchmark counts, base-result provenance, hardware/software, and artifact locations/hashes.

Figures and paper tables must consume the canonical result summaries rather than hard-coded values.

## Testing plan

### Dataset

- [ ] Canonical schema validation.
- [ ] Deterministic IDs and source/variant mapping.
- [ ] Full source solution included in teacher input.
- [ ] Exact-duplicate behaviour.
- [ ] 5-gram/Jaccard threshold behaviour.
- [ ] Group-aware deterministic splitting.
- [ ] Matched-arm equality.
- [ ] Guiding-question-only removal.
- [ ] Rejection audit output.

### Training

- [ ] Configuration validation.
- [ ] Paired configuration equality.
- [ ] Target-module enumeration.
- [ ] Failure on invalid target modules.
- [ ] Trainable parameter reporting.
- [ ] Run status and resource manifests.
- [ ] Adapter byte size and hashing.

### Evaluation

- [ ] Fixed prompt construction.
- [ ] GSM8K shots come only from train.
- [ ] No target/test leakage.
- [ ] Strict numeric extraction and equivalence.
- [ ] Malformed answer rejection.
- [ ] Deterministic sample seeds.
- [ ] Majority voting and tie policy.
- [ ] Safe resume and manifest mismatch rejection.
- [ ] Expected benchmark count enforcement.

### Reporting

- [ ] Accuracy calculation.
- [ ] Wilson interval calculation.
- [ ] Observed denominator used.
- [ ] Partial benchmark rejection.
- [ ] Resource field extraction.
- [ ] Data-driven summary generation.

## Commit plan

1. `chore: establish reviewer rerun scaffold`
2. `test: add dataset and evaluation regression fixtures`
3. `feat: add canonical paired dataset schema`
4. `fix: make synthetic generation resumable and traceable`
5. `fix: implement shared filtering and duplicate detection`
6. `feat: render and validate matched ablation datasets`
7. `fix: correct LoRA targets and freeze reviewer configs`
8. `feat: record training resources and artifact manifests`
9. `feat: add corrected reproducible evaluator`
10. `feat: aggregate results and generate reproducibility note`
11. `docs: document reviewer rerun workflow`

Large datasets, downloaded models, raw full predictions, and adapters are not committed to ordinary Git history. Compact manifests, hashes, summaries, tests, configurations, and documentation are committed.

## Risk controls

### API spend

Before full Batch submission, report request count, estimated input volume, configured maximum output, teacher model, retry policy, and approximate artifact volume. Full paid submission is a separate explicit checkpoint.

### Storage

Do not start full model downloads/training below the storage gate. Do not delete user artifacts without explicit approval.

### Model availability

Preflight every exact model ID, record the resolved revision, and prevent alias changes from silently altering a run.

### Runtime and interruption

Use experiment-specific directories, incremental outputs, checkpoints, and hash-validated resume. Calibrate duration from smoke tests before scheduling full work.

### Scientific outcome

Do not tune the pipeline based on whether the ablation supports the preferred hypothesis. Similar or better non-Socratic performance is a valid result and must be reported.

## Definition of codebase readiness

Before full paid generation or long-running training:

- [x] Submitted state tagged.
- [x] Reviewer branch active.
- [x] Python environment pinned.
- [ ] Storage plan resolved.
- [x] Protocol frozen and hashed.
- [x] GSM8K source snapshot pinned and hashed.
- [ ] 30-source generation pilot passes.
- [x] Canonical validation and real deduplication pass on offline fixtures.
- [x] Paired render invariants pass on offline fixtures.
- [ ] Exact model IDs load.
- [ ] LoRA target-module preflight passes.
- [ ] Short training adapters save and reload.
- [x] Corrected evaluator tests pass.
- [ ] Small benchmark runs resume safely.
- [x] Reporting regenerates metrics from raw fixture records.
- [ ] API volume, storage, and duration estimates are reported.

## Definition of reviewer-work completion

- [ ] Frozen 21,250-example canonical dataset, subject to documented validation feasibility.
- [ ] Exactly matched Socratic and non-Socratic renders.
- [ ] Qwen3-0.6B Socratic and non-Socratic adapters under identical corrected settings.
- [ ] Qwen3-0.6B base/Socratic/non-Socratic full three-benchmark results.
- [ ] Llama-3.2-1B full-Socratic adapter.
- [ ] Llama base/tuned full three-benchmark results.
- [ ] Raw per-example evaluation outputs.
- [ ] Correct totals and Wilson confidence intervals.
- [ ] Complete resource manifests.
- [ ] Generated reproducibility note.
- [ ] Paper-ready result and resource tables.
- [ ] Historical and corrected results clearly separated.

## First execution block

This block performs preparation only and launches no paid Batch job or full training:

1. Preserve Git state and create the rerun branch.
2. Add this plan and live development log.
3. Add protocol and artifact scaffolding.
4. Add regression tests for pairing, deduplication, prompt contamination, answer extraction, and result counts.
5. Implement the canonical paired-data pipeline.
6. Implement the corrected evaluator.
7. Add corrected model configurations and resource wrapper.
8. Run offline unit tests and fixture-based integration tests.
9. Run only local, tiny smoke checks that do not require full model downloads.
10. Produce a readiness report with remaining blockers, API volume estimate inputs, storage status, and the commands for the subsequent pilot.

## Execution log

### 2026-09-14

- Recorded baseline commit `cfc17c444878291c927b3b5d5642032ff905317c`.
- Observed a pre-existing `.DS_Store` modification; it belongs to the user and will remain untouched and uncommitted.
- Created and switched to branch `reviewer-rerun`.
- Added this plan before beginning implementation work.
- Created annotated tag `submitted-draft-baseline` at the baseline commit.
- Recorded submitted PDF SHA-256 `ae43e89f4b36eeab768045badf7f484208a1925ebb02212dbc0c6052678ea677`.
- Added the frozen protocol scaffold and corrected reviewer-run LoRA configurations.
- Resolved immutable benchmark commits for GSM8K, MultiArith, and SVAMP and froze them in the protocol and evaluator registry.
- Resolved immutable model commits for Qwen3-0.6B, Qwen3-1.7B, and Llama-3.2-1B; recorded precision and weight-file sizes without downloading weights.
- Added an exact `rerun` dependency group and synchronized it from `uv.lock`: Python 3.13.2, MLX 0.30.3, MLX-LM 0.29.1, Transformers 4.57.3, Datasets 4.4.2, OpenAI 2.21.0, PyYAML 6.0.3, and Ruff 0.14.10.
- Verified a local MLX Metal arithmetic operation and the MLX-LM LoRA command-line interface without downloading a model.
- Added the canonical matched-record schema, one-time validation/deduplication, source-group split, and paired Socratic/non-Socratic renderer.
- Added separated, resumable OpenAI source/build/estimate/preflight/submit/status/download/assemble/retry/validate/render stages with explicit paid-call guards.
- Added the corrected, resumable evaluator with pinned benchmarks, uncontaminated GSM8K train shots, strict marked-answer extraction, deterministic seeds, greedy and SC@5 modes, and per-example output.
- Added model-training validation/resource manifests, exact model-snapshot materialization, target-module preflight, adapter hashing, and paired-config drift checks.
- Added report generation from raw predictions and run manifests, using actual denominators and Wilson confidence intervals.
- Built and hashed a 30-source offline GSM8K pilot: 30 teacher requests and 90 candidate examples.
- Built and hashed the complete offline teacher input: 7,473 requests, 22,419 candidate examples, and a 21,985,042-byte Batch JSONL.
- Passed 54 preparation tests plus Ruff formatting/lint, lock consistency, and all five training-configuration validations.
- Froze final-adapter selection after inspecting MLX-LM's validation/checkpoint order; resource manifests record the selected weights file path, size, and hash explicitly.
- Versioned the teacher prompt as `matched-pairs-v2` and added rejection of duplicated solution steps, incorrect explicit arithmetic equalities, and a last explicit result that disagrees with the final answer.
- Validated the pinned benchmark schemas and fingerprints. GSM8K has 1,319 test rows, MultiArith has 180, and `ChilleD/SVAMP` has 300 test rows rather than the draft's apparent 1,000 denominator; froze the corrected SVAMP count at 300.
- Recorded the current host as an Apple M2 Max MacBook Pro with 64 GB unified memory. This differs from the submitted paper's M4/16 GB description and must be reported accurately for new runs.
- Rechecked storage and stopped model work: approximately 10.9 GiB is free, below the 40 GiB training gate.
- Confirmed that no paid OpenAI request, Batch submission, model-weight download, training run, or benchmark evaluation was started.
- Wrote `results/reviewer_rerun/reproducibility/readiness_report.md` with hashes, volumes, completed checks, blockers, and the next risk boundary.

### 2026-09-15

- Closed the remaining model-provenance gap in the corrected evaluator: frozen
  remote identifiers now resolve and materialize at their exact commit before
  MLX-LM loads them; unregistered remote models require an explicit immutable
  revision, and local model directories are content-hashed.
- Added the resolved base-model identity to evaluation manifests and raw
  prediction records so alias drift cannot silently change an official or
  resumed evaluation.
- Expanded the generated reproducibility note with exact frozen model commits,
  prompt layout, shot indices, chat-template/thinking behavior, decoding and
  self-consistency settings, seed derivation, base-number provenance, and the
  full paper-facing resource fields.
- Committed the complete first execution block as preparation milestone
  `58b91ec33b8b8b7b01defd793453cceda3b6cc39`, excluding the pre-existing
  user-owned `.DS_Store` modification and all ignored raw JSONL/model artifacts.
- Published `reviewer-rerun` to `origin` from handoff commit
  `0ffae1c6a72b10ce4d44022be6f29c04d82aa389` and configured the local branch
  to track `origin/reviewer-rerun`.
- Added a guarded fresh-M4 bootstrap pinned to `uv 0.9.18` and Python 3.13.2.
  It requires a clean `reviewer-rerun` checkout on an Apple M4-family Mac,
  synchronizes the locked environment, runs preparation checks and no-download
  training dry-runs, and makes no API request or model download. The full
  preparation gate now passes 56 tests.
- Added a separate reviewer-rerun landing page and M4 operational handover.
  The original `README.md` and `HANDOVER.md` remain as historical documents;
  only short notices were added to route new work to the separate rerun files.
- Added `.env.reviewer_rerun.example` for the corrected pipeline while
  retaining the legacy `.env.example` with a pointer-only notice. The new
  template documents explicit shell loading, fail-closed empty credentials,
  optional Hugging Face authentication/cache relocation, and the prohibition
  on bootstrap test-bypass variables during official runs.
- Replaced the planned incremental RA release with a complete one-pass handoff
  protocol. Added scoped retry assembly, manifest-derived hash checks,
  deterministic canonical merging, filtered-source regeneration with
  whole-source replacement, guarded training/evaluation matrix runners, smoke
  isolation, and a strict three-training/15-evaluation reporting gate. The
  preparation suite now contains 68 tests.
- Began the live reviewer-rerun dataset-generation phase on the preparation
  machine. Re-ran all 68 tests, Ruff formatting/lint checks, and all five
  training-configuration validations successfully. Approximately 229 GiB of
  local storage was available.
- Reproduced the pinned GSM8K source snapshots at revision
  `740312add88f781978c0658806c59bc2815b9866`: 7,473 full rows with SHA-256
  `2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5`
  and 30 pilot rows with SHA-256
  `aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f`.
- Rebuilt the offline teacher inputs without making an API call. The pilot is
  30 requests/90 candidates/89,385 bytes with SHA-256
  `e6e2f832dba31bcfa93283ed969890063700b78f43288abede6ae1d404dff5c9`;
  the full input is 7,473 requests/22,419 candidates/21,985,042 bytes with
  SHA-256
  `36a20f62a64c038272ab817eb74ae91c50ff83ff9d434ea4922fedb13bac434b`.
- With explicit user authorization, ran exactly one paid teacher preflight
  using a locally supplied, permission-restricted, Git-ignored credential.
  No credential value was printed or recorded. The requested and returned
  model were both exactly `gpt-5-mini-2025-08-07`; the response completed and
  parsed into exactly three canonical variants. Saved response SHA-256:
  `874201508690dc39045659a2a1b167490c8f8d07fe979915a415a8159a926b3e`.
- The preflight used 411 input tokens and 1,115 output tokens, including 576
  reasoning tokens (1,526 total). Manual inspection confirmed three valid
  numeric/structural variations, separate question-mark-terminated guiding
  questions, coherent declarative reasoning, correct explicit arithmetic, and
  strict positive-integer `#### N` final answers. No pilot Batch or full Batch
  was submitted; the next paid gate is the 30-request pilot Batch.
- With separate explicit user authorization, submitted the 30-request pilot as
  Batch `batch_6aa96e23e2a88190ba9a77172acb2907`. It completed all 30
  requests with zero API/transport failures and no error file. Measured usage
  was 15,008 input tokens and 46,214 output tokens, including 21,824 reasoning
  tokens. The downloaded 387,288-byte output has SHA-256
  `128343accbe2de2be7d47935e212c674237c0b58bc84be85070f701d6dcd8dc0`.
- Initial assembly falsely rejected seven mathematically correct source groups.
  The binary equality regex truncated compound expressions such as
  `80/100 * 20 = 16` and `14 + 19 + 11 = 44`; finite-precision decimal
  evaluation also mishandled `1/3 * 240 = 80`. Paused API work, replaced the
  regex with an AST-restricted exact-rational arithmetic evaluator, and added
  regression coverage for compound sums/subtractions, fractions, parentheses,
  currencies, chains, and intentionally incorrect equations. The full suite
  now passes 71 tests plus Ruff formatting/lint and all configuration checks.
- Reassembled the same downloaded pilot without spending on retries: all 30
  source responses and all 90 candidates now validate, with zero audit rows.
  Canonical pilot SHA-256:
  `a23375066f7ead8b378e3de8b63c186b30c3d0356ac102b6239d8ff29652c573`.
- The shared frozen filter accepted only 74/90 candidates (82.22%), below the
  mandatory 86/90 pilot gate. All 16 filter rejections were `solution_length`
  failures below the 120-character minimum; pairing nevertheless passed for
  all 74 rendered records. A separate quality scan found 12/90 synthetic
  problem statements with no question mark. The current `matched-pairs-v2`
  prompt asks for concise reasoning but does not enforce either the filter's
  minimum solution length or a direct question in the synthetic problem.
- Stopped before full Batch submission and did not change the frozen filters.
  Extrapolating the current measured usage gives approximately 3.74M input and
  11.51M output tokens for the full 7,473 requests, or about USD 23.96 at the
  official Batch prices observed on 2026-09-15; this projection must be
  remeasured after any prompt revision. The v2 acceptance rate would yield only
  about 18,428 accepted candidates, far below the required 21,250.
- With user approval, preserved the complete failed-v2 pilot under
  `data/reviewer_rerun/archive/matched-pairs-v2-failed-pilot/` before changing
  any active request file. The archive records the preflight, Batch/file IDs,
  token usage, v2 pilot/full input hashes, downloaded response, canonical
  output, render manifests, and the 74/90 gate result. Raw JSONL remains
  Git-ignored but is retained locally for the final artifact transfer.
- Revised only the generation contract, not the downstream filters. Protocol
  `1.1`, prompt `matched-pairs-v3`, and canonical schema `1.1` now require a
  direct problem question, two to six nonredundant steps, and 60--300
  characters of standalone declarative reasoning per step. Guiding questions
  must advance the solution, ignore the distractor, and finish by asking for
  the requested quantity. The 120--2,000-character filter, target count,
  deduplication, split seed, model pins, LoRA settings, and evaluation settings
  are unchanged.
- Mirrored the strict generation schema in local canonical validation so
  malformed outputs fail during assembly even if an upstream structured-output
  guarantee is bypassed. Added explicit prompt/schema identities to assembly,
  merge, and retry manifests. The arithmetic equality validator remains the
  AST-restricted exact-rational implementation introduced after the v2 pilot.
- Refreshed the pinned GSM8K snapshots from the existing offline Hugging Face
  cache. Their normalized content is unchanged: 7,473 rows at SHA-256
  `2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5`
  and 30 rows at SHA-256
  `aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f`.
- Built the v3 request files offline without reading the API key or contacting
  OpenAI. The pilot is 30 requests/90 candidates/105,405 bytes at SHA-256
  `dbb92eb9f0a8bf7d54d7f3dd004ec6dfdb7cc14781324dc39760fa233b91f958`;
  the full input is 7,473 requests/22,419 candidates/25,975,624 bytes at
  SHA-256
  `ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359`.
  The active protocol SHA-256 is
  `a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb`.
- Added a separate v3 generation note and v3 readiness report, leaving the
  older readiness report in place as a historical pre-pilot snapshot. Updated
  the active RA protocol and supporting reviewer-rerun pages to use the new
  hashes and planned handoff tag `reviewer-rerun-ra-handoff-v2`.
- Passed 74 unit/integration tests, Ruff lint and formatting, and all five
  reviewer training-configuration validations. Approximately 229 GiB remained
  free. Stopped before the next paid boundary: no v3 preflight, pilot Batch, or
  full Batch has been submitted.

### 2026-09-16

- Reverified that local `HEAD`, `origin/reviewer-rerun`, and tag
  `reviewer-rerun-ra-handoff-v2` all resolve to commit
  `44bb05af82ac6f7bb2ebfe5b853d9654db9c4ea6`. Rechecked the protocol, pilot
  source, and v3 pilot input hashes before crossing the API boundary.
- Confirmed the personal credential was present in the permission-600,
  Git-ignored `.env` without printing its value. The v2 preflight was already
  preserved in its archive, so the new response was written to the separate
  `data/reviewer_rerun/batch_outputs/preflight_v3.json` path.
- With user authorization, sent exactly one paid v3 Responses API request. The
  requested and returned model were both exactly
  `gpt-5-mini-2025-08-07`; the response completed in 12 seconds and has
  SHA-256
  `02b9a4a30a50896d5fe3e24db426e9bc115c156c57e32354a81e6dcdf17997f6`.
  Usage was 519 input and 1,054 output tokens, including 320 reasoning tokens,
  for 1,573 total tokens.
- Parsed the raw response through the canonical schema-1.1 assembler and the
  unchanged shared filter. All three variants passed, with no rejection or
  near-duplicate: each contains two steps, individual reasoning fields contain
  105--132 characters, and shared solutions contain 241--246 characters.
  Guiding questions and direct problem questions are present, distractors are
  absent from all solution steps, and arithmetic and final answers agree.
- Manual review recorded one minor wording defect in variant 3 (`in Monday`
  instead of `on Monday`). It does not affect the mathematical meaning,
  validation result, deduplication, or the matched Socratic/non-Socratic
  ablation, so the preflight gate passed without a prompt or filter change.
- At the official text-token rates checked on 2026-09-16, the observed request
  corresponds to an estimated USD 0.00223775. A purely linear projection is
  about USD 0.0671 for 30 requests and USD 16.72 for 7,473 requests, but these
  figures are planning aids only; the 30-request pilot must supply the measured
  distribution before any full-Batch decision.
- Stopped before the next paid gate. No v3 Batch was submitted. The next
  decision is whether to authorize the 30-request v3 pilot Batch.

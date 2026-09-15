# Reviewer-rerun readiness report

Generated: 2026-09-15

## Outcome

The preparation code is ready for the paid teacher pilot, but the repository is
not yet cleared for model downloads or training. No OpenAI request, Batch job,
model-weight download, training run, or benchmark evaluation was started in
this block.

The immediate blocking gate is storage. The current internal volume has about
10.9 GiB free and is at 99% capacity; the frozen workflow requires at least
40 GiB free, with 60 GiB preferred, or an explicitly configured external cache
and run directory.

## Preserved baseline

- Active branch: `reviewer-rerun`
- Baseline commit: `cfc17c444878291c927b3b5d5642032ff905317c`
- Baseline tag: `submitted-draft-baseline`
- Frozen preparation milestone:
  `58b91ec33b8b8b7b01defd793453cceda3b6cc39`
- Submitted PDF SHA-256:
  `ae43e89f4b36eeab768045badf7f484208a1925ebb02212dbc0c6052678ea677`
- The pre-existing `.DS_Store` modification remains untouched and must not be
  included in reviewer-rerun commits.

## Frozen protocol and environment

- Protocol SHA-256:
  `9b2ebcff1d00309bec5355351c3b9b96de8153d513a7bb06b264adeaf3ac811a`
- Lock-file SHA-256:
  `cea213bade4a75241aa328d902b99e443d8847f5b30d9fa834a3eccf42ff5517`
- Reviewer Python is pinned to 3.13.2; current interpreter: 3.13.2.
- Current host: Apple M2 Max MacBook Pro, 12 CPU cores and 64 GB unified
  memory, on macOS 14.6.1.
- Reviewer environment: MLX 0.30.3, MLX-LM 0.29.1, Transformers 4.57.3,
  Datasets 4.4.2, OpenAI 2.21.0, PyYAML 6.0.3, and Ruff 0.14.10.
- Fresh-M4 bootstrap: install `uv 0.9.18` outside the project environment and
  run `./scripts/bootstrap_m4.sh` from a clean `reviewer-rerun` checkout. The
  script synchronizes the locked dependency group and performs no paid or
  model-weight operation.
- A local Metal/MLX arithmetic check returned `[4, 6]`, and the MLX-LM LoRA
  CLI loaded successfully. It required normal macOS Metal access and did not
  download a model.

This hardware differs from the 16 GB M4 machine described in the submitted
draft. Any new run performed here must be reported as M2 Max/64 GB; historical
hardware claims must not be carried over to the corrected experiments.

## Immutable dataset revisions

- GSM8K: `openai/gsm8k` at
  `740312add88f781978c0658806c59bc2815b9866`
- MultiArith: `ChilleD/MultiArith` at
  `144d44c3fb87c0b9097ac9593c789e716a282e3e`
- SVAMP: `ChilleD/SVAMP` at
  `5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef`

The normalized full GSM8K train snapshot contains 7,473 rows and has SHA-256
`2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5`.
The 30-source pilot snapshot has SHA-256
`aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f`.

## Immutable model revisions and weight volume

| Condition | Repository revision | Weight bytes | Stored precision |
|---|---|---:|---|
| Qwen3-0.6B | `42096995f6402fde107068cf530136fe64b604f8` | 1,192,134,983 | bfloat16 |
| Qwen3-1.7B | `3b1b1768f8f8cf8351c712464f906e86c2b8269e` | 968,080,210 | 4-bit |
| Llama-3.2-1B | `7247cd8c176bbc558293c9b4750e9f97b5beb319` | 2,471,645,521 | bfloat16 |

The required Qwen3-0.6B and Llama weight files total 3,663,780,504 bytes
(about 3.41 GiB). Including the optional Qwen model brings the weight-only
total to 4,631,860,714 bytes (about 4.31 GiB). This does not include download
caches, tokenizer/config files, compiled caches, datasets, checkpoints,
adapters, logs, or evaluation output.

The training wrapper now verifies that the configured commit resolves,
materializes that exact snapshot, passes the local snapshot path to MLX-LM, and
validates all seven intended LoRA target suffixes before an official run. The
evaluator applies the same exact-revision materialization rule; unregistered
remote models require an explicit immutable commit, while local model paths are
content-hashed. The frozen selection policy uses the final adapter after the
last iteration; its exact file path, byte size, and SHA-256 are recorded
separately from the containing adapter directory.

## Offline teacher-request artifacts

The teacher model is frozen as `gpt-5-mini-2025-08-07`, with fallback
forbidden. The original full GSM8K worked solution is included in every teacher
prompt. Guiding questions and declarative reasoning occupy separate structured
fields so both training arms can be rendered from one canonical record.

| Artifact | Requests | Candidate examples | File bytes | SHA-256 |
|---|---:|---:|---:|---|
| 30-source pilot | 30 | 90 | 89,385 | `e6e2f832dba31bcfa93283ed969890063700b78f43288abede6ae1d404dff5c9` |
| Full offline input | 7,473 | 22,419 | 21,985,042 | `36a20f62a64c038272ab817eb74ae91c50ff83ff9d434ea4922fedb13bac434b` |

The full file contains approximately 13,399,110 input characters, or a rough
3,349,778-token planning estimate at four characters per token. The configured
maximum-output ceiling is 26,155,500 tokens. That ceiling is intentionally
conservative and is not a forecast of usage or cost. A single-request paid
preflight must measure the real returned model and token usage before a pilot
or full cost estimate is approved.

## Verification completed

- 56 unit and fixture-based integration tests pass.
- Ruff formatting and lint checks pass.
- Both Qwen Socratic/non-Socratic configuration pairs are byte-semantically
  matched except for their dataset and adapter paths.
- The Llama configuration passes the same seven-target validation.
- Regression tests cover paired rendering, source-group split isolation,
  deterministic selection, exact/5-gram Jaccard deduplication, Batch ID joins,
  failed/missing retry selection, strict teacher-model matching, prompt
  contamination, duplicate solution steps, explicit arithmetic equalities,
  strict final-answer extraction, deterministic voting/seeds, actual benchmark
  denominators, and Wilson intervals.
- The evaluator expects all 1,319 GSM8K, 180 MultiArith, and 300 SVAMP test
  examples, uses GSM8K train examples for its four shots, stores raw
  per-example outputs, and validates resumptions by configuration hash.
- The pinned `ChilleD/SVAMP` repository contains 700 train and 300 test rows.
  The corrected evaluation uses only its 300-row test split; it must not reuse
  the submitted draft's apparent 1,000-example confidence-interval denominator.
- Paid stages fail closed without explicit confirmation strings.

## Gates still open

1. Resolve storage: free at least 40 GiB or configure and verify an external
   Hugging Face cache plus run root.
2. Run one paid teacher request and verify that the requested dated model is
   still returned exactly, structured output is valid, and typical token usage
   is acceptable.
3. Submit and inspect the 30-source paid Batch pilot; assemble, validate,
   deduplicate, and render its matched arms.
4. Download the pinned Qwen3-0.6B snapshot, run target-module preflight, train a
   20-iteration adapter, reload it, and run a small resumable evaluation.
5. Repeat the model smoke for Llama before approving full data generation and
   official training.

## Next approved-risk boundary

The next command that incurs API usage is the one-request teacher preflight in
`docs/reviewer_rerun_workflow.md`. It must not be run merely because an API key
is present. The 30-request pilot Batch and the 7,473-request full Batch each
require a separate explicit decision after inspecting the preceding usage and
quality evidence.

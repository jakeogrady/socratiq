# Reviewer-rerun v3 readiness report

Generated: 2026-09-16

## Outcome

The stricter `matched-pairs-v3` generation contract is implemented, tested,
and built offline. Its pinned source, 30-request pilot input, and 7,473-request
full input all have recorded counts and SHA-256 identities. The single paid v3
preflight passed; no v3 Batch, model-weight download, training run, or corrected
benchmark evaluation has begun.

The prior `matched-pairs-v2` preflight and 30-request Batch are retained as a
failed quality-pilot record. The API completed 30/30 requests, but only 74/90
examples passed the unchanged filter, below the required 86/90 gate. No full
v2 Batch was submitted.

## Current frozen preparation

- Active branch: `reviewer-rerun`.
- Submitted baseline: commit
  `cfc17c444878291c927b3b5d5642032ff905317c`, tag
  `submitted-draft-baseline`.
- Active protocol: version `1.1`, SHA-256
  `a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb`.
- Lock-file SHA-256:
  `cea213bade4a75241aa328d902b99e443d8847f5b30d9fa834a3eccf42ff5517`.
- Teacher model: `gpt-5-mini-2025-08-07`, with fallback forbidden.
- Full source: 7,473 rows, SHA-256
  `2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5`.
- Pilot source: 30 rows, SHA-256
  `aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f`.
- v3 pilot input: 30 requests / 90 candidates / 105,405 bytes, SHA-256
  `dbb92eb9f0a8bf7d54d7f3dd004ec6dfdb7cc14781324dc39760fa233b91f958`.
- v3 full input: 7,473 requests / 22,419 candidates / 25,975,624 bytes,
  SHA-256
  `ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359`.
- v3 preflight response: exact requested/returned model, three canonical rows,
  3/3 filter acceptance, 519 input tokens, 1,054 output tokens, 320 reasoning
  tokens, and SHA-256
  `02b9a4a30a50896d5fe3e24db426e9bc115c156c57e32354a81e6dcdf17997f6`.
- Current M2 preparation volume: approximately 229 GiB free; dataset
  generation is feasible here. Training and evaluation remain assigned to the
  M4 machine and must report that machine's actual environment.
- The local `.env` and its personal API credential remain permission-restricted,
  ignored, unread by offline preparation, and excluded from all artifacts.

## Verification

- 74 unit and fixture-based integration tests pass.
- Ruff lint and format checks pass.
- All five reviewer training configurations validate.
- Both Qwen matched pairs differ only in dataset and adapter paths.
- The v3 strict schema and local assembler independently require a problem
  question mark, two to six solution steps, and 60--300 characters per
  reasoning field.
- The existing 120--2,000-character shared-solution filter, deduplication,
  pairing, seeds, model revisions, training settings, evaluation protocol, and
  benchmark denominators remain unchanged.
- The arithmetic validator safely handles compound expressions, fractions,
  parentheses, currencies, and equality chains using restricted exact-rational
  evaluation.
- Paid stages still fail closed without their explicit confirmation arguments.

## Open gates

1. Separately decide whether to authorize the 30-request v3 Batch pilot.
2. If authorized, submit, download, assemble, and render the pilot; require at
   least 86/90 accepted examples and
   a passed matched-pair audit without weakening filters.
3. Recompute full cost from measured v3 usage and obtain a separate full-Batch
   decision.
4. Generate, assemble, filter, pair, and freeze the final dataset on this
   machine; transfer it without the API key to the M4 operator.
5. On the M4, run the mandatory three-training/15-greedy-evaluation matrix and
   collect the complete resource manifests.

The optional SC@5, Qwen3-1.7B pair, and harder/OOD benchmark remain behind the
mandatory matrix and their own timing/feasibility decisions.

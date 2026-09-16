# Reviewer-rerun v3 readiness report

Generated: 2026-09-16

## Outcome

The stricter `matched-pairs-v3` generation contract is implemented and tested.
Its single-request preflight passed, the paid 30-request Batch completed 30/30
API requests, and a separately authorized two-request retry completed 2/2.
After merging non-overlapping source groups, all 90/90 candidates pass the
unchanged shared filter, deduplication, and matched-pair audit. The mandatory
86/90 pilot gate has passed. No full v3 Batch, model-weight download, training
run, or corrected benchmark evaluation has begun.

The prior `matched-pairs-v2` preflight and 30-request Batch are retained as a
failed quality-pilot record. The API completed 30/30 requests, but only 74/90
examples passed the unchanged filter, below the required 86/90 gate. No full
v2 Batch was submitted.

## Current frozen preparation

- Active branch: `reviewer-rerun`.
- Submitted baseline: commit
  `cfc17c444878291c927b3b5d5642032ff905317c`, tag
  `submitted-draft-baseline`.
- Active protocol: version `1.2`, SHA-256
  `be66a73d43e4ceb34cd599fdbec9a4cd60dec6324d394a79292873fe9624d924`.
- The completed v3 preflight and pilot retain protocol version `1.1` and
  SHA-256
  `a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb`;
  version 1.2 changes only the pre-full-run accepted target to exactly 20,000.
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
- v3 pilot Batch: `batch_6aaa4dc736e48190b81b084b03626d90`,
  30/30 requests completed, zero failed, 18,248 input tokens, 41,715 output
  tokens, 14,208 reasoning tokens, and output SHA-256
  `fe1548c3447c249affc7733bf74c488c0aa5d7c59f35a76f9bdeeab2d05361a5`.
- v3 initial canonical output: 28 complete source groups / 84 candidates,
  SHA-256
  `325a70a223c1c34d8e9240071a85621b63a6fc3f4a4ae9a1774209a91b745e91`;
  all 84 pass filtering and matched pairing.
- Two-request retry input: 6,885 bytes, SHA-256
  `5be8b86b128a3dadc8ff98aa74ed1413a76a960c4a7d37e858d748deda94fd83`;
  submitted as Batch `batch_6aaa54bc4cb8819093b8c2b68ba6d037`.
- v3 retry Batch: 2/2 requests completed, zero failed, 1,155 input tokens,
  2,692 output tokens, 1,088 reasoning tokens, and output SHA-256
  `6f52ea7bfba102b53a60f8c6fed55b9056e0f93520ef20e66f855cb39be73110`.
- Merged v3 pilot: 30 source groups / 90 candidates, canonical SHA-256
  `9ffc4e0a03252e65b8834809950444d54192023d70412aaa260d210bc60f3259`;
  90/90 accepted, zero filter/deduplication rejections, pairing passed, 81
  training rows and 9 validation rows.
- Current M2 preparation volume: approximately 229 GiB free; dataset
  generation is feasible here. Training and evaluation remain assigned to the
  M4 machine and must report that machine's actual environment.
- The local `.env` and its personal API credential remain permission-restricted,
  ignored, unread by offline preparation, and excluded from all artifacts.

## Verification

- 78 unit and fixture-based integration tests pass.
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
  parentheses, currencies, equality chains, Unicode minus, and final numeric
  results from unit-bearing calculations while retaining restricted
  exact-rational evaluation for expressions it evaluates.
- Paid stages still fail closed without their explicit confirmation arguments.

## Open gates

1. Decide whether to authorize the 7,473-request full v3 Batch with an explicit
   maximum cost. The primary-only projection is USD 21.92; applying the
   observed two-of-30 retry rate gives a retry-adjusted projection of USD
   23.33. A USD 30 ceiling would provide contingency.
2. If authorized, generate, assemble, filter, pair, and freeze the final dataset
   on this
   machine; transfer it without the API key to the M4 operator.
3. On the M4, run the mandatory three-training/15-greedy-evaluation matrix and
   collect the complete resource manifests.

The optional SC@5, Qwen3-1.7B pair, and harder/OOD benchmark remain behind the
mandatory matrix and their own timing/feasibility decisions.

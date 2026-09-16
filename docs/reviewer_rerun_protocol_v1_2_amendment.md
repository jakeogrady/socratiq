# Reviewer-rerun protocol 1.2 amendment

Date: 16 September 2026

## Decision

The final reviewer-rerun dataset will contain exactly 20,000 accepted canonical
synthetic examples rather than the previously planned 21,250. Each canonical
example will be rendered once with Socratic guiding questions and once without
them, producing 20,000 matched rows per experimental arm and 40,000 serialized
rows across both arms.

This decision was made after the v3 generation pilot passed and before the
7,473-request full Batch, model training, or benchmark evaluation began. No
training loss, benchmark accuracy, or Socratic-versus-non-Socratic result was
available when the target was changed.

## Rationale

An exact 20,000-example dataset is easier to describe consistently in the
paper, tables, captions, artifact documentation, and reproducibility protocol.
It also gives a clear approximately 18,000/2,000 train/validation target while
preserving source-group separation.

## Changed fields

- Protocol version: `1.1` to `1.2`.
- Accepted canonical target: 21,250 to 20,000.
- Default final-render target: 21,250 to 20,000.
- Planned final handoff tag: `reviewer-rerun-ra-handoff-v3`.

Active protocol SHA-256:

```text
be66a73d43e4ceb34cd599fdbec9a4cd60dec6324d394a79292873fe9624d924
```

## Unchanged fields

- 7,473 pinned GSM8K source problems;
- three requested variants per source;
- 22,419 raw candidate capacity;
- teacher model `gpt-5-mini-2025-08-07`, with fallback forbidden;
- `matched-pairs-v3` prompt and canonical schema `1.1`;
- reasoning-length and solution-length rules;
- arithmetic validation;
- exact and five-gram Jaccard deduplication;
- source-group splitting, validation fraction, and seed;
- matched Socratic/non-Socratic rendering;
- every LoRA, model, benchmark, prompt, seed, decoding, extraction, and
  reporting setting.

The full request JSONL is therefore byte-identical to its protocol-1.1 build:

```text
requests: 7,473
candidates: 22,419
bytes: 25,975,624
SHA-256: ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359
```

Its compact manifest was rebuilt offline to record the protocol-1.2 hash. It
has no Batch ID and no full paid submission has occurred.

## Dataset implications

The raw candidate buffer is now:

```text
22,419 - 20,000 = 2,419 candidates
```

This permits approximately 10.79% of raw candidates to be rejected before a
filtered-source retry becomes necessary. If more than 20,000 examples survive,
the frozen seeded selector retains exactly 20,000 before the source-group
train/validation split. The requested per-arm split is approximately 18,000
training and 2,000 validation rows; the final manifest records the exact counts
while preventing source leakage.

## Historical boundary

The paid v3 preflight, 30-request pilot, and two-request pilot retry remain
immutable protocol-1.1 evidence with protocol SHA-256
`a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb`.
Their prompts, filters, and validation rules are fully compatible with version
1.2 because the amendment changes only the later exact-size selection target.

The submitted-paper `README.md`, `HANDOVER.md`, historical dataset-reconstruction
scripts, and archived v2 pilot remain unchanged.

## Verification

After the amendment:

- all 78 unit and fixture-based integration tests pass;
- Ruff format and lint checks pass;
- all five reviewer training configurations validate;
- both Qwen matched-configuration checks pass;
- the full request count, byte count, and SHA-256 remain unchanged.

The next paid boundary remains a separate decision on the 7,473-request full
Batch and its cost ceiling.

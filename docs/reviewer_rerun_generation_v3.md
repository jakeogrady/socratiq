# Reviewer-rerun generation contract v3

This note records the active synthetic-data generation contract after the paid
`matched-pairs-v2` pilot failed its predeclared quality gate. It supplements the
chronological development log without replacing the older rerun documents or
the preserved v2 evidence.

## Decision

The downstream dataset filters are unchanged. Instead, prompt version
`matched-pairs-v3` and canonical schema version `1.1` make the observed v2
failure modes invalid at generation and assembly time:

- every synthetic problem must contain a direct, unambiguous mathematical
  question ending with `?`;
- each variation must contain two to six nonredundant solution steps;
- every step keeps its guiding question in a separate field;
- every standalone declarative reasoning field contains 60--300 characters;
- the harmless distractor must not be used or mentioned by a guiding question
  or reasoning field;
- the final guiding question asks for the quantity requested by the problem;
- explicit arithmetic equalities must be correct, and the last explicit result
  must equal the positive integer in the final `#### N` answer.

Two 60-character reasoning fields structurally exceed the existing minimum
shared-solution threshold once their separator and final answer are included.
Six 300-character fields remain below the existing 2,000-character maximum.
The renderer still applies the original 120--2,000-character bounds,
positive-integer rule, exact and 5-gram Jaccard deduplication, source-group
split, and matched-pair audit independently.

The operational filename `src/openai_conversion_v2.py` is retained to avoid a
needless CLI rewrite. Prompt and canonical schema versions in every manifest
are the scientific generation identities.

## Frozen identities

- Protocol version: `1.1`.
- Protocol SHA-256:
  `a4c1a95f00321867b2aef8f0403af929782dad1f10885f161d552a2c03e7cadb`.
- Teacher model: `gpt-5-mini-2025-08-07`; fallback forbidden.
- GSM8K revision:
  `740312add88f781978c0658806c59bc2815b9866`.
- Full normalized source: 7,473 rows, SHA-256
  `2aa7add8a119abea3747ae1ec7b090f95338b66a9f3bb606bbbbd419d0ac91d5`.
- Pilot normalized source: 30 rows, SHA-256
  `aeee6369f6695e9e1937ce3dbf1f4c671d283f09581eba4fa8eff15855ae752f`.

| v3 request artifact | Requests | Candidates | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| Pilot | 30 | 90 | 105,405 | `dbb92eb9f0a8bf7d54d7f3dd004ec6dfdb7cc14781324dc39760fa233b91f958` |
| Full | 7,473 | 22,419 | 25,975,624 | `ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359` |

These artifacts were built offline on 15 September 2026. Building them did not
contact OpenAI or read the local API key. Raw JSONL files are Git-ignored; the
tracked manifests record their identities.

## Preserved v2 boundary

The v2 Batch completed 30/30 API requests and assembled all 90 candidates, but
only 74 passed the unchanged filter. All 16 filter rejections were below the
120-character shared-solution minimum, and 12/90 problem statements lacked a
question mark. The gate required at least 86/90 accepted candidates, so no full
v2 Batch was submitted.

The complete v2 inputs, response, canonical output, render, and compact
manifests are preserved locally under
`data/reviewer_rerun/archive/matched-pairs-v2-failed-pilot/`. Its README records
the Batch/file IDs, hashes, usage, and failure. V2 responses must never be
assembled against v3 request files.

## Next gate

The next operation is a single paid v3 preflight. It must be separately
authorized and must return the exact dated model, three valid schema-1.1
variations, and recorded token usage. The 30-request v3 Batch is a later,
separate decision. The full v3 Batch remains prohibited until the new pilot
passes the unchanged requirement of at least 86 accepted examples out of 90
and its measured usage supports an accepted cost projection.

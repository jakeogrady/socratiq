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

## Paid v3 preflight

One separately authorized paid v3 preflight completed on 16 September 2026.
The requested and returned model were both exactly
`gpt-5-mini-2025-08-07`. The raw response has SHA-256
`02b9a4a30a50896d5fe3e24db426e9bc115c156c57e32354a81e6dcdf17997f6`.
It used 519 input and 1,054 output tokens, including 320 reasoning tokens, for
1,573 total tokens.

All three schema-1.1 variants assembled and passed the unchanged filter. Each
has two solution steps; reasoning fields contain 105--132 characters and
complete shared solutions contain 241--246 characters. Distractors are absent
from all guiding questions and reasoning, and all arithmetic and final answers
agree. Manual review noted one harmless wording defect, `in Monday` rather than
`on Monday`, in variant 3; it does not alter the mathematical meaning, filter
result, or matched ablation.

The compact evidence is in
`data/reviewer_rerun/batch_outputs/preflight_v3.manifest.json`. The raw response
remains Git-ignored and local.

## Paid v3 Batch pilot

The separately authorized 30-request v3 pilot was submitted on 16 September
2026 as Batch `batch_6aaa4dc736e48190b81b084b03626d90`. All 30 API requests
completed, none failed, and no error file was produced. The downloaded output
is 365,785 bytes with SHA-256
`fe1548c3447c249affc7733bf74c488c0aa5d7c59f35a76f9bdeeab2d05361a5`.
The Batch used 18,248 input tokens and 41,715 output tokens, including 14,208
reasoning tokens, for 59,963 total tokens.

Initial assembly exposed a local validation defect in three otherwise correct
responses: unit-bearing calculations and Unicode minus caused an earlier
intermediate result to be compared with the final answer. The validator was
corrected conservatively and covered by regression tests. It still rejects
incorrect explicit arithmetic and a last numeric equality result that differs
from `final_answer`.

Reassembling the same downloaded bytes accepts 28 complete source groups and
84 candidates. All 84 pass the unchanged shared filter and pairing audit, with
zero filter or duplicate rejection. Reasoning fields contain 60--270
characters, shared reasoning contains 179--615 characters, step counts range
from two to five, every problem and guiding question has the required question
mark, and every accepted response reports exactly the requested dated model.

Two source groups remain genuine generation failures. Source 20 padded a
54-character stripped reasoning field with whitespace; source 22 returned two
final answers inconsistent with its worked arithmetic. Whole-source rejection
is retained, so the current pilot result is 84/90 and has not yet passed the
predeclared 86/90 gate.

At the official Batch rates checked on 16 September 2026 (USD 0.25 per million
input tokens and USD 2.00 per million output tokens), measured pilot usage
corresponds to USD 0.087992. A linear 7,473-request projection is approximately
USD 21.92 and remains only a planning estimate.

## Current retry gate

A retry file containing only `gsm8k-train-000020` and
`gsm8k-train-000022` has been built offline. It contains two requests, is 6,885
bytes, and has SHA-256
`5be8b86b128a3dadc8ff98aa74ed1413a76a960c4a7d37e858d748deda94fd83`.
At the observed per-request usage, its estimated cost is about USD 0.00587.

The retry has not been submitted. It requires a separate paid-call decision.
The full v3 Batch remains prohibited until the merged pilot reaches at least
86/90 accepted examples, passes the matched-pair audit without filter changes,
and its measured usage supports an accepted full-Batch cost ceiling.

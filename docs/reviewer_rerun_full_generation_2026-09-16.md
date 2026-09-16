# Reviewer rerun full-generation report — 2026-09-16

This report records the completed full synthetic-data generation and final
20,000-example matched render for protocol 1.2. It is additive: the historical
pilot, handover, and generation documents remain unchanged.

## Outcome

The generation gate passed. The final dataset contains exactly 20,000
canonical examples rendered into two matched arms:

| Arm | Train | Validation | Total |
|---|---:|---:|---:|
| Socratic | 18,000 | 2,000 | 20,000 |
| Non-Socratic | 18,000 | 2,000 | 20,000 |

The pairing audit passed for both splits. For every paired row, the example
ID, GSM8K source ID, variant ID, synthetic question, declarative reasoning,
and final answer are preserved. The non-Socratic rendering removes only the
guiding-question lines. No guiding question was found in a non-Socratic
completion, and no shared reasoning step was lost.

The final corpus spans 6,761 unique GSM8K source problems. Generation was
attempted for all 7,473 source problems, but the paper must not claim that all
7,473 appear in the final selected corpus.

## Frozen protocol and input identity

- Protocol version: `1.2`
- Protocol SHA-256:
  `be66a73d43e4ceb34cd599fdbec9a4cd60dec6324d394a79292873fe9624d924`
- Prompt version: `matched-pairs-v3`
- Canonical schema version: `1.1`
- Requested and accepted teacher model: `gpt-5-mini-2025-08-07`
- Model fallback: prohibited
- Source rows: 7,473
- Requested variants per source: 3
- Maximum candidate rows: 22,419
- Batch input bytes: 25,975,624
- Batch input SHA-256:
  `ba4f3715197d3ed58b6eaae0f1fcf60a15aab9eadc990dbd00a34ae9f1af4359`

## Batch execution

- Batch ID: `batch_6aaa758451148190ae31f0ad329d582a`
- Created: 2026-09-16 10:55:00 UTC
- Entered `in_progress`: 2026-09-16 10:56:06 UTC
- Entered `finalizing`: 2026-09-16 17:48:17 UTC
- Completed: 2026-09-16 17:52:47 UTC
- Server-side creation-to-completion time: 6 h 57 min 47 s
- Total requests: 7,473
- HTTP 200 responses: 7,431
- HTTP 429 responses: 42
- Successful-response returned model:
  `gpt-5-mini-2025-08-07` for all 7,431 rows

All 42 HTTP failures reported `credit_balance_exhausted` with error type
`insufficient_quota`. The output and error artifacts together account for
every submitted custom ID: there are no duplicate, overlapping, unexpected,
or unaccounted request IDs.

Reported Batch usage:

| Measure | Tokens |
|---|---:|
| Input | 4,425,998 |
| Output | 10,025,780 |
| of which reasoning | 3,582,912 |
| Total | 14,451,778 |

Batch artifact identities:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Output JSONL | 89,689,519 | `047f6fce8a942367f4742b0a50bf7f6145134dcd9cce4d46bbe86c7ae56723d8` |
| Error JSONL | 18,438 | `df204102c649eaea860389df6bd11db1762e69f9d9f3ea1b37309113faac5084` |

## Strict assembly and feasibility gate

The successful HTTP responses were parsed under the frozen canonical schema
and arithmetic checks. Assembly produced:

| Result | Source responses |
|---|---:|
| Fully valid three-variant responses | 6,761 |
| Invalid structured responses | 639 |
| Incomplete responses | 31 |
| Missing because of Batch credit failure | 42 |
| Retry-eligible source IDs | 712 |

The 6,761 fully valid responses produced 20,283 canonical examples. All
20,283 passed the shared length, answer, exact-duplicate, and five-gram
near-duplicate filters. Therefore the feasibility gate exceeded the 20,000
target by 283 examples without a paid retry.

An offline retry input containing the 712 retry-eligible requests was created
for auditability but was not submitted. Its SHA-256 is
`74a51c46f5a2ec9ef321a08e4d0cb2fbd670870b7da9a64ff92ff0f6dc4f0eb1`.
No retry is required for the frozen 20,000-example target.

The final deterministic selection used seed 42 and retained:

- variant 1: 6,676 examples;
- variant 2: 6,663 examples;
- variant 3: 6,661 examples;
- 6,483 source groups with three retained variants;
- 273 source groups with two retained variants; and
- 5 source groups with one retained variant.

Source-group splitting was then applied with seed 42. The training split has
6,086 source groups and the validation split has 675 source groups. Source IDs
do not cross the split boundary.

## Final artifact identities

| Artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| Canonical accepted | 20,000 | 43,020,480 | `85098647cc0ecc06596e1536eb6dea4b2f97abd46342df2c033541a0b949611d` |
| Socratic train | 18,000 | 16,895,915 | `48c1a33c67afbeda39d987e1969aa2e8e286ed9929eac57ba0e97234aaa64b0e` |
| Socratic validation | 2,000 | 1,870,386 | `d92fcf3dd96387de10c6d5740ea7cd161399a92ad9ae6f4a5613dcf58d0a69fc` |
| Non-Socratic train | 18,000 | 12,976,625 | `b75d1a835e282cef7b6320c8e184794d0846ae718814045f1e1f88193690f291` |
| Non-Socratic validation | 2,000 | 1,441,738 | `12cbc706da56db4266e94286c3faee54a187913fb0e378c353164f87fbaac25c` |

Pairing sequence identities:

- train:
  `4a2ac48f7ffb58a8e98c1e788e3518d964a37e0311143a5de4c5c2ed84204156`
- validation:
  `e4fd9f5951f2d3f55030d282dc045a3c5926b85239b53bceb48c1daeeda71c9b`

Compact tracked evidence is stored in:

- `data/reviewer_rerun/batch_inputs/gsm8k_train_full.manifest.json`
- `data/reviewer_rerun/canonical/full_initial.manifest.json`
- `data/reviewer_rerun/canonical/full_merged.manifest.json`
- `data/reviewer_rerun/manifests/dataset_manifest.json`
- `data/reviewer_rerun/manifests/pairing_report.json`

Raw JSONL artifacts remain excluded from ordinary Git history. They must be
transferred to the M4 machine together with their tracked manifests and
verified against the hashes above before training begins.

## Paper-facing interpretation

The accurate description is that generation was attempted from all 7,473
GSM8K training sources and the strict pipeline yielded a pool of 20,283 valid
examples spanning 6,761 sources. A deterministic, pre-specified resource cap
then selected 20,000 examples and rendered exactly matched Socratic and
non-Socratic arms. The cap removed 283 otherwise valid examples; it did not
weaken any filter.

The 20,000 cap should be described as an end-to-end resource and
quality-control budget, not as an Apple M4 peak-memory ceiling. The dataset is
now ready for the M4 transfer and mandatory training smoke gates.

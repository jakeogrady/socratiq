# `matched-pairs-v2` failed pilot archive

This directory preserves the first paid reviewer-rerun pilot and its offline
inputs. It is retained as immutable provenance after the generation contract
was revised to `matched-pairs-v3`; it is not an input to the v3 run.

## Paid execution record

- Teacher model requested and returned: `gpt-5-mini-2025-08-07`.
- Preflight response SHA-256:
  `874201508690dc39045659a2a1b167490c8f8d07fe979915a415a8159a926b3e`.
- Batch ID: `batch_6aa96e23e2a88190ba9a77172acb2907`.
- Batch input file ID: `file-6NaN6PxJaDQ3ij25V4Yqvk`.
- Batch output file ID: `file-T15ovestdFUBRDvmMmHNBF`.
- Requests: 30 completed, 0 failed.
- Usage: 15,008 input tokens, 46,214 output tokens, 21,824 reasoning
  tokens, and 61,222 total tokens.
- Downloaded Batch output SHA-256:
  `128343accbe2de2be7d47935e212c674237c0b58bc84be85070f701d6dcd8dc0`.

## Input and assembly identities

- 30-request input SHA-256:
  `e6e2f832dba31bcfa93283ed969890063700b78f43288abede6ae1d404dff5c9`.
- 7,473-request offline input SHA-256:
  `36a20f62a64c038272ab817eb74ae91c50ff83ff9d434ea4922fedb13bac434b`.
- Canonical 90-example output SHA-256:
  `a23375066f7ead8b378e3de8b63c186b30c3d0356ac102b6239d8ff29652c573`.
- Assembly result: 30 source responses and 90 canonical examples accepted;
  no API/assembly audit failures.

## Why the pilot failed its release gate

The unchanged dataset filters accepted 74 of 90 canonical examples. All 16
rejections were shorter than the 120-character minimum shared-solution length,
and an additional content scan found 12 of 90 synthetic problem statements
without a question mark. The required gate was at least 86 accepted examples
out of 90. Consequently, no full v2 Batch was submitted.

The v3 generation schema enforces a direct question mark, two to six solution
steps, and 60--300 characters of standalone reasoning per step. The downstream
120--2,000-character filter and all deduplication settings remain unchanged.

Raw JSONL files in this archive are intentionally ignored by Git because they
contain generated datasets. The accompanying JSON manifests and this record
remain versionable; the complete artifact directory must be transferred with
the final handoff bundle.

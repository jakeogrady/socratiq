# Reviewer-rerun data artifacts

This directory contains only this policy file and compact provenance manifests
in Git. The reviewer-rerun commands create the following local subdirectories
as needed:

- `source/`: pinned and normalized source snapshots;
- `batch_inputs/`: OpenAI Batch request JSONL and manifests;
- `batch_outputs/`: downloaded success/error output files;
- `canonical/`: validated canonical paired examples;
- `full_feasibilityNN/`: shared-filter feasibility renders kept outside the
  final training paths;
- `socratic/`: rendered MLX-LM train/validation data;
- `non_socratic/`: matched rendered MLX-LM train/validation data;
- `manifests/`: compact provenance and hash records;
- `audits/`: machine-readable rejection and pairing audits.

Generated JSONL, raw responses, and model artifacts are ignored by Git. Files
named `*.manifest.json` and JSON files beneath a `manifests/` directory remain
trackable; they include SHA-256 hashes so published artifacts can be verified
independently. Do not place API keys, access tokens, model weights, or other
secrets in this directory.

Transport retries must be assembled against their own Batch input. If shared
filtering leaves fewer than 21,250 records, regenerate every affected source
group and use a replacement merge so old and new variants from one GSM8K source
are never mixed. Exact commands are in `RA_REVIEWER_RERUN_PROTOCOL.md`.

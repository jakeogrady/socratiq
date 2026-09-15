# Reviewer-rerun data artifacts

This directory contains only this policy file and compact provenance manifests
in Git. The reviewer-rerun commands create the following local subdirectories
as needed:

- `source/`: pinned and normalized source snapshots;
- `batch_inputs/`: OpenAI Batch request JSONL and manifests;
- `batch_outputs/`: downloaded success/error output files;
- `canonical/`: validated canonical paired examples;
- `socratic/`: rendered MLX-LM train/validation data;
- `non_socratic/`: matched rendered MLX-LM train/validation data;
- `manifests/`: compact provenance and hash records;
- `audits/`: machine-readable rejection and pairing audits.

Generated JSONL, raw responses, and model artifacts are ignored by Git. Files
named `*.manifest.json` and JSON files beneath a `manifests/` directory remain
trackable; they include SHA-256 hashes so published artifacts can be verified
independently. Do not place API keys, access tokens, model weights, or other
secrets in this directory.

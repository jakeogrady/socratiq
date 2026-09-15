# Reviewer-rerun reproducibility note

Generated: 2026-09-15T07:57:24+00:00

This note is generated from run manifests and raw prediction records. Unknown fields are left blank rather than reconstructed.

## Evaluation protocol

- Evaluator: `src/evaluate_v2.py`
- Prompt version: `arithmetic-eval-v2`.
- Task instruction: "Solve the arithmetic word problem using concise step-by-step reasoning. End with the final numeric answer on its own line in exactly this format: #### <number>."
- Prompt layout: one user chat message containing the task instruction, then each demonstration as `Question: ...\nAnswer: ...`, then the target as `Question: ...\nAnswer:`.
- GSM8K: 4 fixed examples at train indices 0, 1, 2, and 3; all 1,319 test rows are targets.
- MultiArith: zero-shot; all 180 test rows are targets.
- SVAMP: zero-shot; all 300 rows in the pinned ChilleD test split are targets.
- Chat template: the selected model tokenizer's template, with Qwen thinking explicitly disabled for the frozen primary protocol.
- Required output: final `#### <number>` marker.
- Extraction: last marked number only; no arbitrary-number fallback.
- Comparison: comma-stripped, finite, normalized `Decimal` equality; missing or malformed marked answers are incorrect.
- Primary decoding: greedy, 1 sample, temperature 0, maximum 512 new tokens.
- Secondary decoding: SC@5 only after the frozen timing gate, temperature 0.7, top-p 0.95, top-k 20, maximum 512 new tokens; ties use a separate greedy sample and then numeric/lexical order.
- Randomness: base seed 42; each sample seed is the first 32 bits of SHA-256 over the seed, experiment ID, benchmark, example index, and sample index.
- Base-model provenance: reviewer-rerun base numbers are re-run locally with this evaluator; submitted/legacy base numbers are not copied into the corrected tables.

## Frozen model identifiers

| Model | Revision |
|---|---|
| `mlx-community/Qwen3-0.6B-bf16` | `42096995f6402fde107068cf530136fe64b604f8` |
| `mlx-community/Qwen3-1.7B-4bit` | `3b1b1768f8f8cf8351c712464f906e86c2b8269e` |
| `mlx-community/Llama-3.2-1B-Instruct-MLXTuned` | `7247cd8c176bbc558293c9b4750e9f97b5beb319` |

## Evaluation runs

No completed evaluation manifests were found.

## Training runs

No completed training manifests were found.

## Provenance rule

Only results produced by the frozen reviewer-rerun protocol are directly comparable. Submitted/legacy evaluator rows are historical context and are not merged into these tables.

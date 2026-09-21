# Reviewer-rerun scoring correction and new results

Date: 21 September 2026

Status: primary results note for the September 2026 reviewer rerun. This is a
new document; the submitted-draft and earlier rerun notes remain preserved as
historical records.

## Scope and result provenance

Every number below was calculated from the new M4 rerun prediction files in
`manifests-and-predictions-20260921/`. No accuracy value from the submitted
draft was copied into this table. The immutable model responses were rescored
offline, so this correction requires neither model inference nor retraining.

The source packet contains 24 completed evaluations and 14,392 prediction
rows. Before rescoring, the source hashes recorded by all 24 evaluation
manifests were checked against the prediction files. Rescoring writes separate
derived records and does not modify the source manifests or predictions.

Primary derived artifacts:

- `results/reviewer_rerun/rescored/submitted-last-marked-decimal-v1/summary.csv`
  (`9bf324964a370cbd1621fd3447f3251007b38382d2f4e08a47b4c2313143fbbc`)
- `results/reviewer_rerun/rescored/submitted-last-marked-decimal-v1/paired_comparisons.csv`
  (`0e03f46b9ad7f94be67c36a7df93af76c3bfbe4166fd184333fed4894b621cad`)
- `results/reviewer_rerun/rescored/submitted-last-marked-decimal-v1/rescore_manifest.json`
  (`6e250238c6630cd1fa92d760f184f2c91f4343de73b230be68851a6fffd9fc8e`)

The aggregate manifest records the scorer-source hashes and the source and
derived hash for every individual run.

## Why the checker was corrected

Section 2.3 on page 3 of the submitted manuscript defines answer extraction as
the last occurrence of the `####` delimiter followed by an integer or decimal.
It does not require that the number be the final character sequence in the
response. The rerun code at commit `8cd20d5` accidentally implemented a stricter
terminal regular expression. For example, it rejected this otherwise marked
answer:

```text
#### 8
?
```

That mismatch disproportionately affected models fine-tuned on Socratic
traces. It was a format-scoring artifact rather than evidence that the
underlying arithmetic answer was absent.

The corrected primary rule is therefore:

1. find all explicitly `####`-marked integer or decimal answers in the model
   response;
2. select the last marked answer;
3. normalize commas, signs, trailing decimal zeroes, and negative zero;
4. compare it with the normalized reference using exact finite-decimal
   equality; and
5. mark the response invalid if no marked number exists. There is no fallback
   to arbitrary numbers in the reasoning text.

Text after the last marked number, including `?`, no longer makes that answer
invalid. The former terminal-only extraction remains available as a
format-adherence diagnostic, but it is not the primary accuracy scorer.

This is not a post-hoc alternative selected because it produces preferred
numbers: it restores the extraction rule already declared in the submitted
manuscript. The correction, the old terminal diagnostic, every per-example
decision change, and every source hash are retained in the derived artifact.

## Corrected primary results

All entries are exact-match accuracy under the submitted-paper extraction rule.
Counts in parentheses are correct answers divided by the complete benchmark
size. GSM8K uses 1,319 rows, MultiArith 180, and SVAMP 300.

| Model and condition | GSM8K | MultiArith | SVAMP |
|---|---:|---:|---:|
| Llama-3.2-1B base | 21.68% (286/1,319) | 30.56% (55/180) | 16.33% (49/300) |
| Llama-3.2-1B Socratic | 7.35% (97/1,319) | 76.67% (138/180) | 46.33% (139/300) |
| Qwen3-0.6B base | 37.38% (493/1,319) | 65.56% (118/180) | 50.00% (150/300) |
| Qwen3-0.6B Socratic | 33.89% (447/1,319) | 87.22% (157/180) | 47.67% (143/300) |
| Qwen3-0.6B non-Socratic | 40.33% (532/1,319) | 91.67% (165/180) | 54.33% (163/300) |
| Qwen3-1.7B base | 37.98% (501/1,319) | 93.33% (168/180) | 74.67% (224/300) |
| Qwen3-1.7B Socratic | 55.42% (731/1,319) | 93.89% (169/180) | 64.67% (194/300) |
| Qwen3-1.7B non-Socratic | 56.03% (739/1,319) | 96.11% (173/180) | 63.33% (190/300) |

## Matched ablation results

The paired tests compare correctness on the same benchmark examples. The
reported p-values are two-sided exact McNemar tests and are not corrected for
multiple comparisons.

| Model | Benchmark | Socratic | Non-Socratic | Soc. minus non-Soc. | Exact p |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | GSM8K | 33.89% | 40.33% | -6.44 pp | 0.00000397 |
| Qwen3-0.6B | MultiArith | 87.22% | 91.67% | -4.44 pp | 0.1153 |
| Qwen3-0.6B | SVAMP | 47.67% | 54.33% | -6.67 pp | 0.0594 |
| Qwen3-1.7B | GSM8K | 55.42% | 56.03% | -0.61 pp | 0.7018 |
| Qwen3-1.7B | MultiArith | 93.89% | 96.11% | -2.22 pp | 0.3877 |
| Qwen3-1.7B | SVAMP | 64.67% | 63.33% | +1.33 pp | 0.7202 |

The direct matched comparison provides no evidence that adding guiding
questions improves accuracy. At 0.6B, non-Socratic training is numerically
higher on all three benchmarks and significantly higher on GSM8K. At 1.7B,
the variants are close and none of the three pairwise differences is
significant.

The broader finding is more qualified but still present: synthetic-data LoRA
fine-tuning can produce substantial gains in some model/benchmark settings.
For example, both Qwen3-1.7B variants improve GSM8K by more than 17 percentage
points over the newly rerun base model, and both Qwen3-0.6B variants improve
MultiArith substantially. The effects are not universal: Qwen3-1.7B loses
accuracy on SVAMP, Qwen3-0.6B Socratic loses accuracy on GSM8K and SVAMP, and
the Llama transfer result is positive on MultiArith and SVAMP but negative on
GSM8K. The manuscript should present this heterogeneity rather than claim a
general or uniquely Socratic improvement.

## Large-decimal crash correction

`normalize_numeric()` previously called `Decimal.normalize()` followed by
`quantize(Decimal(1))`. The default decimal context can raise
`decimal.InvalidOperation` for otherwise valid integers with 29 or more digits.
Base Qwen3-1.7B produced 63 such responses on GSM8K, causing the original
evaluation attempt to abort.

The replacement uses fixed-point formatting and strips only fractional trailing
zeroes. It preserves all parsed digits, is scoring-equivalent for values that
the former implementation successfully normalized, and no longer crashes on
answers at or above `10^28`. Regression tests cover this boundary. The three
Qwen3-1.7B GSM8K raw runs (base, Socratic, and non-Socratic) were completed with
the one-line crash fix; the other raw runs were made from commit `8cd20d5`.
The current correction commit consolidates that fix with the manuscript-aligned
checker.

## Fresh-M4 bootstrap corrections

The bootstrap still requires the separately installed, pinned `uv 0.9.18`
executable because `uv` must exist before it can create and synchronize the
project environment. It is therefore intentionally not made a circular
dependency of its own bootstrap. The bootstrap now passes the resolved
executable to the test process through `SOCRATIQ_UV_BIN`; the test also checks
that variable and `PATH` before considering a sibling `.venv/bin/uv`.

The dry-run training test no longer assumes that the dataset is absent. It now
verifies the stronger invariant: dry-run configuration validation must not
resolve a remote model revision or materialize/download a model snapshot. This
makes the documented handoff order—install the transferred dataset overlay,
then run the bootstrap—valid while retaining the no-download guarantee.

## Reproduction

From the repository root, with the source evidence packet present:

```sh
.venv/bin/python -m src.rescore_predictions \
  --evaluation-root manifests-and-predictions-20260921/runs/reviewer_rerun/evaluation \
  --output-root results/reviewer_rerun/rescored/submitted-last-marked-decimal-v1
```

The command deliberately refuses to overwrite an existing output directory.
Use a new output path for an independent reproduction and compare the generated
`summary.csv`, `paired_comparisons.csv`, per-run JSONL files, and recorded
hashes.

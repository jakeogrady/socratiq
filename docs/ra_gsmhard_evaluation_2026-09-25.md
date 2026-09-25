# RA handoff: GSM-Hard evaluation extension

This is a new, isolated follow-on to the completed `evaluation_clean_v1` rerun.
Do not modify, delete, resume, or replace any clean-v1 output. The handoff
message will state the exact Git commit to use.

No OpenAI API key, dataset generation, or model training is required. This job
downloads the pinned GSM-Hard/GSM8K datasets and pinned base models, then reuses
the five final LoRA adapters from the completed rerun.

## 1. Pull and verify

```sh
git switch reviewer-rerun
git pull --ff-only origin reviewer-rerun
git rev-parse HEAD
./scripts/bootstrap_m4.sh
make rerun-check
```

The reported commit must match the handoff message. The M4 workspace must still
contain these five adapter directories:

```text
runs/reviewer_rerun/training/qwen3_0.6b_socratic/adapter
runs/reviewer_rerun/training/qwen3_0.6b_non_socratic/adapter
runs/reviewer_rerun/training/qwen3_1.7b_socratic/adapter
runs/reviewer_rerun/training/qwen3_1.7b_non_socratic/adapter
runs/reviewer_rerun/training/llama3.2_1b_socratic/adapter
```

Verify the selected weights before any inference:

```sh
shasum -a 256 -c configs/reviewer_rerun/gsmhard_adapters.sha256
```

All five lines must say `OK`. The execution script repeats this check. If a
file is absent or differs, stop and restore the original adapter directory from
the previous M4 workspace. Do not retrain, rename another checkpoint, or use a
substitute adapter.

## 2. Review the frozen extension

Read:

- `configs/reviewer_rerun/extensions/gsmhard_extension_v1.yaml`
- `scripts/run_gsmhard_evaluation_matrix.sh`

The target is the official `reasoning-machines/gsm-hard` dataset pinned at
commit `960448f73503112d4226baeb8eb41d3fb5ae2506`: 1,319 target rows from its
train-labelled split. Each prompt uses the same four fixed GSM8K training
demonstrations at indices 0, 1, 2, and 3, pinned separately at
`740312add88f781978c0658806c59bc2815b9866`. This prevents GSM-Hard targets from
being used as demonstrations.

The matrix has eight conditions and two decoding modes, for 16 full runs:

- Qwen3-0.6B: base, Socratic, non-Socratic;
- Qwen3-1.7B: base, Socratic, non-Socratic;
- Llama-3.2-1B: base, Socratic;
- greedy and SC@5 for every condition.

Plan all four stages before executing anything:

```sh
make rerun-plan-gsmhard
```

Each of the four plan commands must print exactly eight `PLAN:` lines. All run
paths must contain `evaluation_gsmhard_v1` or
`evaluation_gsmhard_v1_smoke`; none may contain `evaluation_clean_v1`.

## 3. Run both smoke modes

```sh
caffeinate -s ./scripts/run_gsmhard_evaluation_matrix.sh --smoke --greedy --execute
caffeinate -s ./scripts/run_gsmhard_evaluation_matrix.sh --smoke --sc5 --execute
```

The smoke root must finish with 16 completed manifests, each covering 20 rows:

```sh
find runs/reviewer_rerun/evaluation_gsmhard_v1_smoke -name manifest.json | wc -l
rg -l '"status": "completed"' runs/reviewer_rerun/evaluation_gsmhard_v1_smoke --glob manifest.json | wc -l
```

Both commands must print `16`. Inspect the smoke responses and timings before
starting full inference. Stop if there is an exception, a non-completed
manifest, a model/adapter mismatch, or obviously broken output formatting.

## 4. Run the full matrix

Run greedy first, then SC@5:

```sh
caffeinate -s ./scripts/run_gsmhard_evaluation_matrix.sh --full --greedy --execute
caffeinate -s ./scripts/run_gsmhard_evaluation_matrix.sh --full --sc5 --execute
```

The commands are resumable only when every configuration field and file hash
matches. Re-run the same command after a normal interruption. Never edit a
manifest or prediction JSONL to force a resume.

Budget approximately 36--42 hours of M4 wall time for both full modes, based on
the completed GSM8K timings. Keep the machine on power with adequate free disk
space and cooling. The full output must contain:

- 16 completed manifests and 16 prediction JSONL files;
- 1,319 decisions per run;
- 21,104 decisions in total;
- 63,312 scheduled sample generations, plus any deterministic greedy tie-break
  generations required by SC@5.

## 5. Generate and validate reports

```sh
make rerun-gsmhard-reports
```

This command rejects partial rows and requires the exact eight-condition,
two-mode matrix. Its `report_manifest.json` must show:

```text
matrix_validation.status = passed
matrix_validation.evaluation_runs = 16
evaluation_errors = []
training_errors = []
```

Then generate the immutable-response scoring audit into a new, absent output
directory:

```sh
.venv/bin/python -m src.rescore_predictions \
  --evaluation-root runs/reviewer_rerun/evaluation_gsmhard_v1 \
  --output-root results/reviewer_rerun/evaluation_gsmhard_v1_scoring_audit
```

The audit must report 16 runs, 21,104 rows, and 14 paired comparisons. It uses
the submitted-paper rule: the last `####`-marked integer or decimal anywhere in
the response, including before trailing text such as `?`. It also records the
stricter terminal-only extraction as a diagnostic, not as the primary result.

If the audit output directory already exists, choose a clearly versioned new
directory and report the name. Do not overwrite or delete an earlier audit.

## 6. Package and return

Create one archive containing the full and smoke evaluations, both result
directories, and the small training/adapter manifests. Do not include model
caches or adapter weight files.

```sh
tar -czf results/reviewer-rerun-gsmhard-v1-results.tar.gz \
  runs/reviewer_rerun/evaluation_gsmhard_v1 \
  runs/reviewer_rerun/evaluation_gsmhard_v1_smoke \
  results/reviewer_rerun/evaluation_gsmhard_v1 \
  results/reviewer_rerun/evaluation_gsmhard_v1_scoring_audit \
  runs/reviewer_rerun/training/*/manifest.json \
  runs/reviewer_rerun/training/*/adapter_manifest.json

shasum -a 256 results/reviewer-rerun-gsmhard-v1-results.tar.gz \
  > results/reviewer-rerun-gsmhard-v1-results.tar.gz.sha256
```

Return these files:

1. `reviewer-rerun-gsmhard-v1-results.tar.gz`;
2. `reviewer-rerun-gsmhard-v1-results.tar.gz.sha256`;
3. `evaluation_gsmhard_v1/summaries/summary.csv`;
4. `evaluation_gsmhard_v1/reproducibility/reproducibility.md`;
5. `evaluation_gsmhard_v1/report_manifest.json`;
6. `evaluation_gsmhard_v1_scoring_audit/summary.csv`;
7. `evaluation_gsmhard_v1_scoring_audit/paired_comparisons.csv`;
8. `evaluation_gsmhard_v1_scoring_audit/rescore_manifest.json`.

Also send the output of `git rev-parse HEAD`, the five adapter checksum lines,
and a short note about any interruption or anomaly.

## Interpretation for the paper

Describe GSM-Hard as a large-number numerical-robustness perturbation of the
GSM8K test set, not as a wholly independent out-of-distribution dataset. Focus
on paired relative differences between base, Socratic, and non-Socratic
conditions. Note that automatic numerical perturbation can produce awkward,
negative, or non-commonsensical quantities, which can depress absolute scores.

Primary references:

- Dataset: <https://huggingface.co/datasets/reasoning-machines/gsm-hard>
- PAL paper introducing the benchmark: <https://www.cs.cmu.edu/~callan/Papers/icml23-Luyu-Gao.pdf>

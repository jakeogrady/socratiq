# Socratiq Repository Handover

What is in the repository, what is missing from it, and what each component does —
prepared so the code and the experiments can be picked up by someone who has not
worked in it before.

**Prepared** 4 September 2026 · **Repo state** `main` @ `e6c911a`, clean ·
**Paper** O'Grady & Ramlan, *MLWA* short communication

---

## Contents

1. [The repository at a glance](#1-the-repository-at-a-glance)
2. [What each component does](#3-what-each-component-does)
3. [The pipeline end to end](#4-the-pipeline-end-to-end)
4. [What is present](#5-what-is-present)
5. [What is missing](#6-what-is-missing)
6. [Behaviour to know before re-running](#7-behaviour-to-know-before-re-running)

---

## 1. The repository at a glance

The single most important fact for anyone picking this up: **cloning the repository
gets you the code and none of the artefacts.** Git tracks 49 files. The experimental
outputs — adapters, evaluation CSVs, training logs, the generated dataset — live only
on the original machine, excluded by `.gitignore`.

| Path                     | Contents                                         |
|--------------------------|--------------------------------------------------|
| `src/`                   | 9 Python files, 1,754 lines — the whole codebase |
| `*.yaml` (root)          | 20 LoRA training configs                         |
| `Makefile`               | Every command used to run the project            |
| `src/outputs/`           | The five paper figures, PDF + PNG                |
| `evaluation_summary.csv` | 7 summary rows — the only results in git         |
| `adapters/`              | Trained LoRA weights (partially deleted — §6)    |
| `eval_results/`          | 70 per-question evaluation CSVs                  |
| `batch_inputs/`          | Exact requests sent to the OpenAI Batch API      |
| `socratic_results/`      | Raw Batch API responses, 8 chunks                |
| `merged_results.jsonl`   | Merged responses, 7,451 records, 0 errors        |
| `qa_pairs.jsonl`         | 22,351 extracted QA pairs                        |
| `new_data/`              | Training set, `{question, answer}` form          |
| `new_data_text/`         | Training set, single `{text}` field              |
| `logs/`                  | 24 training logs                                 |

---

## 2. What each component does

### Python modules — `src/`

| File                     | Role                                                                                                                                                                                                                                                                                |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `openai_conversion.py`   | Builds the Socratic dataset. Loads GSM8K train, formats one Batch API request per problem, submits in chunks of 1,000, polls, merges responses, extracts QA pairs, filters by answer length, deduplicates, and writes the 90/10 split at seed 42.                                   |
| `baseline_evaluation.py` | The evaluation harness. Loads a model (optionally with a LoRA adapter), builds a few-shot prompt, generates, parses a final number, compares to gold, and appends a row per question to `eval_results/`. Supports self-consistency by majority vote and resumes from a partial CSV. |
| `generate_figures.py`    | Draws figures 1–4. **All values are hardcoded literals**; the script does not read `eval_results/`.                                                                                                                                                                                 |
| `lora_figure.py`         | Draws the LoRA architecture schematic. Purely illustrative. For the FYP I used this, but it isn't really dynamic or useful for journal entries. I will leave it here initially for the handover, but it should be removed to reduce redundancy.                                     |
| `dataset_improvement.py` | The cleaning pass. Removes a rhetorical question sitting immediately before the `####` marker and flattens each example into a single `text` field, producing `new_data_text/`. Questions inside the reasoning chain are preserved. Reconstructed from ground truth in September 2026 — see §6. |
| `val_loss.py`            | Parses a training log into a validation-loss table with an ASCII curve; identifies the best checkpoint.                                                                                                                                                                             |
| `constants.py`           | Prompt templates, the GSM8K answer regex `####\s*(-?\d+)`, dataset identifiers, defaults.                                                                                                                                                                                           |
| `models.py`              | Loads GSM8K from Hugging Face and formats each row into a single `Question:… Answer:…` string.                                                                                                                                                                                      |

### Training configs — 20 YAML files at the repository root

Each is passed straight to `mlx_lm.lora`. They are grouped by prefix, and the group
determines the base model:

| Prefix                       | Count | Base model            | Notes                                                                                                   |
|------------------------------|------:|-----------------------|---------------------------------------------------------------------------------------------------------|
| `lora-config-run-*`          |     6 | Qwen3-0.6B-bf16       | Trained on `new_data`; targets attention only in practice (§7)                                          |
| `qwen-1.7-lora-config-run-*` |     8 | Qwen3-1.7B-4bit       | Runs 1–4 on `new_data`; runs 5–8 on `new_data_text` with correct MLP targets. Run 8 is bf16, not 4-bit. |
| `llama-lora-config-run-*`    |     6 | Llama-3.2-1B-Instruct | All six trained to 6,000 iterations                                                                     |

Some models received more training runs as they seemed due to the time available for the FYP and also to improve the results.

### Makefile

Every experiment was launched through here.
Targets: `conversion` (build the dataset), `train CONFIG=…` (fine-tune, with a
timestamped log), `baseline-eval-{gsm8k,svamp,multiarith}` (single evaluation),
`loop-eval-*` (the same at n = 1, 2, 4), `val-loss LOG_FILE=…`, and `lint`.

Note the shot counts are fixed inside the loop targets, not passed in:
`loop-eval-gsm8k` uses 4-shot, while `loop-eval-svamp` and `loop-eval-multiarith` use
0-shot. This was a deliberate choice — 4-shot exemplars on those two benchmarks are
bare numbers with no reasoning, which suppresses the model's chain of thought.

### Environment

Python 3.13+, dependencies managed by `uv`, and `mlx-lm[train] >= 0.29.1` — which
means **training and evaluation both require Apple Silicon**.

---

## 4. The pipeline end to end

Three stages. Each writes files the next stage reads, so a stage can be re-run in
isolation if its inputs are present.

### Stage 1 — dataset generation (`make conversion`)

GSM8K train → `batch_inputs/` → OpenAI Batch API (`gpt-5-mini`, `/v1/responses`, low
reasoning effort) → `socratic_results/` → `merged_results.jsonl` → `qa_pairs.jsonl` →
`new_data/{train,valid}.jsonl`.

### Stage 2 — fine-tuning (`make train CONFIG=…`)

Runs `mlx_lm.lora` against a YAML config, tees output to a timestamped file in
`logs/`, and writes adapter weights plus a copy of the effective config into the
directory named by `adapter_path`.

The `adapter_config.json` written alongside the weights is the authoritative record of
what a run actually used — more reliable than the root YAML, which was edited between
runs.

### Stage 3 — evaluation (`make loop-eval-…`)

Loads base model + optional adapter, generates up to 256 tokens per question, parses
the final number, and appends one row per question — question, gold answer, raw
response, parsed answer, correct — to a CSV named for the model, dataset, adapter and
sample count. Re-running the same command resumes from the end of the existing file.

---

## 5. What is present

### Adapter weights

| Directory             | Model           |  Iters | Rank |   Size | State                  |
|-----------------------|-----------------|-------:|-----:|-------:|------------------------|
| `adapters-run-2`      | Qwen3-0.6B-bf16 |  6,000 |   16 | 123 MB | complete               |
| `adapters-run-3`      | Qwen3-0.6B-bf16 |  6,000 |   32 | 245 MB | complete               |
| `adapters-run-4`      | Qwen3-0.6B-bf16 | 15,000 |   16 | 280 MB | complete               |
| `adapters-run-5`      | Qwen3-0.6B-bf16 | 15,000 |    8 | 140 MB | complete               |
| `adapters-run-6`      | Qwen3-0.6B-bf16 | 15,000 |    8 |  18 MB | **2 checkpoints only** |
| `qwen-adapters-run-1` | Qwen3-1.7B-4bit |  6,000 |   16 |  74 MB | complete               |
| `qwen-adapters-run-2` | Qwen3-1.7B-4bit |  6,000 |   16 | 172 MB | complete               |
| `qwen-adapters-run-3` | Qwen3-1.7B-4bit |  6,000 |   32 |   4 KB | **config only**        |
| `qwen-adapters-run-4` | Qwen3-1.7B-4bit | 10,000 |   48 |   4 KB | **config only**        |
| `qwen-adapters-run-5` | Qwen3-1.7B-4bit |  6,000 |   32 |   4 KB | **config only**        |
| `qwen-adapters-run-6` | Qwen3-1.7B-4bit |  6,000 |   16 |   4 KB | **config only**        |
| `qwen-adapters-run-7` | Qwen3-1.7B-4bit |  6,000 |   16 |   4 KB | **config only**        |
| `qwen-adapters-run-8` | Qwen3-1.7B-bf16 |  6,000 |   16 |   4 KB | **config only**        |

Directory contents as of 4 Sep 2026. `adapters-run-*` are Qwen3-0.6B;
`qwen-adapters-run-*` are Qwen3-1.7B.

### Training logs — 24 files

| Family                         | Complete | Stubs | Notes                                                                                              |
|--------------------------------|---------:|------:|----------------------------------------------------------------------------------------------------|
| `llama-lora-config-run-1…6`    |        6 |     0 | 676 lines each, all reaching iteration 6,000 and a final save                                      |
| `qwen-1.7-lora-config-run-1…8` |        8 |     5 | Run 4's log stops at iteration 6,260 of 10,000 with no final save due to the loss curve plateauing |
| `lora-config-run-*` (0.6B)     |        0 |     5 | 0–9 lines each — aborted restarts, not the original training                                       |

### Evaluation results — 70 CSVs in `eval_results/`

One row per question, with the full raw model response retained. This is the most
valuable surviving material: it makes every reported accuracy re-checkable, and
supports re-scoring under a corrected parser without re-running any model. Coverage
spans nine base models on GSM8K, plus Qwen3-0.6B and Qwen3-1.7B on SVAMP and
MultiArith at n = 1, 2 and 4, and the six Llama-3.2-1B fine-tuning runs.

### Dataset artefacts

Every intermediate of stage 1 survives, including the exact request bodies sent to the
API. The generated dataset exists in two formats — `new_data/` with
`{question, answer}` fields and `new_data_text/` with a single joined `{text}` field —
both 19,125 train / 2,125 valid.

---

## 6. What is missing

### Weights that were written and are now gone

> **Blocks reproduction.** The Qwen3-1.7B runs 3–8 and all six Llama-3.2-1B runs left
> their `adapter_config.json` or their training logs behind, but their `.safetensors`
> files no longer exist. The Llama logs record
> `Saved final weights to adapters-llama-run-N/adapters.safetensors` for all six; no
> such directory exists anywhere on disk.
>
> This includes **run 6, the adapter behind the paper's headline Qwen3-1.7B result**.
> Its evaluation CSVs survive, so the numbers can be audited — but the model itself
> cannot be reloaded, re-evaluated on a new benchmark, or released.
>
> The full recipe does survive in `adapters/qwen-adapters-run-6/adapter_config.json`:
> Qwen3-1.7B-4bit, 28 layers, rank 16, scale 8.0, dropout 0.05, lr 1e-4, batch 4,
> 6,000 iterations, `new_data_text`. Retraining is therefore possible; it is the first
> thing that needs doing.

### Code that was used but never committed

- **The original script that produced `new_data_text/`** was never committed. The
  transformation has since been **reconstructed** from the 21,250 input/output pairs
  and now lives in `src/dataset_improvement.py`. It reproduces the shipped data for
  21,240 of them (99.953%); `--verify` confirms this and lists the rest. The rule:
  remove a question sentence that sits directly against the `####` marker and carries
  no internal sentence punctuation. Excluding periods is what preserves decimals —
  *"Does 260 × 0.05 match?"* is correctly left alone.

### Known defects in the shipped training data

- **10 of 21,250 examples are truncated mid-word** in `new_data_text/`, the set the
  Qwen3-1.7B runs 5–8 trained on. The original cleaning pass cut into a decimal rather
  than at a sentence boundary: `train` row 632 ends `"= 0.75 ####"` where the source
  reads `"= 0.75W look consistent with a 25% decrease?"`. Affected rows are `train`
  632, 5208, 6639, 7486, 7908, 10338, 13285, 16823, 18963 and `valid` 1282. At 0.047%
  of the corpus this cannot have moved the results, but it should be fixed before the
  data is deposited. `dataset_improvement.py --verify` lists them.

### Stages described but never run

- **Near-duplicate filtering.** `openai_conversion.py` defines `NGRAM_N = 5` and
  `SIM_THRESHOLD = 0.85` and builds an n-gram index, but never compares against it —
  the threshold constant is referenced nowhere but its own definition. Deduplication
  in practice was exact string matching only.

### Not in version control

- All 1.2 GB of artefacts listed in §1, excluded by `.gitignore` patterns
  `**adapters*/`, `**.jsonl`, `eval_results/**.csv` and `**.log`.
- No `LICENSE` and no citation file.
- `.DS_Store` and a stray file named `OGFILE` are tracked and should be removed.
- The README was corrected in September 2026 — project structure, the cleaning
  section, the clone URL and the `make train` example. One stale figure remains by
  design: the 66.49% fine-tuned Qwen3-1.7B accuracy in the results table, which is
  open question Q2 below and should not be edited until that is settled.

---

## 7. Behaviour to know before re-running

Points where the code does something other than what the paper or README describes.
These are not opinions about style — each was confirmed by running the code or
re-parsing its outputs. Anyone re-running the experiments will hit them.

### LoRA target modules silently do not match

**Location:** all six 0.6B configs; qwen-1.7 runs 1–4; llama runs 2–3

Configs list `self_attn.gate_proj`, `self_attn.up_proj`, `self_attn.down_proj`. Those
paths do not exist — the MLP projections live under `mlp.*`. `mlx-lm` skips unmatched
keys without warning.

Those runs adapted attention only. Confirmed by trainable-parameter counts: a run with
correct `mlp.*` keys reports 11.272 M against 3.408 M for the same rank and layer count
with the broken keys. Runs 5–8 of the 1.7B family use the correct paths, so the
headline result is unaffected.

### Self-consistency at n = 2 is identical to n = 1

**Location:** `baseline_evaluation.py:156`

The majority vote uses `Counter.most_common(1)`, which breaks a two-way tie in favour
of whichever answer was inserted first.

Every n = 2 column reports the first sample. Re-checked across the stored CSVs: the
vote equals sample 1 in 1304/1319 GSM8K, 297/300 SVAMP, 179/180 MultiArith rows — the
remainder are parsing artefacts, not genuine votes.

### Few-shot exemplars are drawn from the test split

**Location:** `baseline_evaluation.py:213`

`generate_prompt` is called with the test set in both the exemplar and target
positions. Test items 0–3 appear in the prompt and are then scored. Small — worth
about 0.3 pp — but it is contamination and a reviewer will find it.

### Generation is capped at 256 tokens, which truncates Qwen3 reasoning blocks

**Location:** `baseline_evaluation.py`

On SVAMP and MultiArith the base models open a `<think>` block on 100% of items but
close it on only 10–18% and emit a `####` marker on none; the fine-tuned models close
it every time. Base scores there come from the last number in a cut-off trace, so the
base–tuned gap on those two benchmarks is inflated by an unknown amount.


### Self-consistency cannot be switched off

**Location:** `baseline_evaluation.py`

The `--self_consistency` flag is `action="store_true"` with `default=True`. Sampling is
always on; single-sample runs are one draw at temperature 0.7, not greedy decoding.

### Output filenames did not always distinguish runs

**Location:** `eval_filename`

Earlier versions of the filename template omitted the adapter and sample-count tags. At
least one file was overwritten: the CSV behind Table 1's Qwen3-0.6B base row now holds
a different 5-sample run. That row's count survives only in `evaluation_summary.csv`.
Two other run CSVs are byte-identical to one another.


---

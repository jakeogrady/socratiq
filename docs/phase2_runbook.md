# Phase 2 runbook: revision-v2 runs on `chee` and `isik`

Reference for the two students running the revision-v2 experiments on the M4
Pro machines. **Start with your checklist,** which lists your steps in order
and what you should see:

- `isik`: `docs/phase2_todo_isik.md`
- `chee`: `docs/phase2_todo_chee.md`

This runbook explains what those steps do. Every command is run from the
repository folder in Terminal.

**What you are running:**

- six LoRA fine-tunes: three models, each trained on the Socratic and on the
  non-Socratic version of the same 20,000 examples;
- their evaluation on four benchmarks;
- the three base models under the same protocol;
- on `chee` only, four extra Qwen3-0.6B runs on smaller data subsets (the second tier).

The protocol is frozen in the tag `protocol-v2-frozen`. The code checks every
setting itself, so your job is to set up, start the queue, and return the
results. Nothing requires waiting for approval or reporting along the way.

## 0. Rules

1. **Never edit code, configs or data.** If anything looks wrong, stop and ask the PI.
2. **Never touch `runs/reviewer_rerun/`.** The old (v1) results stay exactly as they are.
3. **On any failure, stop.** Do not retry, delete or "fix" anything. Send the PI the message and the log (section 6).
4. **Keep `.env` out of every archive and transfer.** No step here needs it.
5. **Run only the commands in your checklist and this file.** In particular, do not run `make phase2-gates`.

## 1. Setup

1. **Get the frozen code:**

   ```sh
   git fetch origin --tags
   git checkout protocol-v2-frozen
   ```

   Quote your repository path when you `cd`: `isik`'s path contains spaces and an apostrophe.
2. **Archive v1:**

   ```sh
   .venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-<machine>.tar
   ```

   This reads `runs/reviewer_rerun/` and `results/reviewer_rerun/` without
   changing them, and writes a `.tar` and a `.sha256` file. Copy both off the machine.
3. **Fetch the Llama checkpoint:**

   ```sh
   .venv/bin/python -m src.revision_v2.llama_base fetch
   ```

   It downloads `mlx-community/Llama-3.2-1B-Instruct-bf16` at the pinned
   revision `863c846`, whose weights are byte-identical to Meta's
   Llama-3.2-1B-Instruct. It then fixes the chat template's date to
   "26 Jul 2024" and refuses unless every file matches the hashes in
   `configs/revision_v2/llama_base_manifest.json`. Both machines therefore
   end up with the identical checkpoint (directory hash
   `4990058eddea56ac54c5800131a87fdb128f69fbfa3a9a9d6bff7d047760e07a`).
   Do not download any other Llama model.
4. **Setup check:**

   ```sh
   ./scripts/phase2_setup.sh --machine <machine>
   ```

   It checks the chip, `uv` 0.9.18 and the frozen tag. It then:
   1. syncs the locked environment;
   2. rebuilds the v2 training data and checks every file against the tracked hashes;
   3. verifies the Llama checkpoint;
   4. runs the tests and config checks;
   5. prints your queue.

   The last line must be `Setup passed on <machine>. No training or evaluation was started.`

The machine needs network access during the runs: the queue checks the pinned
Qwen model revisions on Hugging Face. The models and benchmarks are already
cached from v1.

## 2. The queue

Start it with `./scripts/phase2_queue_<machine>.sh run`. Leave the Terminal
window open and the Mac plugged in, with the lid open. The script keeps the
Mac awake, runs every item in order, and stops at the first failure.

- **The pilot (`isik`).** The queue first trains Qwen3-0.6B Socratic at peak
  learning rate 8e-5 and applies the protocol's rule. The rule passes if the
  validation loss is finite, falls overall, and ends at or below 0.949 (the
  v1 final value).
  - If the run fails the rule, the queue repeats it once at 4e-5.
  - The first accepted peak is saved in `runs/revision_v2/pilot_decision.json`
    and used for every Qwen3-0.6B and Llama run.
  - The accepted pilot run is kept as the Qwen3-0.6B Socratic result.
  - If both peaks fail, the queue stops; send the two `pilot_check.json` files to the PI.
- **The peak on `chee`.** The Qwen3-1.7B pair always uses 1e-4, so `chee`
  starts without a peak.
  - After its 35 core items, the queue stops and asks for the pilot's peak.
    Asena emails it as the `pilot decision:` line of `isik`'s report.
  - Then run `./scripts/phase2_queue_chee.sh run --peak <value>`, and the
    second tier runs.
- **Restarts.** Run the same command again.
  - Finished items are skipped.
  - An interrupted evaluation continues from the next unscored item.
  - An interrupted training run is moved aside to `<run>.incomplete-<time>`, never
    deleted, and restarted from the beginning.
- **Determinism.** Each machine first evaluates the Qwen3-0.6B base model on
  MultiArith with the v1 prompt. The queue compares the responses with that
  machine's own v1 run of the same prompt and shows `vs v1: identical` in the
  report. The PI compares the two machines with each other from the returned packets.
- **Logs.** Every start and finish is recorded in
  `runs/revision_v2/queue_logs/<machine>/queue_events.jsonl`. Each item's full
  output is in the `.log` file next to it.

## 3. Item lists

One row per item, in queue order. The tables assume the pilot chose `8e-5`;
if it chose `4e-5`, every `lr8e-5` reads `lr4e-5`. Qwen3-1.7B items keep
`lr1e-4`. The "command" column is the module the queue runs
(`python -m src.revision_v2.<module> …`). You never type these commands yourself.

Expected minutes come from the v1 manifests. Training on `isik` was about 15%
slower than on `chee` in v1. Zero-shot (P0) GSM8K and GSM-Hard use the v1
four-shot timings as an upper estimate. Llama non-Socratic and the
second-tier runs had no v1 counterpart and are scaled from the closest v1 run.

### `isik`: pilot, Qwen3-0.6B pair, Llama pair, Qwen3-0.6B base (45 items)

| # | Stage | Item | Command run by the queue | Expected (min) | Produces |
|---:|---|---|---|---:|---|
| 1 | 1_core_training | qwen3_0.6b_socratic_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5.yaml` | 103 | `runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 2 | 1_core_training | qwen3_0.6b_non_socratic_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_non_socratic_lr8e-5.yaml` | 85 | `runs/revision_v2/training/qwen3_0.6b_non_socratic_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 3 | 1_core_training | llama3.2_1b_socratic_lr8e-5 | `train run --config configs/revision_v2/training/llama3.2_1b_socratic_lr8e-5.yaml` | 162 | `runs/revision_v2/training/llama3.2_1b_socratic_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 4 | 1_core_training | llama3.2_1b_non_socratic_lr8e-5 | `train run --config configs/revision_v2/training/llama3.2_1b_non_socratic_lr8e-5.yaml` | 133 | `runs/revision_v2/training/llama3.2_1b_non_socratic_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 5 | 2_determinism | qwen3_0.6b_base/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark multiarith --prompt P0 --mode greedy` | 2 | `runs/revision_v2/determinism/isik/qwen3_0.6b_base/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 6 | 3_core_greedy | qwen3_0.6b_socratic_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 46 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 7 | 3_core_greedy | qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 35 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 8 | 3_core_greedy | llama3.2_1b_socratic_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 63 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 9 | 3_core_greedy | llama3.2_1b_non_socratic_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 50 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 10 | 3_core_greedy | qwen3_0.6b_base/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark gsm8k --prompt P0 --mode greedy` | 18 | `runs/revision_v2/evaluation/qwen3_0.6b_base/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 11 | 3_core_greedy | qwen3_0.6b_socratic_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 4 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 12 | 3_core_greedy | qwen3_0.6b_non_socratic_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 13 | 3_core_greedy | llama3.2_1b_socratic_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 4 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 14 | 3_core_greedy | llama3.2_1b_non_socratic_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 15 | 3_core_greedy | qwen3_0.6b_base/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark multiarith --prompt P0 --mode greedy` | 2 | `runs/revision_v2/evaluation/qwen3_0.6b_base/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 16 | 3_core_greedy | qwen3_0.6b_socratic_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 7 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 17 | 3_core_greedy | qwen3_0.6b_non_socratic_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 5 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 18 | 3_core_greedy | llama3.2_1b_socratic_lr8e-5/svamp/P0_greedy | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 6 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 19 | 3_core_greedy | llama3.2_1b_non_socratic_lr8e-5/svamp/P0_greedy | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 5 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 20 | 3_core_greedy | qwen3_0.6b_base/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark svamp --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/qwen3_0.6b_base/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 21 | 3_core_greedy | qwen3_0.6b_socratic_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 62 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 22 | 3_core_greedy | qwen3_0.6b_non_socratic_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 47 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 23 | 3_core_greedy | llama3.2_1b_socratic_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 75 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 24 | 3_core_greedy | llama3.2_1b_non_socratic_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 60 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 25 | 3_core_greedy | qwen3_0.6b_base/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark gsm_hard --prompt P0 --mode greedy` | 21 | `runs/revision_v2/evaluation/qwen3_0.6b_base/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 26 | 3_core_greedy | qwen3_0.6b_socratic_lr8e-5/gsm8k/P4_greedy | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark gsm8k --prompt P4 --mode greedy` | 46 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 27 | 3_core_greedy | qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P4_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P4 --mode greedy` | 35 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 28 | 3_core_greedy | llama3.2_1b_socratic_lr8e-5/gsm8k/P4_greedy | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark gsm8k --prompt P4 --mode greedy` | 63 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 29 | 3_core_greedy | llama3.2_1b_non_socratic_lr8e-5/gsm8k/P4_greedy | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P4 --mode greedy` | 50 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 30 | 3_core_greedy | qwen3_0.6b_base/gsm8k/P4_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark gsm8k --prompt P4 --mode greedy` | 18 | `runs/revision_v2/evaluation/qwen3_0.6b_base/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 31 | 4_sc5 | qwen3_0.6b_socratic_lr8e-5/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode sc5` | 258 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 32 | 4_sc5 | qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode sc5` | 192 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 33 | 4_sc5 | llama3.2_1b_socratic_lr8e-5/gsm8k/P0_sc5 | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode sc5` | 263 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 34 | 4_sc5 | llama3.2_1b_non_socratic_lr8e-5/gsm8k/P0_sc5 | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark gsm8k --prompt P0 --mode sc5` | 210 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 35 | 4_sc5 | qwen3_0.6b_base/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_0.6b_base --benchmark gsm8k --prompt P0 --mode sc5` | 96 | `runs/revision_v2/evaluation/qwen3_0.6b_base/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 36 | 4_sc5 | qwen3_0.6b_socratic_lr8e-5/multiarith/P0_sc5 | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode sc5` | 21 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 37 | 4_sc5 | qwen3_0.6b_non_socratic_lr8e-5/multiarith/P0_sc5 | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode sc5` | 14 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 38 | 4_sc5 | llama3.2_1b_socratic_lr8e-5/multiarith/P0_sc5 | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode sc5` | 22 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 39 | 4_sc5 | llama3.2_1b_non_socratic_lr8e-5/multiarith/P0_sc5 | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark multiarith --prompt P0 --mode sc5` | 18 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 40 | 4_sc5 | qwen3_0.6b_base/multiarith/P0_sc5 | `evaluate --condition-id qwen3_0.6b_base --benchmark multiarith --prompt P0 --mode sc5` | 9 | `runs/revision_v2/evaluation/qwen3_0.6b_base/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 41 | 4_sc5 | qwen3_0.6b_socratic_lr8e-5/svamp/P0_sc5 | `evaluate --condition-id qwen3_0.6b_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode sc5` | 38 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_lr8e-5/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 42 | 4_sc5 | qwen3_0.6b_non_socratic_lr8e-5/svamp/P0_sc5 | `evaluate --condition-id qwen3_0.6b_non_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode sc5` | 28 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_lr8e-5/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 43 | 4_sc5 | llama3.2_1b_socratic_lr8e-5/svamp/P0_sc5 | `evaluate --condition-id llama3.2_1b_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode sc5` | 34 | `runs/revision_v2/evaluation/llama3.2_1b_socratic_lr8e-5/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 44 | 4_sc5 | llama3.2_1b_non_socratic_lr8e-5/svamp/P0_sc5 | `evaluate --condition-id llama3.2_1b_non_socratic_lr8e-5 --benchmark svamp --prompt P0 --mode sc5` | 27 | `runs/revision_v2/evaluation/llama3.2_1b_non_socratic_lr8e-5/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 45 | 4_sc5 | qwen3_0.6b_base/svamp/P0_sc5 | `evaluate --condition-id qwen3_0.6b_base --benchmark svamp --prompt P0 --mode sc5` | 15 | `runs/revision_v2/evaluation/qwen3_0.6b_base/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| | | **Total** | | **2461 (41.0 h)** | |

### `chee`: Qwen3-1.7B pair and base, Llama base, then the second tier (55 items)

| # | Stage | Item | Command run by the queue | Expected (min) | Produces |
|---:|---|---|---|---:|---|
| 1 | 1_core_training | qwen3_1.7b_socratic_lr1e-4 | `train run --config configs/revision_v2/training/qwen3_1.7b_socratic_lr1e-4.yaml` | 240 | `runs/revision_v2/training/qwen3_1.7b_socratic_lr1e-4/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 2 | 1_core_training | qwen3_1.7b_non_socratic_lr1e-4 | `train run --config configs/revision_v2/training/qwen3_1.7b_non_socratic_lr1e-4.yaml` | 193 | `runs/revision_v2/training/qwen3_1.7b_non_socratic_lr1e-4/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 3 | 2_determinism | qwen3_0.6b_base/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_base --benchmark multiarith --prompt P0 --mode greedy` | 2 | `runs/revision_v2/determinism/chee/qwen3_0.6b_base/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 4 | 3_core_greedy | qwen3_1.7b_socratic_lr1e-4/gsm8k/P0_greedy | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark gsm8k --prompt P0 --mode greedy` | 54 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 5 | 3_core_greedy | qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P0_greedy | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark gsm8k --prompt P0 --mode greedy` | 43 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 6 | 3_core_greedy | qwen3_1.7b_base/gsm8k/P0_greedy | `evaluate --condition-id qwen3_1.7b_base --benchmark gsm8k --prompt P0 --mode greedy` | 16 | `runs/revision_v2/evaluation/qwen3_1.7b_base/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 7 | 3_core_greedy | llama3.2_1b_base/gsm8k/P0_greedy | `evaluate --condition-id llama3.2_1b_base --benchmark gsm8k --prompt P0 --mode greedy` | 35 | `runs/revision_v2/evaluation/llama3.2_1b_base/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 8 | 3_core_greedy | qwen3_1.7b_socratic_lr1e-4/multiarith/P0_greedy | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark multiarith --prompt P0 --mode greedy` | 4 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 9 | 3_core_greedy | qwen3_1.7b_non_socratic_lr1e-4/multiarith/P0_greedy | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark multiarith --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 10 | 3_core_greedy | qwen3_1.7b_base/multiarith/P0_greedy | `evaluate --condition-id qwen3_1.7b_base --benchmark multiarith --prompt P0 --mode greedy` | 1 | `runs/revision_v2/evaluation/qwen3_1.7b_base/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 11 | 3_core_greedy | llama3.2_1b_base/multiarith/P0_greedy | `evaluate --condition-id llama3.2_1b_base --benchmark multiarith --prompt P0 --mode greedy` | 1 | `runs/revision_v2/evaluation/llama3.2_1b_base/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 12 | 3_core_greedy | qwen3_1.7b_socratic_lr1e-4/svamp/P0_greedy | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark svamp --prompt P0 --mode greedy` | 6 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 13 | 3_core_greedy | qwen3_1.7b_non_socratic_lr1e-4/svamp/P0_greedy | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark svamp --prompt P0 --mode greedy` | 5 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 14 | 3_core_greedy | qwen3_1.7b_base/svamp/P0_greedy | `evaluate --condition-id qwen3_1.7b_base --benchmark svamp --prompt P0 --mode greedy` | 2 | `runs/revision_v2/evaluation/qwen3_1.7b_base/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 15 | 3_core_greedy | llama3.2_1b_base/svamp/P0_greedy | `evaluate --condition-id llama3.2_1b_base --benchmark svamp --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/llama3.2_1b_base/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 16 | 3_core_greedy | qwen3_1.7b_socratic_lr1e-4/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark gsm_hard --prompt P0 --mode greedy` | 75 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 17 | 3_core_greedy | qwen3_1.7b_non_socratic_lr1e-4/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark gsm_hard --prompt P0 --mode greedy` | 63 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 18 | 3_core_greedy | qwen3_1.7b_base/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_1.7b_base --benchmark gsm_hard --prompt P0 --mode greedy` | 19 | `runs/revision_v2/evaluation/qwen3_1.7b_base/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 19 | 3_core_greedy | llama3.2_1b_base/gsm_hard/P0_greedy | `evaluate --condition-id llama3.2_1b_base --benchmark gsm_hard --prompt P0 --mode greedy` | 37 | `runs/revision_v2/evaluation/llama3.2_1b_base/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 20 | 3_core_greedy | qwen3_1.7b_socratic_lr1e-4/gsm8k/P4_greedy | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark gsm8k --prompt P4 --mode greedy` | 54 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 21 | 3_core_greedy | qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P4_greedy | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark gsm8k --prompt P4 --mode greedy` | 43 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 22 | 3_core_greedy | qwen3_1.7b_base/gsm8k/P4_greedy | `evaluate --condition-id qwen3_1.7b_base --benchmark gsm8k --prompt P4 --mode greedy` | 16 | `runs/revision_v2/evaluation/qwen3_1.7b_base/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 23 | 3_core_greedy | llama3.2_1b_base/gsm8k/P4_greedy | `evaluate --condition-id llama3.2_1b_base --benchmark gsm8k --prompt P4 --mode greedy` | 35 | `runs/revision_v2/evaluation/llama3.2_1b_base/gsm8k/P4_greedy/`: manifest.json, predictions.jsonl |
| 24 | 4_sc5 | qwen3_1.7b_socratic_lr1e-4/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark gsm8k --prompt P0 --mode sc5` | 298 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 25 | 4_sc5 | qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark gsm8k --prompt P0 --mode sc5` | 242 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 26 | 4_sc5 | qwen3_1.7b_base/gsm8k/P0_sc5 | `evaluate --condition-id qwen3_1.7b_base --benchmark gsm8k --prompt P0 --mode sc5` | 85 | `runs/revision_v2/evaluation/qwen3_1.7b_base/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 27 | 4_sc5 | llama3.2_1b_base/gsm8k/P0_sc5 | `evaluate --condition-id llama3.2_1b_base --benchmark gsm8k --prompt P0 --mode sc5` | 184 | `runs/revision_v2/evaluation/llama3.2_1b_base/gsm8k/P0_sc5/`: manifest.json, predictions.jsonl |
| 28 | 4_sc5 | qwen3_1.7b_socratic_lr1e-4/multiarith/P0_sc5 | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark multiarith --prompt P0 --mode sc5` | 18 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 29 | 4_sc5 | qwen3_1.7b_non_socratic_lr1e-4/multiarith/P0_sc5 | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark multiarith --prompt P0 --mode sc5` | 15 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 30 | 4_sc5 | qwen3_1.7b_base/multiarith/P0_sc5 | `evaluate --condition-id qwen3_1.7b_base --benchmark multiarith --prompt P0 --mode sc5` | 7 | `runs/revision_v2/evaluation/qwen3_1.7b_base/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 31 | 4_sc5 | llama3.2_1b_base/multiarith/P0_sc5 | `evaluate --condition-id llama3.2_1b_base --benchmark multiarith --prompt P0 --mode sc5` | 9 | `runs/revision_v2/evaluation/llama3.2_1b_base/multiarith/P0_sc5/`: manifest.json, predictions.jsonl |
| 32 | 4_sc5 | qwen3_1.7b_socratic_lr1e-4/svamp/P0_sc5 | `evaluate --condition-id qwen3_1.7b_socratic_lr1e-4 --benchmark svamp --prompt P0 --mode sc5` | 32 | `runs/revision_v2/evaluation/qwen3_1.7b_socratic_lr1e-4/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 33 | 4_sc5 | qwen3_1.7b_non_socratic_lr1e-4/svamp/P0_sc5 | `evaluate --condition-id qwen3_1.7b_non_socratic_lr1e-4 --benchmark svamp --prompt P0 --mode sc5` | 25 | `runs/revision_v2/evaluation/qwen3_1.7b_non_socratic_lr1e-4/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 34 | 4_sc5 | qwen3_1.7b_base/svamp/P0_sc5 | `evaluate --condition-id qwen3_1.7b_base --benchmark svamp --prompt P0 --mode sc5` | 12 | `runs/revision_v2/evaluation/qwen3_1.7b_base/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 35 | 4_sc5 | llama3.2_1b_base/svamp/P0_sc5 | `evaluate --condition-id llama3.2_1b_base --benchmark svamp --prompt P0 --mode sc5` | 17 | `runs/revision_v2/evaluation/llama3.2_1b_base/svamp/P0_sc5/`: manifest.json, predictions.jsonl |
| 36 | 5_second_tier_training | qwen3_0.6b_socratic_5k_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_socratic_5k_lr8e-5.yaml` | 90 | `runs/revision_v2/training/qwen3_0.6b_socratic_5k_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 37 | 5_second_tier_training | qwen3_0.6b_non_socratic_5k_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_non_socratic_5k_lr8e-5.yaml` | 74 | `runs/revision_v2/training/qwen3_0.6b_non_socratic_5k_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 38 | 5_second_tier_training | qwen3_0.6b_socratic_10k_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_socratic_10k_lr8e-5.yaml` | 90 | `runs/revision_v2/training/qwen3_0.6b_socratic_10k_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 39 | 5_second_tier_training | qwen3_0.6b_non_socratic_10k_lr8e-5 | `train run --config configs/revision_v2/training/qwen3_0.6b_non_socratic_10k_lr8e-5.yaml` | 74 | `runs/revision_v2/training/qwen3_0.6b_non_socratic_10k_lr8e-5/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors |
| 40 | 6_second_tier_greedy | qwen3_0.6b_socratic_5k_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_5k_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 46 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_5k_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 41 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_5k_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_5k_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 35 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_5k_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 42 | 6_second_tier_greedy | qwen3_0.6b_socratic_10k_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_10k_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 46 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_10k_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 43 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_10k_lr8e-5/gsm8k/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_10k_lr8e-5 --benchmark gsm8k --prompt P0 --mode greedy` | 35 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_10k_lr8e-5/gsm8k/P0_greedy/`: manifest.json, predictions.jsonl |
| 44 | 6_second_tier_greedy | qwen3_0.6b_socratic_5k_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_5k_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 4 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_5k_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 45 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_5k_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_5k_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_5k_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 46 | 6_second_tier_greedy | qwen3_0.6b_socratic_10k_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_10k_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 4 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_10k_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 47 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_10k_lr8e-5/multiarith/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_10k_lr8e-5 --benchmark multiarith --prompt P0 --mode greedy` | 3 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_10k_lr8e-5/multiarith/P0_greedy/`: manifest.json, predictions.jsonl |
| 48 | 6_second_tier_greedy | qwen3_0.6b_socratic_5k_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_5k_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 7 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_5k_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 49 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_5k_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_5k_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 5 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_5k_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 50 | 6_second_tier_greedy | qwen3_0.6b_socratic_10k_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_10k_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 7 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_10k_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 51 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_10k_lr8e-5/svamp/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_10k_lr8e-5 --benchmark svamp --prompt P0 --mode greedy` | 5 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_10k_lr8e-5/svamp/P0_greedy/`: manifest.json, predictions.jsonl |
| 52 | 6_second_tier_greedy | qwen3_0.6b_socratic_5k_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_5k_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 62 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_5k_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 53 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_5k_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_5k_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 47 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_5k_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 54 | 6_second_tier_greedy | qwen3_0.6b_socratic_10k_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_socratic_10k_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 62 | `runs/revision_v2/evaluation/qwen3_0.6b_socratic_10k_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| 55 | 6_second_tier_greedy | qwen3_0.6b_non_socratic_10k_lr8e-5/gsm_hard/P0_greedy | `evaluate --condition-id qwen3_0.6b_non_socratic_10k_lr8e-5 --benchmark gsm_hard --prompt P0 --mode greedy` | 47 | `runs/revision_v2/evaluation/qwen3_0.6b_non_socratic_10k_lr8e-5/gsm_hard/P0_greedy/`: manifest.json, predictions.jsonl |
| | | **Total** | | **2640 (44.0 h)** | |

## 4. How to check a finished run

`./scripts/phase2_queue_<machine>.sh report` (add `--peak <value>` on `chee`
once known) prints every item. For each finished run it shows the checks
below and marks the line `OK` or `CHECK`. The queue only marks an item `done`
if these checks pass.

**Training run** (`runs/revision_v2/training/<run>/`):

- **Files:** `train.log`, `manifest.json`, `config.yaml`, `adapter/adapters.safetensors`,
  `adapter/adapter_config.json`, and checkpoint files. The pilot also has `pilot_check.json`.
- **Logged learning rate:** update 16 is logged at iteration 520 and update 188 at iteration 6016.

  | Model | Iteration 520 (update 16) | Iteration 6016 (update 188) |
  |---|---|---|
  | Qwen3-0.6B, Llama, peak 8e-5 | `Learning Rate 8.000e-05` | `Learning Rate 1.000e-06` |
  | Qwen3-0.6B, Llama, peak 4e-5 | `Learning Rate 4.000e-05` | `Learning Rate 1.000e-06` |
  | Qwen3-1.7B (no warm-up) | `Learning Rate 9.844e-05` | `Learning Rate 1.000e-06` |

- **`manifest.json`:**
  - `"status": "completed"`;
  - `post_run_checks.learning_rate.status` is `"passed"`, comparing every logged rate with the schedule;
  - `"seed": 42`;
  - `environment.machine.label` is your machine;
  - mlx 0.30.3 and mlx-lm 0.29.1;
  - `environment.git.commit` is the frozen commit.

**Evaluation run** (`runs/revision_v2/evaluation/<condition>/<benchmark>/<prompt>_<mode>/`):

- **Files:** `manifest.json` and `predictions.jsonl`.
- **`manifest.json`:**
  - `"status": "completed"`;
  - `observed_rows` is 1319 (GSM8K, GSM-Hard), 180 (MultiArith) or 300 (SVAMP);
  - `summary.rules` holds all three scoring rules;
  - `summary.finish_reasons` counts `stop` and `length`.

Do not interpret accuracies. The PI analyses everything afterwards.

## 5. Package and return

```sh
./scripts/phase2_queue_<machine>.sh package     # on chee: add --peak <value>
```

This writes `runs/revision_v2/return/socratiq-v2-<machine>-<time>.tar.gz` and
a `.sha256` file. The packet contains:

- each training run's `train.log`, `manifest.json`, `config.yaml`, and its
  final `adapter/adapters.safetensors` with `adapter_config.json`;
- every evaluation's `manifest.json` and `predictions.jsonl`;
- the pilot files on `isik`;
- the determinism comparison;
- the queue logs;
- a packet manifest with every file's SHA-256 and every item's status.

Unfinished items are listed as unfinished. Intermediate checkpoints and `.env`
are never included. Every manifest records which machine produced it, so the
resource table can name the machine for every row.

## 6. If something fails

The queue prints `STOP: …` with the reason and the path of the item's log, then exits.

1. **Do not** rerun, edit, delete or move anything.
2. Send the PI:
   - the `STOP` message;
   - the item log;
   - `runs/revision_v2/queue_logs/<machine>/queue_events.jsonl`;
   - the item's `manifest.json`, if it exists.
3. Wait. Restart only when the PI says so, with the same `run` command.

The setup and queue scripts also refuse to start in several situations, and say why:

- a tracked file was edited;
- HEAD is not the frozen tag;
- the data or Llama checkpoint hashes do not match;
- the installed mlx or mlx-lm version differs;
- less than 60 GB is free.

Send that message to the PI too.

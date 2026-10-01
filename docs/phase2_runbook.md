# Phase 2 runbook: revision-v2 runs on `chee` and `isik`

For the two students running the revision-v2 experiments on the M4 Pro
machines over the weekend of 3–4 October 2026. Everything you need is in this
file. Every command is run from the repository folder in Terminal.

**Start with your checklist:** `docs/phase2_todo_isik.md` or
`docs/phase2_todo_chee.md`. It lists your steps in order and what to send the
PI after each one. This runbook is the reference behind it.

**What you are running.** Six LoRA fine-tunes (three models, each trained on
the Socratic and the non-Socratic version of the same 20,000 examples), their
evaluation on four benchmarks, and the three base models under the same
protocol. Then, on `chee` only, four extra Qwen3-0.6B runs on smaller data
subsets. The protocol is frozen: the code checks every setting itself, so your
job is to start the queue, watch it, and return the results.

## 0. Rules

1. **Never edit code, configs or data.** If anything looks wrong, stop and ask the PI.
2. **Never touch `runs/reviewer_rerun/`.** The old (v1) results stay exactly as they are.
3. **On any failure, stop.** Do not retry, delete or "fix" anything. Send the log (section 9).
4. **Keep `.env` out of every archive and transfer.** No step here needs it.
5. **Run only the commands in this file.** In particular, do not run `make phase2-gates`.

## 1. Timeline

| When | Who | What |
|---|---|---|
| Friday afternoon | PI | Pushes branch `revision-v2` to GitHub and hands both students the Llama checkpoint file |
| Friday evening | both | Section 2: archive v1, get the code, place the Llama checkpoint, set up |
| Friday evening | `isik` | Section 3: the pilot (about 1 h 45 min), then send the result to the PI |
| Friday night | PI | Approves the pilot and creates the freeze tag `protocol-v2-frozen` |
| Saturday morning | both | Section 4: switch to the tag, rerun setup, start the queue (section 5) |
| Saturday–Sunday | both | Queues run unattended; check progress twice a day (section 5) |
| Monday 5 October, morning | both | Section 10: package and return whatever has finished |

Expected queue length from the v1 timings: **`isik` 41.0 h** (including the
1.7 h pilot) and **`chee` 44.0 h**. The order on each machine is: core
training, determinism check, core greedy evaluation, SC@5, then (on `chee`)
the second tier. If time runs out on Monday, unfinished SC@5 and second-tier
items are dropped; the core greedy results are the minimum.

## 2. Setup (both machines, Friday)

### 2.1 Open the repository

Use the clone you used for v1. Quote the path; `isik`'s path contains spaces and an apostrophe.

```sh
# on isik
cd "/Users/isik/Desktop/PhD/Jake's Thesis Paper/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
# on chee
cd "/Users/chee/Desktop/Jake/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
```

### 2.2 Archive v1 first

Before anything new starts, archive the v1 logs, adapters and predictions,
then copy the archive off the machine (external drive or the PI's shared folder).

```sh
git fetch origin
git checkout revision-v2
.venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-$(hostname -s).tar
```

This reads `runs/reviewer_rerun/` and `results/reviewer_rerun/` without
changing them, and writes the `.tar` plus a `.tar.sha256` file next to it.
Copy both files off the machine and send the `.sha256` line to the PI.

### 2.3 Get the code

- **Friday (pilot, `isik` only):** you are already on `revision-v2` from step 2.2.
- **After the freeze (both machines):**

  ```sh
  git fetch origin --tags
  git checkout protocol-v2-frozen
  ```

If GitHub is unavailable, the PI will give you a `.bundle` file instead; then run
`git fetch /path/to/socratiq-revision-v2.bundle 'refs/tags/*:refs/tags/*' 'refs/heads/*:refs/remotes/bundle/*'`
and the same `git checkout` command.

### 2.4 Place the Llama checkpoint

The PI gives you one file and its checksum:
`llama3.2-1b-instruct-meta-bf16.tar` (2.5 GB) and `llama3.2-1b-instruct-meta-bf16.tar.sha256`.
It is Meta's Llama-3.2-1B-Instruct weights in MLX format, with the chat-template
date fixed. Do not download any other Llama model.

```sh
mkdir -p models/revision_v2
cp /path/to/llama3.2-1b-instruct-meta-bf16.tar* models/revision_v2/
cd models/revision_v2
shasum -a 256 -c llama3.2-1b-instruct-meta-bf16.tar.sha256     # must print: OK
tar -xf llama3.2-1b-instruct-meta-bf16.tar
cd ../..
.venv/bin/python -m src.revision_v2.llama_base verify models/revision_v2/llama3.2-1b-instruct-meta-bf16
```

The tar checksum is `ea7eac20ede340d712d0ef6dea1d55c113118d20ed868c46dbc2a2e8477243cf`.
The verify command must print `"status": "passed"` and
`"directory_sha256": "4990058eddea56ac54c5800131a87fdb128f69fbfa3a9a9d6bff7d047760e07a"`.

### 2.5 Run the setup check

```sh
# Friday, on branch revision-v2 (before the freeze), on each machine:
./scripts/phase2_setup.sh --machine isik --before-freeze
./scripts/phase2_setup.sh --machine chee --before-freeze
# Saturday, on the tag protocol-v2-frozen (after the freeze), on each machine:
./scripts/phase2_setup.sh --machine isik
./scripts/phase2_setup.sh --machine chee
```

It checks the chip, `uv` 0.9.18 and the checkout. It then:

1. syncs the locked environment;
2. rebuilds the v2 training data and checks every file against the tracked hashes;
3. verifies the Llama checkpoint;
4. runs the tests and config checks;
5. prints your queue.

The last line must be `Setup passed on <machine>. No training or evaluation was started.`
If it stops with `ERROR`, send the screen output to the PI.

The machine needs network access during the runs: the queue checks the
pinned Qwen model revisions on Hugging Face. The models and benchmarks are
already cached from v1.

## 3. Friday evening: the pilot (`isik` only)

```sh
./scripts/phase2_queue_isik.sh pilot --peak 8e-5
```

This trains Qwen3-0.6B on the Socratic data for about 1 h 45 min, then applies
the acceptance rule. The rule passes if the validation loss is finite, falls
overall, and ends at or below 0.949 (the v1 final value). It prints the
decision and writes:

- `runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/pilot_check.json`
- `runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/train.log`
- `runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/manifest.json`

**Send these three files to the PI and wait.** Do not start the queue.

- **If the PI approves:** wait for the freeze tag (section 4). The pilot is kept as the Qwen3-0.6B Socratic run.
- **If the PI says the pilot failed:** run the halved-rate pilot once, then send the same three files from `runs/revision_v2/training/qwen3_0.6b_socratic_lr4e-5/`:

  ```sh
  ./scripts/phase2_queue_isik.sh pilot --peak 4e-5
  ```

  If that also fails, stop; the PI decides what happens next.

**The approved peak** (`8e-5` or `4e-5`) is what you pass as `--peak` from now
on, on both machines. It applies to Qwen3-0.6B and Llama. Qwen3-1.7B always
uses 1e-4.

## 4. The freeze

**PI:**

1. Read the commit from the pilot manifest:

   ```sh
   python3 -c "import json;print(json.load(open('runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/manifest.json'))['environment']['git']['commit'])"
   ```

   Use the `lr4e-5` path if the halved pilot was approved.
2. Tag that commit and push it:

   ```sh
   git tag -a protocol-v2-frozen <commit> -m "Revision-v2 protocol frozen after pilot approval"
   git push origin protocol-v2-frozen
   ```

**Students, on both machines:** do section 2.3 (after-freeze part), then section 2.5 without `--before-freeze`.

## 5. Start and watch the queue (both machines)

Plug the Mac into power, then start the queue with the approved peak:

```sh
./scripts/phase2_queue_isik.sh run --peak 8e-5     # on isik
./scripts/phase2_queue_chee.sh run --peak 8e-5     # on chee
```

Leave the Terminal window open. The script keeps the Mac awake, runs every
item in order, prints each item as it starts, and stops at the first failure.

- **Progress:** open a second Terminal window in the same folder and run
  `./scripts/phase2_queue_<machine>.sh plan --peak <peak>`. It lists every item as
  `done`, `partial` or `pending`, with expected minutes and hours remaining.
- **Report for the PI:** `./scripts/phase2_queue_<machine>.sh report --peak <peak>` prints every
  item's status, and for each finished run the learning-rate and row checks with `OK` or
  `CHECK`. Paste it into your check-in email.
- **After a restart, power cut or Ctrl-C:** run the same `run` command again.
  - Finished items are skipped.
  - An interrupted evaluation continues from the next unscored item.
  - An interrupted training run is moved aside to `<run>.incomplete-<time>`, never
    deleted, and restarted from the beginning.
- **Event log:** every start and finish is recorded in
  `runs/revision_v2/queue_logs/<machine>/queue_events.jsonl`. Each item's full
  output is in the `.log` file next to it.

## 6. Determinism check

Item 2 on both machines evaluates the Qwen3-0.6B base model on MultiArith
(about 2 minutes). When both machines have finished it, copy `chee`'s file to
`isik` and compare:

```sh
.venv/bin/python -m src.revision_v2.queue determinism \
  runs/revision_v2/determinism/isik/qwen3_0.6b_base/multiarith/P0_greedy/predictions.jsonl \
  /path/to/chee/predictions.jsonl
```

It must print `"identical": true` with `"items_compared": 180`. On `chee`, you
can also compare against the v1 run of the same prompt, which should also be
identical:
`runs/reviewer_rerun/evaluation_clean_v1/qwen3_0.6b_base/multiarith/greedy/predictions.jsonl`.
Send the output to the PI. If it is not identical, tell the PI but let the queues keep running.

## 7. Your item list

One row per item, in queue order. The tables show the default peak `8e-5`. If
the PI approved `4e-5`, every `lr8e-5` reads `lr4e-5`. Qwen3-1.7B items keep
`lr1e-4`.

Expected minutes come from the v1 manifests. Training on `isik` was about 15%
slower than on `chee` in v1. Zero-shot (P0) GSM8K and GSM-Hard use the v1
four-shot timings as an upper estimate. Llama non-Socratic and the
second-tier runs had no v1 counterpart and are scaled from the closest v1 run.

The "command" column is the module the queue runs (`python -m src.revision_v2.<module> …`).
For adapter evaluations, the queue also passes the adapter path and the expected
adapter SHA-256 from the training manifest. You never type these commands yourself.

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

## 8. How to check a finished run

The queue checks all of this automatically and only marks an item `done` if
it passes. These are the manual checks to do once per finished training run,
and for any item the PI asks about.

### Training run: `runs/revision_v2/training/<run>/`

**Files:** `train.log`, `manifest.json`, `config.yaml`, `adapter/adapters.safetensors`,
`adapter/adapter_config.json`, and checkpoint files `adapter/00xxxxx_adapters.safetensors`.
The pilot also has `pilot_check.json`.

**Logged learning rate.** Update 16 is logged at iteration 520 and update 188 at iteration 6016:

```sh
grep -E "^Iter (520|6016): Train" runs/revision_v2/training/<run>/train.log
```

| Model | Iteration 520 (update 16) | Iteration 6016 (update 188) |
|---|---|---|
| Qwen3-0.6B, Llama, peak 8e-5 | `Learning Rate 8.000e-05` | `Learning Rate 1.000e-06` |
| Qwen3-0.6B, Llama, peak 4e-5 | `Learning Rate 4.000e-05` | `Learning Rate 1.000e-06` |
| Qwen3-1.7B (no warm-up) | `Learning Rate 9.844e-05` | `Learning Rate 1.000e-06` |

Qwen3-1.7B also logs `1.000e-04` from the start.

**`manifest.json` fields:**

- `"status": "completed"`
- `"post_run_checks"` → `"learning_rate"` → `"status": "passed"`, which compares every logged rate with the expected schedule
- `"seed": 42`
- `"environment"` → `"machine"` → `"label"` is your machine name
- `"environment"` → `"packages"` → `"mlx": "0.30.3"` and `"mlx-lm": "0.29.1"`
- `"environment"` → `"git"` → `"commit"` is the frozen commit
- `"adapter"` → `"selected_weights_sha256"` is present
- `"validation_losses"` ends at iteration 6016

### Evaluation run: `runs/revision_v2/evaluation/<condition>/<benchmark>/<prompt>_<mode>/`

**Files:** `manifest.json` and `predictions.jsonl`.

**`manifest.json` fields:**

- `"status": "completed"`
- `"observed_rows"`: 1319 for GSM8K and GSM-Hard, 180 for MultiArith, 300 for SVAMP
- `"summary"` → `"rules"` has `last_marked`, `first_marked` and `last_marked_fallback_last_number`
- `"summary"` → `"finish_reasons"` counts `stop` and `length`
- `"environment"` → `"machine"` → `"label"` is your machine name

Do not interpret accuracies. The PI analyses everything after Monday.

## 9. If something fails

The queue prints `STOP: …` with the reason and the path of the item's log, then exits.

1. **Do not** rerun, edit, delete or move anything.
2. Send the PI:
   - the `STOP` message;
   - the item log, `runs/revision_v2/queue_logs/<machine>/<item>.log`;
   - `runs/revision_v2/queue_logs/<machine>/queue_events.jsonl`;
   - the item's `manifest.json`, if it exists.
3. Wait. Restart only when the PI says so, with exactly the same `run` command.

The setup and queue scripts also refuse to start in several situations, and say why:

- a tracked file was edited;
- HEAD is not the frozen tag;
- the data or Llama checkpoint hashes do not match;
- the installed mlx or mlx-lm version differs;
- less than 60 GB is free.

Send that message to the PI too.

## 10. Monday morning: package and return

When the queue has finished, or on Monday morning if the PI tells you to stop
(press Ctrl-C once and wait for the prompt):

```sh
./scripts/phase2_queue_<machine>.sh package --peak <peak>
```

This writes `runs/revision_v2/return/socratiq-v2-<machine>-<time>.tar.gz` and a `.sha256` file. The packet contains:

- for each training run: `train.log`, `manifest.json`, `config.yaml`, `pilot_check.json` if present, and the final `adapter/adapters.safetensors` with its `adapter_config.json`;
- for each evaluation: `manifest.json` and `predictions.jsonl`;
- the queue logs;
- a packet manifest listing every file's SHA-256 and every item's status (`done`, `partial` or `pending`).

Unfinished items are listed as unfinished, not hidden. Intermediate checkpoints
and `.env` are never included. Copy the `.tar.gz` and its `.sha256` off the
machine and send both to the PI, with the determinism output from section 6.

Every manifest records which machine produced it (`environment.machine` and
`environment.hardware`), so the resource table can name the machine for every row.

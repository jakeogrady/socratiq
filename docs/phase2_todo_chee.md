# Checklist for `chee` (revision-v2 runs)

Work through the boxes in order. Each step gives the command, what you should
see, and what to send the PI. Run every command in Terminal from the
repository folder. Copy commands from this file, not from email, because email
can change quote characters. The background is in `docs/phase2_runbook.md`.

**Your machine runs** the Qwen3-1.7B pair, the Qwen3-1.7B and Llama base
models, and then four smaller-data Qwen3-0.6B runs (the "second tier"): 55
items, about 44 hours, unattended. You can start straight away. Only the
second tier needs one number from Asena (the pilot's learning rate), and the
queue tells you when.

## Setup

- [ ] **0. Prepare the Mac**
  - Plug it in and keep the lid open while the queue runs.
  - Postpone macOS updates.
  - Close other heavy programs.
  - Check free space with `df -h .`: at least 100 GB should be available.

- [ ] **1. Open the repository**

  ```sh
  cd "/Users/chee/Desktop/Jake/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
  ```

- [ ] **2. Get the frozen code**

  ```sh
  git fetch origin --tags
  git checkout protocol-v2-frozen
  git log -1 --oneline
  ```

  The last line must show the commit given in the PI's email.

- [ ] **3. Archive v1 (10–20 min)**

  ```sh
  .venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-chee.tar
  ```

  This writes the archive and a `.sha256` file to your Desktop; nothing in the
  repository changes. Copy both files off the machine, to an external drive or
  OneDrive.

- [ ] **4. Fetch the Llama checkpoint (about 2 min)**

  ```sh
  .venv/bin/python -m src.revision_v2.llama_base fetch
  ```

  This downloads Meta's Llama-3.2-1B-Instruct weights (pinned, about 2.5 GB),
  applies the fixed date, and checks every file against the hashes in the
  repository. It must end with `"status": "passed"` and
  `"directory_sha256": "4990058eddea56ac54c5800131a87fdb128f69fbfa3a9a9d6bff7d047760e07a"`.

- [ ] **5. Setup check (about 10 min)**

  ```sh
  ./scripts/phase2_setup.sh --machine chee
  ```

  The last line must be `Setup passed on chee. No training or evaluation was started.`

  ✉ **Send the PI:** the line in `~/Desktop/socratiq-v1-archive-chee.tar.sha256`
  and the last 5 lines of the setup output.

## Run

- [ ] **6. Start the queue**

  ```sh
  ./scripts/phase2_queue_chee.sh run
  ```

  The first item is `=== 1_core_training/qwen3_1.7b_socratic_lr1e-4 ===`
  (about 4 hours). Leave the window open and the Mac plugged in.

  ✉ **Send the PI:** the first 5 lines on screen.

- [ ] **7. Add the pilot's learning rate when the queue asks for it**

  Asena will email you a line like `pilot decision: 8e-5`. Keep that value;
  `PEAK` below means it.
  - When your queue has finished its first 35 items, it prints
    `Core items finished. The second tier needs the learning rate …` and stops.
  - Start it again with the value:

    ```sh
    ./scripts/phase2_queue_chee.sh run --peak PEAK
    ```

  Already-finished items are skipped, and the second tier (20 more items) starts.

- [ ] **8. Send a report from time to time**

  ```sh
  ./scripts/phase2_queue_chee.sh report
  ```

  Once you know `PEAK`, add `--peak PEAK` to this command.

  ✉ **Send the PI** the whole output.

  - Every finished training line must end in `OK`, with these values:
    - Qwen3-1.7B: `LR update16 9.844e-05 (expected 9.844e-05) update188 1.000e-06 (expected 1.000e-06)`.
    - Second tier: update16 `8.000e-05` (or `4.000e-05` if `PEAK` is 4e-5), update188 `1.000e-06`.
  - The determinism line (item 3) must show `vs v1: identical` and end in `OK`.
  - Every finished evaluation line must end in `OK`.
  - If any line says `CHECK`, or the queue window shows `STOP`, see below.

- [ ] **9. Package and return**

  When the report's last line says `done 55`, or when the PI asks you to stop
  (press Ctrl-C once in the queue window and wait for the prompt), run:

  ```sh
  ./scripts/phase2_queue_chee.sh package --peak PEAK
  ./scripts/phase2_queue_chee.sh report --peak PEAK
  ```

  This creates `runs/revision_v2/return/socratiq-v2-chee-<time>.tar.gz` and a
  `.sha256` file. Upload both to OneDrive, or copy them to an external drive.

  ✉ **Send the PI:** the link or location, the `.sha256` line, and the final report.

## If something goes wrong

- **The queue prints `STOP:` or a report line says `CHECK`:** do not rerun,
  edit, delete or move anything. Send the PI:
  - the `STOP` message;
  - the log path it names;
  - `runs/revision_v2/queue_logs/chee/queue_events.jsonl`.

  Restart only when the PI says so, with the same `run` command.
- **The Mac restarted or lost power:** open the repository folder and run the
  same `run` command again. Add `--peak PEAK` if you already have it.
  - Finished items are skipped.
  - An interrupted evaluation continues where it stopped.
  - An interrupted training run is set aside (never deleted) and restarted.

  Tell the PI it happened.
- **Never** edit code or configs, touch `runs/reviewer_rerun/`, or include `.env` in anything you send.

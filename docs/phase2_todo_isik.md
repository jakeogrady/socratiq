# Checklist for `isik` (revision-v2 runs)

Work through the boxes in order. Each step gives the command and what you
should see. Run every command in Terminal from the repository folder. Copy commands from this file, not from email, because email
can change quote characters. The background is in `docs/phase2_runbook.md`.

**Your machine runs** the pilot, then the Qwen3-0.6B pair, the Llama pair and
the Qwen3-0.6B base model: 45 items, about 41 hours, unattended. You don't
need to wait for anyone: the pilot chooses the learning rate itself, and the
queue carries on.

## Setup

- [ ] **0. Prepare the Mac**
  - Plug it in and keep the lid open while the queue runs.
  - Postpone macOS updates.
  - Close other heavy programs.
  - Check free space with `df -h .`: at least 100 GB should be available.

- [ ] **1. Open the repository**

  ```sh
  cd "/Users/isik/Desktop/PhD/Jake's Thesis Paper/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
  ```

- [ ] **2. Get the frozen code**

  ```sh
  git fetch origin --tags
  git checkout protocol-v2-frozen
  git describe --tags --exact-match
  ```

  The last command must print `protocol-v2-frozen`.

- [ ] **3. Archive v1 (10–20 min)**

  ```sh
  .venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-isik.tar
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
  repository. The output must include `"status": "passed"` and
  `"directory_sha256": "4990058eddea56ac54c5800131a87fdb128f69fbfa3a9a9d6bff7d047760e07a"`.

- [ ] **5. Setup check (about 10 min)**

  ```sh
  ./scripts/phase2_setup.sh --machine isik
  ```

  The last line must be `Setup passed on isik. No training or evaluation was started.`

## Run

- [ ] **6. Start the queue**

  ```sh
  ./scripts/phase2_queue_isik.sh run
  ```

  First it trains the pilot (Qwen3-0.6B Socratic, about 1 h 45 min). Then it
  prints, for example, `Pilot at peak 8e-5: accept (final validation loss 0.93, limit 0.949)`.
  - If the result is `reject`, it automatically repeats the pilot once at
    `4e-5`.
  - After an `accept`, it continues with the rest of the list on its own.

  Leave the window open and the Mac plugged in.

- [ ] **7. Tell Chee the pilot's learning rate**

  The pilot is the first thing your queue does. Once its `Pilot at peak …: accept`
  line has appeared, run (in a second Terminal window, leaving the queue running):

  ```sh
  ./scripts/phase2_queue_isik.sh report
  ```

  The first line reads `pilot decision: 8e-5 …` (or `4e-5`).

  ✉ **Email that line to Chee.** Chee's queue needs the value for its last 20
  items. Your own queue needs nothing from you; leave it running.

- [ ] **8. Check progress whenever you like (optional)**

  ```sh
  ./scripts/phase2_queue_isik.sh report
  ```

  - Every finished training line must end in `OK`, with
    `LR update16 8.000e-05 (expected 8.000e-05) update188 1.000e-06 (expected 1.000e-06)`.
    If the pilot chose 4e-5, the update16 value is `4.000e-05`.
  - The determinism line (item 5) must show `vs v1: identical` and end in `OK`.
  - Every finished evaluation line must end in `OK`.
  - If any line says `CHECK`, or the queue window shows `STOP`, see below.

- [ ] **9. Package and return**

  When the queue window prints `All items finished`, run:

  ```sh
  ./scripts/phase2_queue_isik.sh package
  ```

  If the PI asks you to stop early, press Ctrl-C once in the queue window,
  wait for the prompt, and run the same command.

  This creates `runs/revision_v2/return/socratiq-v2-isik-<time>.tar.gz` and a
  `.sha256` file. Upload both to OneDrive, or copy them to an external drive.

  ✉ **Send the PI:** the link or location and the `.sha256` line. Unless
  something goes wrong, this is the only thing you send.

## If something goes wrong

- **The queue prints `STOP:` or a report line says `CHECK`:** do not rerun,
  edit, delete or move anything. Send the PI the `STOP` message (or the
  `CHECK` line) and attach the log file it names. Restart only when the PI
  says so, with the same `run` command.
- **The Mac restarted or lost power:** open the repository folder and run
  `./scripts/phase2_queue_isik.sh run` again.
  - Finished items, including the pilot and its decision, are kept.
  - An interrupted evaluation continues where it stopped.
  - An interrupted training run is set aside (never deleted) and restarted.
- **Never** edit code or configs, touch `runs/reviewer_rerun/`, or include `.env` in anything you send.

# Checklist for `isik` (revision-v2 runs, 2–5 October 2026)

Work through the boxes in order. Each step gives the command, what you should
see, and what to send the PI. Run every command in Terminal from the
repository folder. Copy commands from this file, not from email, because email
can change quote characters. The background for every step is in
`docs/phase2_runbook.md`.

**Your machine runs** the pilot, then the Qwen3-0.6B pair, the Llama pair, and
the Qwen3-0.6B base model: 45 items, about 41 hours, almost all unattended.

**Hands-on time** is about 45 minutes on Friday, plus waiting for the pilot.
After that, about 15 minutes on Saturday, 5 minutes per check-in, and 15
minutes on Monday.

## Friday

- [ ] **0. Prepare the Mac (5 min)**
  - Plug it in and keep the lid open all weekend.
  - Postpone macOS updates until Monday.
  - Close other heavy programs.
  - Check free space with `df -h .`: at least 100 GB should be available.

- [ ] **1. Open the repository**

  ```sh
  cd "/Users/isik/Desktop/PhD/Jake's Thesis Paper/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
  ```

- [ ] **2. Get the new code**

  ```sh
  git fetch origin
  git checkout revision-v2
  git log -1 --oneline
  ```

  The last line must show the commit given in the PI's email.

- [ ] **3. Archive v1 (10–20 min)**

  ```sh
  .venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-isik.tar
  ```

  This writes `~/Desktop/socratiq-v1-archive-isik.tar` and a `.sha256` file;
  nothing in the repository changes. Copy both files to the external drive or
  shared folder the PI named.

- [ ] **4. Place the Llama checkpoint**

  Put the two files from the PI (`llama3.2-1b-instruct-meta-bf16.tar` and
  `.tar.sha256`) in your Downloads folder, then:

  ```sh
  mkdir -p models/revision_v2
  cp ~/Downloads/llama3.2-1b-instruct-meta-bf16.tar* models/revision_v2/
  cd models/revision_v2 && shasum -a 256 -c llama3.2-1b-instruct-meta-bf16.tar.sha256 && tar -xf llama3.2-1b-instruct-meta-bf16.tar && cd ../..
  .venv/bin/python -m src.revision_v2.llama_base verify models/revision_v2/llama3.2-1b-instruct-meta-bf16
  ```

  You should see `llama3.2-1b-instruct-meta-bf16.tar: OK`, then `"status": "passed"`
  and `"directory_sha256": "4990058eddea56ac54c5800131a87fdb128f69fbfa3a9a9d6bff7d047760e07a"`.

- [ ] **5. Setup check (about 10 min)**

  ```sh
  ./scripts/phase2_setup.sh --machine isik --before-freeze
  ```

  The last line must be `Setup passed on isik. No training or evaluation was started.`

  ✉ **Send the PI:** the line in `~/Desktop/socratiq-v1-archive-isik.tar.sha256`
  and the last 5 lines of the setup output.

- [ ] **6. Run the pilot (about 1 h 45 min)**

  ```sh
  ./scripts/phase2_queue_isik.sh pilot --peak 8e-5
  ```

  At the end it prints a block that contains `"decision": "accept"` or
  `"decision": "reject"`. Then check the learning rate:

  ```sh
  grep -E "^Iter (520|6016): Train" runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/train.log
  ```

  It must show `Learning Rate 8.000e-05` (iteration 520) and
  `Learning Rate 1.000e-06` (iteration 6016).

  ✉ **Send the PI** these three files from
  `runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/`:
  `pilot_check.json`, `train.log` and `manifest.json`.

  **Then stop and wait for the PI's reply.**
  - If the PI asks for the halved rate, run
    `./scripts/phase2_queue_isik.sh pilot --peak 4e-5` and send the same three
    files from `runs/revision_v2/training/qwen3_0.6b_socratic_lr4e-5/`. In
    that run the learning rate at iteration 520 is `4.000e-05`.

## Saturday (after the PI's "go" email)

The email gives you the **approved peak** (`8e-5` or `4e-5`) and the
**frozen commit**. Below, `PEAK` means the approved peak.

- [ ] **7. Switch to the frozen tag**

  ```sh
  git fetch origin --tags
  git checkout protocol-v2-frozen
  git log -1 --oneline
  ```

  The commit must match the email.

- [ ] **8. Setup check again (about 10 min)**

  ```sh
  ./scripts/phase2_setup.sh --machine isik
  ```

  The last line must be `Setup passed on isik. …`

- [ ] **9. Start the queue**

  ```sh
  ./scripts/phase2_queue_isik.sh run --peak PEAK
  ```

  The first line must be `skip (done)  1_core_training/qwen3_0.6b_socratic_lr<PEAK>`:
  your pilot is kept, not retrained. The next item starts training the
  non-Socratic run. Leave this window open and the Mac plugged in.

  ✉ **Send the PI:** the start time and the first 5 lines on screen.

- [ ] **10. Determinism check (about 6.5 h after the start)**

  Run this when item 5 (`2_determinism/…`) shows `done` in the report (step 11):

  ```sh
  .venv/bin/python -m src.revision_v2.queue determinism \
    runs/revision_v2/determinism/isik/qwen3_0.6b_base/multiarith/P0_greedy/predictions.jsonl \
    runs/reviewer_rerun/evaluation/qwen3_0.6b_base/multiarith/greedy/predictions.jsonl
  ```

  This compares the new run with your v1 run of the same prompt. It should
  print `"identical": true` and `"items_compared": 180`.

  ✉ **Send the PI:** the output, with the first file
  (`runs/revision_v2/determinism/isik/qwen3_0.6b_base/multiarith/P0_greedy/predictions.jsonl`)
  attached. The PI compares it with `chee`'s.

## Saturday to Monday: check-ins

- [ ] **11. Report at each check-in: Saturday evening, Sunday morning, Sunday evening**

  Open a second Terminal window in the repository folder:

  ```sh
  ./scripts/phase2_queue_isik.sh report --peak PEAK
  ```

  ✉ **Send the PI** the whole output.

  - Every finished training line must end in `OK`, with these values:
    `LR update16 8.000e-05 (expected 8.000e-05) update188 1.000e-06 (expected 1.000e-06)`.
    With `PEAK` 4e-5, the update16 value is `4.000e-05`.
  - Every finished evaluation line must end in `OK`.
  - If any line says `CHECK`, or the queue window shows `STOP`, go to "If something goes wrong".

  Rough timeline if the queue starts at 09:00 on Saturday:

  | Stage | Finishes around |
  |---|---|
  | Training (items 2–4) | 15:30 Saturday |
  | Core greedy evaluation (items 6–30) | 04:00 Sunday |
  | SC@5 (items 31–45) | 00:30 Monday |

## Monday morning

- [ ] **12. Package and return**

  If the report says `done 45`, or the PI asks you to stop (press Ctrl-C once
  in the queue window and wait for the prompt), run:

  ```sh
  ./scripts/phase2_queue_isik.sh package --peak PEAK
  ./scripts/phase2_queue_isik.sh report --peak PEAK
  ```

  This creates `runs/revision_v2/return/socratiq-v2-isik-<time>.tar.gz` and a
  `.sha256` file. Copy both to the drive or shared folder.

  ✉ **Send the PI:** where you put them, the `.sha256` line, and the final report.

## If something goes wrong

- **The queue prints `STOP:` or a report line says `CHECK`:** do not rerun,
  edit, delete or move anything. Send the PI:
  - the `STOP` message;
  - the log path it names;
  - `runs/revision_v2/queue_logs/isik/queue_events.jsonl`.

  Restart only when the PI says so, with the same `run` command.
- **The Mac restarted or lost power:** open the repository folder and run the
  same `run` command again.
  - Finished items are skipped.
  - An interrupted evaluation continues where it stopped.
  - An interrupted training run is set aside (never deleted) and restarted.

  Tell the PI it happened.
- **Never** edit code or configs, touch `runs/reviewer_rerun/`, or include `.env` in anything you send.

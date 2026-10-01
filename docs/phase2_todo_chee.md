# Checklist for `chee` (revision-v2 runs, 2–5 October 2026)

Work through the boxes in order. Each step gives the command, what you should
see, and what to send the PI. Run every command in Terminal from the
repository folder. Copy commands from this file, not from email, because email
can change quote characters. The background for every step is in
`docs/phase2_runbook.md`.

**Your machine runs** the Qwen3-1.7B pair, the Qwen3-1.7B and Llama base
models, and then the four smaller-data Qwen3-0.6B runs (the second tier): 55
items, about 44 hours, almost all unattended. There is no pilot on your
machine; it waits for the PI's approval of the pilot on `isik`.

**Hands-on time** is about 45 minutes on Friday, about 15 minutes on Saturday,
5 minutes per check-in, and 15 minutes on Monday.

## Friday

- [ ] **0. Prepare the Mac (5 min)**
  - Plug it in and keep the lid open all weekend.
  - Postpone macOS updates until Monday.
  - Close other heavy programs.
  - Check free space with `df -h .`: at least 100 GB should be available.

- [ ] **1. Open the repository**

  ```sh
  cd "/Users/chee/Desktop/Jake/reviewer-rerun-m4-handoff-v3-20260916/socratiq"
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
  .venv/bin/python -m src.revision_v2.queue archive-v1 --destination ~/Desktop/socratiq-v1-archive-chee.tar
  ```

  This writes `~/Desktop/socratiq-v1-archive-chee.tar` and a `.sha256` file;
  nothing in the repository changes. Copy both files to the external drive or
  shared folder the PI named.

- [ ] **4. Place the Llama checkpoint**

  Your machine evaluates the Llama base model. Put the two files from the PI
  (`llama3.2-1b-instruct-meta-bf16.tar` and `.tar.sha256`) in your Downloads
  folder, then:

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
  ./scripts/phase2_setup.sh --machine chee --before-freeze
  ```

  The last line must be `Setup passed on chee. No training or evaluation was started.`

  ✉ **Send the PI:** the line in `~/Desktop/socratiq-v1-archive-chee.tar.sha256`
  and the last 5 lines of the setup output. Then wait for the "go" email.

## Saturday (after the PI's "go" email)

The email gives you the **approved peak** (`8e-5` or `4e-5`) and the
**frozen commit**. Below, `PEAK` means the approved peak. It sets the rate
for your second-tier runs; your Qwen3-1.7B pair always uses 1e-4.

- [ ] **6. Switch to the frozen tag**

  ```sh
  git fetch origin --tags
  git checkout protocol-v2-frozen
  git log -1 --oneline
  ```

  The commit must match the email.

- [ ] **7. Setup check again (about 10 min)**

  ```sh
  ./scripts/phase2_setup.sh --machine chee
  ```

  The last line must be `Setup passed on chee. …`

- [ ] **8. Start the queue**

  ```sh
  ./scripts/phase2_queue_chee.sh run --peak PEAK
  ```

  The first item is `=== 1_core_training/qwen3_1.7b_socratic_lr1e-4 ===`
  (about 4 hours). Leave this window open and the Mac plugged in.

  ✉ **Send the PI:** the start time and the first 5 lines on screen.

- [ ] **9. Determinism check (about 7 h after the start)**

  Run this when item 3 (`2_determinism/…`) shows `done` in the report (step 10):

  ```sh
  .venv/bin/python -m src.revision_v2.queue determinism \
    runs/revision_v2/determinism/chee/qwen3_0.6b_base/multiarith/P0_greedy/predictions.jsonl \
    runs/reviewer_rerun/evaluation_clean_v1/qwen3_0.6b_base/multiarith/greedy/predictions.jsonl
  ```

  This compares the new run with your v1 run of the same prompt. It should
  print `"identical": true` and `"items_compared": 180`.

  ✉ **Send the PI:** the output, with the first file
  (`runs/revision_v2/determinism/chee/qwen3_0.6b_base/multiarith/P0_greedy/predictions.jsonl`)
  attached. The PI compares it with `isik`'s.

## Saturday to Monday: check-ins

- [ ] **10. Report at each check-in: Saturday evening, Sunday morning, Sunday evening**

  Open a second Terminal window in the repository folder:

  ```sh
  ./scripts/phase2_queue_chee.sh report --peak PEAK
  ```

  ✉ **Send the PI** the whole output.

  - Every finished training line must end in `OK`, with these values:
    - Qwen3-1.7B: `LR update16 9.844e-05 (expected 9.844e-05) update188 1.000e-06 (expected 1.000e-06)`.
    - Second-tier Qwen3-0.6B: update16 `8.000e-05` (or `4.000e-05` if `PEAK` is 4e-5), update188 `1.000e-06`.
  - Every finished evaluation line must end in `OK`.
  - If any line says `CHECK`, or the queue window shows `STOP`, go to "If something goes wrong".

  Rough timeline if the queue starts at 09:00 on Saturday:

  | Stage | Finishes around |
  |---|---|
  | Qwen3-1.7B training (items 1–2) | 16:15 Saturday |
  | Core greedy evaluation (items 4–23) | 01:00 Sunday |
  | SC@5 (items 24–35) | 17:00 Sunday |
  | Second-tier training (items 36–39) | 22:00 Sunday |
  | Second-tier evaluation (items 40–55) | 05:00 Monday |

## Monday morning

- [ ] **11. Package and return**

  If the report says `done 55`, or the PI asks you to stop (press Ctrl-C once
  in the queue window and wait for the prompt), run:

  ```sh
  ./scripts/phase2_queue_chee.sh package --peak PEAK
  ./scripts/phase2_queue_chee.sh report --peak PEAK
  ```

  This creates `runs/revision_v2/return/socratiq-v2-chee-<time>.tar.gz` and a
  `.sha256` file. Copy both to the drive or shared folder.

  ✉ **Send the PI:** where you put them, the `.sha256` line, and the final report.

## If something goes wrong

- **The queue prints `STOP:` or a report line says `CHECK`:** do not rerun,
  edit, delete or move anything. Send the PI:
  - the `STOP` message;
  - the log path it names;
  - `runs/revision_v2/queue_logs/chee/queue_events.jsonl`.

  Restart only when the PI says so, with the same `run` command.
- **The Mac restarted or lost power:** open the repository folder and run the
  same `run` command again.
  - Finished items are skipped.
  - An interrupted evaluation continues where it stopped.
  - An interrupted training run is set aside (never deleted) and restarted.

  Tell the PI it happened.
- **Never** edit code or configs, touch `runs/reviewer_rerun/`, or include `.env` in anything you send.

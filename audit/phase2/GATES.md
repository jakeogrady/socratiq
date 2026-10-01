# Phase 2a gates: evidence

All seven gates pass. They were run on 2 October 2026 (from 00:54 Irish time)
at commit `c2e41ca` on branch `revision-v2`, on the PI machine (MacBook Pro
M2 Max, macOS 14.6.1). No training or evaluation was run: the gates load
tokenizers and compare files only.

To reproduce, run `make phase2-gates` on the PI machine. Gate 4 needs Meta's
file in `audit/downloads/`. The machine-readable results are in
`audit/phase2/gates.json`.

| # | Requirement | Result | Key numbers | Evidence |
|---|---|---|---|---|
| 1 | Schedule replay: peak at update 16 (update 0 for 1.7B), 1e-6 at update 188 | PASS | All 18 configs: 188 updates, 0 discarded micro-batches. 0.6B/Llama: update *k* ≤ 16 gets *k*/16 of the peak, peak 8e-5 at update 16, 1e-6 at update 188. 1.7B: 1e-4 at update 1 (scheduler step 0), 1e-6 at update 188 | `gate1_schedule_replay.json` (rate at every update, per config) |
| 2 | First training row of each arm and model rendered and saved as text | PASS | 6 renders through MLX-LM's own `CompletionsDataset.process`. Decoded tokens equal the chat-template text, and the loss covers the whole sequence | `gate2_training_renders/*.txt`, `index.json` |
| 3 | Llama training and evaluation renders show the fixed date | PASS | Training (both arms) and evaluation (P0 GSM8K, P0 MultiArith, P4 GSM8K) all read `Today Date: 26 Jul 2024`, rendered on 1–2 October 2026. The template no longer references `strftime_now` | `gate3_llama_date_renders/` |
| 4 | Llama base equals Meta's weights tensor by tensor | PASS | 146/146 tensors byte-identical (2,471,628,800 parameter bytes). The Meta file matches the SHA-256 Meta publishes (`1ff795ff…8538f`). Checkpoint hashes match `configs/revision_v2/llama_base_manifest.json` | `gate4_llama_base_vs_meta.json` |
| 5 | Pairing audit passes after the shuffle and the `??` fix | PASS | 20,000 pairs, plus both subsets. Non-Socratic rows identical to v1 (20,000/20,000). Socratic rows differ from v1 only by collapsed `??` (1,581 rows changed, 4,535 questions). Same 18,000/2,000 split, 0 sources across splits, subsets exact (5,000 and 10,000), nested, whole source groups. A rebuild reproduces every tracked hash | `gate5_pairing_audit.json` |
| 6 | Scorer unit tests cover all three rules | PASS | 18 tests over the three rules, the vote and the diagnostics. The frozen primary rule reproduces all 49,888 stored v1 decisions (64 runs) with 0 mismatches | `gate6_scorer_tests.txt`, `gate6_v1_equivalence.json` |
| 7 | Evaluation dry run on five items writes the full manifest | PASS | 4 dry runs, each with 5 items and 24/24 required manifest fields: Qwen3-0.6B P0 GSM8K, Qwen3-1.7B P4 GSM8K, Qwen3-0.6B P0 GSM-Hard, Llama P0 MultiArith SC@5. They exercise pinned model revisions, the local Llama checkpoint, all prompts and both modes. A training dry run (Llama) is included | `gate7_dry_runs/` |

## Notes

1. **Update numbering.** mlx-lm evaluates the schedule on its update counter
   *s* = 0…187, so update *k* uses *s* = *k* − 1.
   - The brief's "update 0 for 1.7B" is the first update (*s* = 0).
   - Because of how mlx-lm joins warm-up and cosine, the cosine starts at
     the peak, so updates 16 and 17 both run at the peak.
2. **Meta's weights.** No Hugging Face token was available for Meta's gated
   repository. Meta's file was taken from an ungated mirror and accepted only
   because its SHA-256 equals the hash Meta publishes. The checkpoint uses
   `mlx-community/Llama-3.2-1B-Instruct-bf16@863c846`: its weights are
   byte-identical to Meta's, and its tokenizer files have the same git blob
   IDs as Meta's.
3. **Llama on the students' machines.** Each machine runs
   `python -m src.revision_v2.llama_base fetch`. It downloads that pinned
   repository, applies the date fix and refuses unless every file matches the
   manifest. Both machines therefore hold the byte-identical checkpoint
   without a file transfer. A test fetch on the PI machine took about 1.5
   minutes and reproduced the directory hash `4990058e…`.
4. **Date mechanism.** The only edit to the checkpoint is in
   `tokenizer_config.json`, where the date block becomes
   `{%- set date_string = "26 Jul 2024" %}`. Training (`mlx_lm.lora`) and
   evaluation both read this one file, so both use the same mechanism and no
   caller can override it.
5. **Pilot decision.** There is no separate approval step (PI decision,
   2 October 2026).
   - On `isik`, the queue trains the pilot at 8e-5 and applies the protocol
     rule. If the pilot fails, it repeats it once at 4e-5.
   - The first accepted peak is recorded in `runs/revision_v2/pilot_decision.json`
     and used for every Qwen3-0.6B and Llama run.
   - If both peaks fail, the queue stops for the PI.
   - The tag `protocol-v2-frozen` is therefore set before the pilot, on the
     commit both machines run.
6. **Row shuffle.** Seed 2026 for rows and 2027 for subsets. Validation rows
   are also shuffled.
7. **SC@5 tie-break.** The greedy tie-break response is generated when the
   vote of *any* rule ties. The primary rule is unchanged from v1.
8. **Second tier evaluation.** Set to P0 greedy on all four benchmarks, with
   no P4 and no SC@5.
9. **Machine split.** This is the brief's default, rebalanced by moving only
   the Llama base-model evaluations to `chee`. Expected time is 41.0 h on
   `isik` and 44.0 h on `chee`.
10. **Determinism.** Each queue compares its MultiArith check with that
    machine's own v1 run of the same prompt. The two v1 runs are themselves
    identical (180/180). The PI compares the two v2 runs from the returned
    packets.
11. **Manifest privacy.** v2 manifests record the commit, the branch and
    tracked-file changes only, never untracked file names.

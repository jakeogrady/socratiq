"""Run the seven Phase 2a gates and write their evidence to audit/phase2/.

Gate 1  schedule replay: peak at update 16 (update 1, i.e. scheduler step 0, for
        Qwen3-1.7B) and 1e-6 at update 188, for every configuration
Gate 2  first training row of each arm and model, rendered through MLX-LM's own
        training code path and saved as text
Gate 3  Llama training and evaluation renders both show the fixed date
Gate 4  the Llama base equals Meta's weights tensor by tensor
Gate 5  pairing audit after the shuffle and the "??" fix
Gate 6  scorer unit tests for all three rules, plus agreement with every stored
        v1 decision
Gate 7  evaluation dry runs on five items write the full manifest

This is a PI-machine command. It loads tokenizers only; it never loads model
weights for generation and never trains.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.paired_dataset import CanonicalExample, render_answer
from src.rerun_utils import file_sha256, object_sha256, read_jsonl, utc_now, write_json
from src.revision_v2 import configs, data, evaluate, llama_base, scoring, train
from src.revision_v2.protocol import REPOSITORY_ROOT, load_protocol

EVIDENCE = REPOSITORY_ROOT / "audit/phase2"
META_REFERENCE_GLOB = (
    "audit/downloads/llama3.2_1b_meta_weights_via_unsloth_mirror@*/model.safetensors"
)
DATE_LINE = re.compile(r"Today Date: ([^\n]+)\n")
WARMUP_UPDATES = 16
OPTIMIZER_UPDATES = 188
END_LR = 1e-6
LR_TOLERANCE = 1e-12
PEAK_RELATIVE_TOLERANCE = 1e-6
CANONICAL_ROWS = 20000
DRY_RUN_ITEMS = 5


def tokenizer_for(model_key: str) -> Any:
    """Load the pinned tokenizer for a protocol model (tokenizer files only)."""
    model = load_protocol()["models"][model_key]
    path, _ = evaluate.materialize_model(
        model.get("local_path", model["identifier"]), dry_run=True
    )
    return evaluate.load_tokenizer_only(path)


def gate1() -> dict[str, Any]:
    """Replay every configuration's schedule and check the peak and final rates."""
    results, ok = {}, True
    for run in configs.training_runs():
        replay = configs.replay_schedule(train.load_config(Path(run["config_path"])))
        lr = replay["lr_by_update"]
        if run["model_key"] == "qwen3_1.7b":
            passed = (
                replay["peak_updates"] == [1]
                and abs(lr[0] - run["peak"]) <= run["peak"] * PEAK_RELATIVE_TOLERANCE
            )
        else:
            passed = (
                replay["peak_updates"][0] == WARMUP_UPDATES
                and abs(lr[WARMUP_UPDATES - 1] - run["peak"])
                <= run["peak"] * PEAK_RELATIVE_TOLERANCE
                and all(
                    abs(lr[k - 1] - run["peak"] * k / WARMUP_UPDATES)
                    <= run["peak"] * PEAK_RELATIVE_TOLERANCE
                    for k in range(1, WARMUP_UPDATES + 1)
                )
            )
        passed = (
            passed
            and replay["updates"] == OPTIMIZER_UPDATES
            and replay["discarded_microbatches"] == 0
            and abs(lr[-1] - END_LR) <= LR_TOLERANCE
        )
        ok &= passed
        results[run["run_id"]] = {
            "passed": passed,
            "updates": replay["updates"],
            "discarded_microbatches": replay["discarded_microbatches"],
            "peak_lr": replay["peak_lr"],
            "peak_updates": replay["peak_updates"],
            "lr_update_1": lr[0],
            "lr_update_16": lr[15],
            "lr_update_17": lr[16],
            "lr_update_188": lr[-1],
            "logged_iteration_520_update_16": f"{replay['logged_lr_by_iteration'][520]:.3e}",
            "logged_iteration_6016_update_188": f"{replay['logged_lr_by_iteration'][6016]:.3e}",
            "lr_by_update": lr,
        }
    write_json(
        EVIDENCE / "gate1_schedule_replay.json",
        {
            "checked_at": utc_now(),
            "convention": "update k (1-based) uses scheduler step s = k - 1",
            "runs": results,
        },
    )
    return {
        "passed": ok,
        "evidence": "audit/phase2/gate1_schedule_replay.json",
        "configs": len(results),
    }


def training_render(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Render one row exactly as MLX-LM's CompletionsDataset does for training."""
    from mlx_lm.tuner.datasets import CompletionsDataset

    tokens, offset = CompletionsDataset(
        [row], tokenizer, "question", "answer", mask_prompt=False
    ).process(row)
    text = tokenizer.decode(tokens)
    template_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        tokenize=False,
    )
    return {
        "text": text,
        "tokens": len(tokens),
        "loss_offset": offset,
        "token_ids_sha256": object_sha256(tokens),
        "decode_equals_template_text": text == template_text,
    }


def gate2() -> dict[str, Any]:
    """Render the first training row of each arm and model and save it as text."""
    out_dir = EVIDENCE / "gate2_training_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    index, ok = {}, True
    for model_key in ("qwen3_0.6b", "qwen3_1.7b", "llama3.2_1b"):
        tokenizer = tokenizer_for(model_key)
        for arm in data.ARMS:
            row = next(read_jsonl(Path(f"data/revision_v2/full/{arm}/train.jsonl")))
            render = training_render(tokenizer, row)
            name = f"{model_key}__{arm}__train_row0.txt"
            (out_dir / name).write_text(render["text"], encoding="utf-8")
            ok &= render["decode_equals_template_text"] and render["loss_offset"] == 0
            index[name] = {k: v for k, v in render.items() if k != "text"} | {
                "example_id": row["example_id"]
            }
    write_json(out_dir / "index.json", index)
    return {
        "passed": ok,
        "evidence": "audit/phase2/gate2_training_renders/",
        "renders": len(index),
    }


def gate3() -> dict[str, Any]:
    """Check that Llama training and evaluation renders carry the fixed date."""
    tokenizer = tokenizer_for("llama3.2_1b")
    template = tokenizer.chat_template
    renders = {}
    for arm in data.ARMS:
        row = next(read_jsonl(Path(f"data/revision_v2/full/{arm}/train.jsonl")))
        renders[f"training_{arm}"] = training_render(tokenizer, row)["text"]
    for prompt, benchmark in (("P0", "gsm8k"), ("P0", "multiarith"), ("P4", "gsm8k")):
        plan = evaluate.evaluation_plan(benchmark, prompt, "greedy")
        semantic = evaluate.build_prompt(
            "Ann has 2 apples and gets 3 more. How many apples?",
            [{"question": "Q?", "answer": "#### 1"}]
            * len(plan["spec"].few_shot_indices),
            question_column="question",
            answer_column="answer",
        )
        renders[f"evaluation_{prompt}_{benchmark}"] = evaluate.render_model_prompt(
            tokenizer, semantic, enable_thinking=False
        )
    dates = {name: DATE_LINE.findall(text) for name, text in renders.items()}
    ok = "strftime_now" not in template and all(
        found == [llama_base.FIXED_DATE] for found in dates.values()
    )
    out_dir = EVIDENCE / "gate3_llama_date_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in renders.items():
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    write_json(
        out_dir / "index.json",
        {
            "checked_at": utc_now(),
            "rendered_on_date": utc_now()[:10],
            "template_references_strftime_now": "strftime_now" in template,
            "dates_found": dates,
        },
    )
    return {
        "passed": ok,
        "evidence": "audit/phase2/gate3_llama_date_renders/",
        "dates_found": sorted({d for found in dates.values() for d in found}),
    }


def gate4() -> dict[str, Any]:
    """Compare the Llama base with Meta's weights tensor by tensor."""
    meta_reference = next(REPOSITORY_ROOT.glob(META_REFERENCE_GLOB))
    local = REPOSITORY_ROOT / load_protocol()["models"]["llama3.2_1b"]["local_path"]
    comparison = llama_base.compare_safetensors(
        meta_reference, local / "model.safetensors"
    )
    verification = llama_base.verify_checkpoint(local)
    manifest = json.loads(
        (REPOSITORY_ROOT / llama_base.DEFAULT_MANIFEST).read_text("utf-8")
    )
    meta_ok = (
        comparison["left_sha256"]
        == manifest["meta_reference"]["model_safetensors_sha256"]
    )
    write_json(
        EVIDENCE / "gate4_llama_base_vs_meta.json",
        {
            "checked_at": utc_now(),
            "meta_file_matches_published_sha256": meta_ok,
            "checkpoint_verification": verification,
            **comparison,
        },
    )
    passed = (
        comparison["all_tensors_bitwise_equal"]
        and meta_ok
        and verification["status"] == "passed"
    )
    return {
        "passed": passed,
        "evidence": "audit/phase2/gate4_llama_base_vs_meta.json",
        "tensors_bitwise_equal": f"{comparison['tensors_bytes_equal']}/{comparison['tensors_compared']}",
    }


def gate5() -> dict[str, Any]:
    """Audit pairing after the shuffle and the "??" fix, and cross-check against v1."""
    audit = data.audit_written(Path("data/revision_v2"))
    rebuild = data.build(Path("data/revision_v2"), write_manifest=False)
    canonical = {
        r["example_id"]: CanonicalExample.from_mapping(r)
        for r in read_jsonl(Path(load_protocol()["data"]["canonical_input"]["path"]))
    }
    cross = {
        "rows": 0,
        "non_socratic_identical_to_v1": 0,
        "socratic_identical_to_v1": 0,
        "socratic_differs_only_by_collapsed_question_marks": 0,
    }
    for split in data.SPLITS:
        v1 = {
            arm: {
                r["example_id"]: r
                for r in read_jsonl(Path(f"data/reviewer_rerun/{arm}/{split}.jsonl"))
            }
            for arm in data.ARMS
        }
        for soc, non in zip(
            read_jsonl(Path(f"data/revision_v2/full/socratic/{split}.jsonl")),
            read_jsonl(Path(f"data/revision_v2/full/non_socratic/{split}.jsonl")),
            strict=True,
        ):
            record = canonical[soc["example_id"]]
            cross["rows"] += 1
            cross["non_socratic_identical_to_v1"] += (
                non == v1["non_socratic"][non["example_id"]]
            )
            old = v1["socratic"][soc["example_id"]]
            cross["socratic_identical_to_v1"] += soc == old
            expected = render_answer(record, socratic=True)
            for step in record.solution_steps:
                expected = expected.replace(
                    step.guiding_question + "\n",
                    data.collapse_terminal_question_marks(step.guiding_question) + "\n",
                    1,
                )
            cross["socratic_differs_only_by_collapsed_question_marks"] += (
                old["answer"] == render_answer(record, socratic=True)
                and soc["answer"] == expected
            )
    write_json(
        EVIDENCE / "gate5_pairing_audit.json",
        {
            "checked_at": utc_now(),
            "audit_of_written_files": audit,
            "deterministic_rebuild": {k: v for k, v in rebuild.items() if k != "files"},
            "cross_check_against_v1_files": cross,
        },
    )
    passed = (
        audit["status"] == "passed"
        and rebuild["status"] == "passed"
        and cross["non_socratic_identical_to_v1"] == cross["rows"] == CANONICAL_ROWS
        and cross["socratic_differs_only_by_collapsed_question_marks"] == cross["rows"]
    )
    return {
        "passed": passed,
        "evidence": "audit/phase2/gate5_pairing_audit.json",
        **{k: cross[k] for k in cross},
    }


def v1_equivalence() -> dict[str, Any]:
    """Check that the frozen primary rule reproduces every stored v1 decision."""
    roots = [
        REPOSITORY_ROOT
        / "results/add-runs/reviewer-rerun-clean-v1-results/runs/reviewer_rerun/evaluation_clean_v1",
        REPOSITORY_ROOT
        / "results/add-runs/reviewer-rerun-gsmhard-v1-results/runs/reviewer_rerun/evaluation_gsmhard_v1",
    ]
    decisions = mismatches = runs = 0
    for root in roots:
        for path in sorted(root.glob("*/*/*/predictions.jsonl")):
            runs += 1
            for row in read_jsonl(path):
                answers = [scoring.last_marked(s["response"]) for s in row["samples"]]
                tie = row.get("greedy_tie_break")
                predicted, _ = scoring.majority_vote(
                    answers,
                    greedy_answer=scoring.last_marked(tie["response"]) if tie else None,
                )
                decisions += 1
                mismatches += (
                    predicted != row["predicted_answer"]
                    or scoring.answers_equal(predicted, row["reference_answer"])
                    != row["correct"]
                )
    return {"runs": runs, "decisions": decisions, "mismatches": mismatches}


def gate6() -> dict[str, Any]:
    """Run the scorer unit tests and check agreement with every stored v1 decision."""
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_revision_v2_scoring"
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (EVIDENCE / "gate6_scorer_tests.txt").write_text(
        stream.getvalue(), encoding="utf-8"
    )
    equivalence = v1_equivalence()
    write_json(
        EVIDENCE / "gate6_v1_equivalence.json",
        {
            "checked_at": utc_now(),
            "scorer_code_sha256": file_sha256(Path(scoring.__file__)),
            **equivalence,
        },
    )
    passed = (
        result.wasSuccessful()
        and result.testsRun > 0
        and equivalence["mismatches"] == 0
        and equivalence["decisions"] > 0
    )
    return {
        "passed": passed,
        "evidence": "audit/phase2/gate6_scorer_tests.txt, audit/phase2/gate6_v1_equivalence.json",
        "tests_run": result.testsRun,
        **equivalence,
    }


REQUIRED_MANIFEST_FIELDS = (
    "status",
    "protocol.sha256",
    "configuration.evaluator_code_sha256",
    "configuration.scorer_code_sha256",
    "configuration.prompt",
    "configuration.mode",
    "configuration.decoding",
    "configuration.max_tokens",
    "configuration_sha256",
    "base_model",
    "dataset.requested_revision",
    "seeds.base_seed",
    "environment.machine.hostname",
    "environment.platform",
    "environment.packages.mlx",
    "environment.packages.mlx-lm",
    "environment.git.commit",
    "summary.rules.last_marked",
    "summary.rules.first_marked",
    "summary.rules.last_marked_fallback_last_number",
    "summary.diagnostics_responses",
    "summary.finish_reasons",
    "summary.response_tokens",
    "prediction_sha256",
)
DRY_RUNS = (
    ("qwen3_0.6b_base", "mlx-community/Qwen3-0.6B-bf16", "gsm8k", "P0", "greedy"),
    ("qwen3_1.7b_base", "mlx-community/Qwen3-1.7B-4bit", "gsm8k", "P4", "greedy"),
    ("qwen3_0.6b_base", "mlx-community/Qwen3-0.6B-bf16", "gsm_hard", "P0", "greedy"),
    (
        "llama3.2_1b_base",
        "models/revision_v2/llama3.2-1b-instruct-meta-bf16",
        "multiarith",
        "P0",
        "sc5",
    ),
)


def lookup(mapping: dict[str, Any], dotted: str) -> Any:
    """Return a nested value addressed as a.b.c."""
    value: Any = mapping
    for part in dotted.split("."):
        value = value[part]
    return value


def gate7() -> dict[str, Any]:
    """Run five-item evaluation dry runs and check that each writes the full manifest."""
    root = EVIDENCE / "gate7_dry_runs"
    results, ok = {}, True
    for condition, model, benchmark, prompt, mode in DRY_RUNS:
        run_dir = root / condition / benchmark / f"{prompt}_{mode}"
        if run_dir.exists():
            for item in run_dir.iterdir():
                item.unlink()
        command = [
            sys.executable,
            "-m",
            "src.revision_v2.evaluate",
            "--condition-id",
            condition,
            "--model",
            model,
            "--benchmark",
            benchmark,
            "--prompt",
            prompt,
            "--mode",
            mode,
            "--run-dir",
            str(run_dir.relative_to(REPOSITORY_ROOT)),
            "--limit",
            str(DRY_RUN_ITEMS),
            "--dry-run",
        ]
        completed = subprocess.run(
            command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
        )
        manifest = (
            json.loads((run_dir / "manifest.json").read_text("utf-8"))
            if (run_dir / "manifest.json").is_file()
            else {}
        )
        missing = []
        for field in REQUIRED_MANIFEST_FIELDS:
            try:
                if lookup(manifest, field) in (None, ""):
                    missing.append(field)
            except (KeyError, TypeError):
                missing.append(field)
        rows = (
            list(read_jsonl(run_dir / "predictions.jsonl"))
            if (run_dir / "predictions.jsonl").is_file()
            else []
        )
        passed = (
            completed.returncode == 0
            and manifest.get("status") == "dry_run_completed"
            and not missing
            and len(rows) == DRY_RUN_ITEMS
        )
        ok &= passed
        results[f"{condition}/{benchmark}/{prompt}_{mode}"] = {
            "passed": passed,
            "return_code": completed.returncode,
            "missing_fields": missing,
            "rows": len(rows),
            "stderr_tail": completed.stderr[-500:],
        }
    training = train.run(
        Path("configs/revision_v2/training/llama3.2_1b_socratic_lr8e-5.yaml"),
        dry_run=True,
    )
    write_json(root / "training_dry_run_llama3.2_1b_socratic_lr8e-5.json", training)
    write_json(
        root / "index.json",
        {
            "checked_at": utc_now(),
            "required_fields": list(REQUIRED_MANIFEST_FIELDS),
            "runs": results,
        },
    )
    return {
        "passed": ok,
        "evidence": "audit/phase2/gate7_dry_runs/",
        "dry_runs": len(results),
    }


GATES = {1: gate1, 2: gate2, 3: gate3, 4: gate4, 5: gate5, 6: gate6, 7: gate7}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested gates and update audit/phase2/gates.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gates", nargs="*", type=int, default=sorted(GATES))
    args = parser.parse_args(argv)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE / "gates.json"
    summary = (
        json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else {}
    )
    for number in args.gates:
        result = GATES[number]()
        result["checked_at"] = utc_now()
        result["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        summary[f"gate{number}"] = result
        print(
            f"gate {number}: {'PASS' if result['passed'] else 'FAIL'}  {json.dumps({k: v for k, v in result.items() if k not in {'passed'}})}"
        )
    write_json(summary_path, summary)
    return 0 if all(summary[f"gate{n}"]["passed"] for n in args.gates) else 1


if __name__ == "__main__":
    sys.exit(main())

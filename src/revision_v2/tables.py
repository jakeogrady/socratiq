"""Evidence for the revision-v2 manuscript: every reported number, from the returned runs.

Reads runs/revision_v2/ (both machines' return packets unpacked together) and
writes to the evidence directory:

  accuracy.csv           one row per evaluation and scoring rule: counts,
                         accuracy, Wilson 95% interval, answer outcomes
  diagnostics.csv        one row per evaluation: format diagnostics, finish
                         reasons, response lengths
  paired.csv             every Socratic-minus-non-Socratic comparison on the
                         same items. family "primary" is the 12 pre-registered
                         tests (Holm-adjusted). "sensitivity" is the same 12
                         under the two other rules. Everything else is
                         "secondary": unadjusted and descriptive
  training.csv           one row per training run: machine, time, memory,
                         adapter size, validation loss, logged learning rates
  evaluation_timing.csv  wall time per evaluation
  determinism.json       the cross-machine and v1 determinism comparisons
  tables/*.tex           LaTeX tables that the manuscript and supplement include
  MANIFEST.json          SHA-256 of every input and output, the commit, the
                         protocol, and the integrity checks that passed

Every response is re-scored from its raw text with the frozen scorer. The
result must equal the stored decision and the evaluator's manifest summary,
and the primary family must equal src.revision_v2.stats. Any mismatch stops
the build. Nothing here reads v1 runs, except the v1 comparison files that the
queues wrote on the students' machines.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.rerun_utils import file_sha256, read_jsonl, utc_now, write_json
from src.revision_v2 import scoring, stats
from src.revision_v2.protocol import (
    REPOSITORY_ROOT,
    RUNS_ROOT,
    load_protocol,
    protocol_identity,
)

DEFAULT_OUTPUT = REPOSITORY_ROOT / "manuscript/revision-v2/evidence"
PACKET_DIRECTORY = REPOSITORY_ROOT / "results/reviewer-rerun-v2"
FREEZE_TAG = "protocol-v2-frozen"
FROZEN_CODE = ("src/revision_v2/evaluate.py", "src/revision_v2/scoring.py")

MODELS = {
    "qwen3_0.6b": "Qwen3-0.6B",
    "qwen3_1.7b": "Qwen3-1.7B",
    "llama3.2_1b": "Llama-3.2-1B",
}
BENCHMARKS = {
    "gsm8k": "GSM8K",
    "multiarith": "MultiArith",
    "svamp": "SVAMP",
    "gsm_hard": "GSM-Hard",
}
ARMS = {"base": "base", "socratic": "Socratic", "non_socratic": "non-Socratic"}
DATA_SIZES = {"full": "18,000", "10k": "10,000", "5k": "5,000", "none": "--"}
# Manuscript labels for the two M4 Pro machines (queue labels in the CSV files).
MACHINE_LABELS = {"isik": "A", "chee": "B"}
RULE_LABELS = {
    "last_marked": "last marked (primary)",
    "first_marked": "first marked",
    "last_marked_fallback_last_number": "last marked, fallback to last number",
}
RULE_SHORT = {
    "last_marked": "Last marked",
    "first_marked": "First marked",
    "last_marked_fallback_last_number": "Fallback",
}
PRIMARY = scoring.PRIMARY_RULE
FALLBACK = "last_marked_fallback_last_number"
DIAGNOSTIC_KEYS = ("bare_answer", "limit_hit", "repetition_loop", "question_lines")
TOKEN_MEAN_TOLERANCE = 1e-9
BYTES_PER_MB = 1_000_000
SECONDS_PER_MINUTE = 60
P_TWO_DECIMALS = 0.1
P_FLOOR = 0.0001


class EvidenceError(RuntimeError):
    """Raised when the returned runs disagree with themselves or the protocol."""


# --------------------------------------------------------------------------
# Number formatting. Half-up rounding of exact values, so a reader who
# recomputes a percentage from its counts gets the printed digits.


def fixed(value: float | Fraction, places: int = 1) -> str:
    """Round half up to a fixed number of decimal places."""
    if isinstance(value, Fraction):
        exact = Decimal(value.numerator) / Decimal(value.denominator)
    else:
        exact = Decimal(repr(value))
    text = str(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
    return "0." + "0" * places if text == "-0." + "0" * places else text


def signed(value: float | Fraction, places: int = 1) -> str:
    """Fixed-point text with an explicit sign for non-zero values."""
    text = fixed(value, places)
    return text if text.startswith("-") or float(text) == 0 else "+" + text


def p_text(p: float) -> str:
    """Two decimals from 0.1 up, otherwise two significant figures."""
    if p < P_FLOOR:
        return "<0.0001"
    if p >= P_TWO_DECIMALS:
        return fixed(p, 2)
    digits = 1 - math.floor(math.log10(p))
    return fixed(p, digits)


def percent(count: int, total: int) -> Fraction:
    """Exact percentage."""
    return Fraction(100 * count, total)


def tex_number(text: str) -> str:
    """Typeset a signed or negative number in math mode, others unchanged."""
    return f"${text}$" if text[:1] in "+-" else text


def tex_interval(low: str, high: str) -> str:
    """Format a bracketed interval with proper minus signs."""
    return f"[{tex_number(low)}, {tex_number(high)}]"


# --------------------------------------------------------------------------
# Loading and re-scoring.


def parse_condition(condition_id: str) -> dict[str, str]:
    """qwen3_0.6b_socratic_5k_lr8e-5 -> model qwen3_0.6b, arm socratic, data 5k."""
    stem = condition_id.split("_lr")[0]
    model = next((m for m in MODELS if stem.startswith(m + "_")), None)
    if model is None:
        raise EvidenceError(f"unknown model in condition {condition_id}")
    rest, data = stem[len(model) + 1 :], "full"
    for size in ("5k", "10k"):
        if rest.endswith("_" + size):
            rest, data = rest[: -len(size) - 1], size
    if rest not in ARMS:
        raise EvidenceError(f"unknown arm in condition {condition_id}")
    return {"model": model, "arm": rest, "data": "none" if rest == "base" else data}


@dataclass
class Evaluation:
    """One finished evaluation directory, re-scored."""

    run_dir: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    flags: dict[str, list[tuple[bool, bool]]]  # rule -> per item (valid, correct)
    condition_id: str
    model: str
    arm: str
    data: str
    benchmark: str
    prompt: str
    mode: str

    @property
    def machine(self) -> str:
        """Machine label recorded by the evaluator."""
        return self.manifest["environment"]["machine"]["label"]

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        """Pairing key shared by the two arms of a matched comparison."""
        return (self.model, self.data, self.benchmark, self.prompt, self.mode)


def rescore_row(row: dict[str, Any], where: str) -> dict[str, tuple[bool, bool]]:
    """Re-extract every answer from raw text and re-decide every rule."""
    for sample in [*row["samples"], *([row["tie_break"]] if row["tie_break"] else [])]:
        if scoring.extract_all(sample["response"]) != sample["extracted"]:
            raise EvidenceError(f"{where}: stored extraction differs from the scorer")
        diagnosis = scoring.diagnose(sample["response"], sample["finish_reason"])
        if diagnosis != sample["diagnostics"]:
            raise EvidenceError(f"{where}: stored diagnostics differ from the scorer")
    result = {}
    for rule in scoring.RULES:
        answers = [scoring.RULES[rule](s["response"]) for s in row["samples"]]
        tie = row["tie_break"]
        greedy_answer = scoring.RULES[rule](tie["response"]) if tie else None
        predicted, _ = scoring.majority_vote(answers, greedy_answer=greedy_answer)
        correct = scoring.answers_equal(predicted, row["reference_answer"])
        stored = row["decisions"][rule]
        if (predicted, predicted is not None, correct) != (
            stored["predicted_answer"],
            stored["valid_answer"],
            stored["correct"],
        ):
            raise EvidenceError(
                f"{where}: stored {rule} decision differs from re-scoring"
            )
        result[rule] = (predicted is not None, correct)
    if result[PRIMARY][1] != row["correct"]:
        raise EvidenceError(f"{where}: top-level correct flag is not the primary rule")
    return result


def load_evaluation(run_dir: Path) -> Evaluation:
    """Load, check and re-score one evaluation directory."""
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    predictions = run_dir / "predictions.jsonl"
    if manifest.get("status") != "completed":
        raise EvidenceError(f"{run_dir} is not completed")
    if manifest["prediction_sha256"] != file_sha256(predictions):
        raise EvidenceError(f"{run_dir}: prediction hash differs from the manifest")
    rows = sorted(read_jsonl(predictions), key=lambda r: r["example_index"])
    expected = manifest["range"]["expected_rows"]
    if [r["example_index"] for r in rows] != list(range(expected)):
        raise EvidenceError(f"{run_dir}: items are missing or duplicated")
    configuration = manifest["configuration"]
    flags: dict[str, list[tuple[bool, bool]]] = {rule: [] for rule in scoring.RULES}
    for row in rows:
        decided = rescore_row(row, f"{run_dir}#{row['example_index']}")
        for rule, flag in decided.items():
            flags[rule].append(flag)
    condition = parse_condition(configuration["condition_id"])
    return Evaluation(
        run_dir=run_dir,
        manifest=manifest,
        rows=rows,
        flags=flags,
        condition_id=configuration["condition_id"],
        benchmark=configuration["benchmark"]["key"],
        prompt=configuration["prompt"],
        mode=configuration["mode"],
        **condition,
    )


def responses(evaluation: Evaluation) -> list[dict[str, Any]]:
    """Every generated response (one per item for greedy, five for SC@5)."""
    return [s for r in evaluation.rows for s in r["samples"]]


def check_summary(evaluation: Evaluation) -> None:
    """Check that the re-scored counts equal what the evaluator wrote."""
    summary, where = evaluation.manifest["summary"], evaluation.run_dir
    for rule, flags in evaluation.flags.items():
        stored = summary["rules"][rule]
        valid = sum(v for v, _ in flags)
        correct = sum(c for _, c in flags)
        if (stored["correct"], stored["valid"], stored["invalid"]) != (
            correct,
            valid,
            len(flags) - valid,
        ):
            raise EvidenceError(f"{where}: {rule} counts differ from the manifest")
    sampled = responses(evaluation)
    counts = {k: sum(s["diagnostics"][k] for s in sampled) for k in DIAGNOSTIC_KEYS}
    counts["invalid_primary"] = sum(s["extracted"][PRIMARY] is None for s in sampled)
    if counts != summary["diagnostics_responses"]:
        raise EvidenceError(f"{where}: diagnostics differ from the manifest")
    if dict(Counter(s["finish_reason"] for s in sampled)) != summary["finish_reasons"]:
        raise EvidenceError(f"{where}: finish reasons differ from the manifest")
    mean = sum(s["response_tokens"] for s in sampled) / len(sampled)
    if abs(mean - summary["response_tokens"]["mean"]) > TOKEN_MEAN_TOLERANCE:
        raise EvidenceError(f"{where}: mean response tokens differ from the manifest")


def tag_commit() -> str:
    """Commit of the freeze tag."""
    return git("rev-parse", f"{FREEZE_TAG}^{{commit}}")


def git(*args: str) -> str:
    """Run git in the repository and return stripped output."""
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def frozen_code_hashes() -> dict[str, str]:
    """SHA-256 of the evaluator and scorer as committed at the freeze tag."""
    import hashlib

    result = {}
    for path in FROZEN_CODE:
        blob = subprocess.run(
            ["git", "show", f"{FREEZE_TAG}:{path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        result[path] = hashlib.sha256(blob).hexdigest()
    return result


def check_provenance(
    evaluation: Evaluation, commit: str, code: dict[str, str], adapters: dict[str, str]
) -> None:
    """Protocol, commit, frozen code, adapter and prompt checks for one evaluation."""
    manifest, where = evaluation.manifest, evaluation.run_dir
    configuration = manifest["configuration"]
    environment = manifest["environment"]
    protocol = load_protocol()
    checks = {
        "protocol": configuration["protocol_sha256"] == protocol_identity()["sha256"],
        "commit": environment["git"]["commit"] == commit,
        "tracked tree": environment["git"]["status_porcelain_tracked_only"]
        in ("", "M .DS_Store"),
        "evaluator code": configuration["evaluator_code_sha256"]
        == code["src/revision_v2/evaluate.py"],
        "scorer code": configuration["scorer_code_sha256"]
        == code["src/revision_v2/scoring.py"],
        "not a dry run": configuration["dry_run"] is False
        and configuration["limit"] is None
        and configuration["start_index"] == 0,
        "max tokens": configuration["max_tokens"]
        == protocol["evaluation"]["max_new_tokens"],
        "thinking off": configuration["enable_thinking"] is False,
    }
    if evaluation.arm != "base":
        checks["adapter"] = (
            configuration["adapter_weights_sha256"] == adapters[evaluation.condition_id]
        )
    prompts = [r["model_prompt"] for r in evaluation.rows]
    if evaluation.model == "llama3.2_1b":
        date = protocol["models"]["llama3.2_1b"]["fixed_date_string"]
        checks["fixed date"] = all(f"Today Date: {date}" in p for p in prompts)
    else:
        checks["thinking block"] = all("<think>\n\n</think>" in p for p in prompts)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EvidenceError(f"{where}: failed {', '.join(failed)}")


def load_training(runs_root: Path, commit: str) -> list[dict[str, Any]]:
    """Training manifests, checked, one row per run."""
    protocol = load_protocol()
    rows = []
    for manifest_path in sorted((runs_root / "training").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text("utf-8"))
        run_dir = manifest_path.parent
        adapter = manifest["adapter"]
        weights = run_dir / "adapter/adapters.safetensors"
        environment = manifest["environment"]
        lr = manifest["post_run_checks"]["learning_rate"]
        checks = {
            "completed": manifest["status"] == "completed"
            and manifest["return_code"] == 0,
            "protocol": manifest["protocol"]["sha256"] == protocol_identity()["sha256"],
            "commit": environment["git"]["commit"] == commit,
            "adapter hash": file_sha256(weights) == adapter["selected_weights_sha256"],
            "log hash": file_sha256(run_dir / "train.log")
            == manifest["train_log"]["sha256"],
            "learning rate": lr["status"] == "passed"
            and lr["logged_at_update_16"] == lr["expected_at_update_16"]
            and lr["logged_at_update_188"] == lr["expected_at_update_188"],
            "data": all(
                d["sha256"] == d["expected_sha256"]
                for d in manifest["dataset"].values()
            ),
            "iterations": manifest["configuration"]["iters"]
            == protocol["training"]["iters"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise EvidenceError(f"{run_dir}: failed {', '.join(failed)}")
        condition = parse_condition(manifest["run_id"])
        model = protocol["models"][condition["model"]]
        schedule = protocol["training"]["schedules"][model["schedule"]]
        validation = manifest["training_metrics"]["validation"]
        metrics = manifest["training_metrics"]
        rows.append(
            {
                "run_id": manifest["run_id"],
                **condition,
                "tier": manifest["tier"],
                "train_rows": manifest["dataset"]["train"]["rows"],
                "valid_rows": manifest["dataset"]["valid"]["rows"],
                "machine": environment["machine"]["label"],
                "os": environment["platform"],
                "chip": environment["hardware"]["chip"],
                "memory": environment["hardware"]["physical_memory"],
                "precision": model["precision"],
                "layers": model["num_layers"],
                "lora_scale": model["lora_scale"],
                "peak_lr": manifest["configuration"]["learning_rate"],
                "warmup_updates": 0
                if schedule["warmup"] == 0
                else schedule["warmup"] + 1,
                "iterations": manifest["configuration"]["iters"],
                "optimizer_updates": manifest["schedule_expectation"][
                    "optimizer_updates"
                ],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "elapsed_minutes": fixed(
                    manifest["elapsed_seconds"] / SECONDS_PER_MINUTE, 0
                ),
                "peak_mlx_memory_gb": fixed(metrics["peak_mlx_memory_gb"], 2),
                "trainable_parameters": metrics["trainable_parameters"][
                    "trainable_count"
                ],
                "trainable_millions": fixed(
                    Fraction(metrics["trainable_parameters"]["trainable_count"], 10**6),
                    2,
                ),
                "adapter_bytes": adapter["selected_weights_bytes"],
                "adapter_mb": fixed(
                    Fraction(adapter["selected_weights_bytes"], BYTES_PER_MB), 1
                ),
                "adapter_sha256": adapter["selected_weights_sha256"],
                "validation_initial": validation[0]["loss"],
                "validation_final": validation[-1]["loss"],
                "validation_final_iteration": validation[-1]["iteration"],
                "validation_best": metrics["best_validation"]["loss"],
                "validation_best_iteration": metrics["best_validation"]["iteration"],
                "lr_update_16": lr["logged_at_update_16"],
                "lr_update_188": lr["logged_at_update_188"],
                "started_at": manifest["started_at"],
                "completed_at": manifest["completed_at"],
            }
        )
    return rows


# --------------------------------------------------------------------------
# Tables as rows.


def accuracy_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    """One row per evaluation and rule."""
    rows = []
    for ev in evaluations:
        items = len(ev.rows)
        for rule, flags in ev.flags.items():
            correct = sum(c for _, c in flags)
            valid = sum(v for v, _ in flags)
            low, high = stats.wilson(correct, items)
            rows.append(
                {
                    "condition_id": ev.condition_id,
                    "model": ev.model,
                    "arm": ev.arm,
                    "data": ev.data,
                    "benchmark": ev.benchmark,
                    "prompt": ev.prompt,
                    "mode": ev.mode,
                    "machine": ev.machine,
                    "rule": rule,
                    "items": items,
                    "correct": correct,
                    "valid_wrong": valid - correct,
                    "invalid": items - valid,
                    "accuracy_pct": float(percent(correct, items)),
                    "wilson_low_pct": 100 * low,
                    "wilson_high_pct": 100 * high,
                    "accuracy": fixed(percent(correct, items)),
                    "wilson_low": fixed(100 * low),
                    "wilson_high": fixed(100 * high),
                }
            )
    return rows


def diagnostics_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    """One row per evaluation."""
    rows = []
    for ev in evaluations:
        sampled = responses(ev)
        total = len(sampled)
        counts = {k: sum(s["diagnostics"][k] for s in sampled) for k in DIAGNOSTIC_KEYS}
        finish = Counter(s["finish_reason"] for s in sampled)
        tokens = [s["response_tokens"] for s in sampled]
        rows.append(
            {
                "condition_id": ev.condition_id,
                "model": ev.model,
                "arm": ev.arm,
                "data": ev.data,
                "benchmark": ev.benchmark,
                "prompt": ev.prompt,
                "mode": ev.mode,
                "machine": ev.machine,
                "items": len(ev.rows),
                "responses": total,
                **counts,
                "invalid_primary_responses": sum(
                    s["extracted"][PRIMARY] is None for s in sampled
                ),
                "question_lines_pct": fixed(percent(counts["question_lines"], total)),
                "finish_stop": finish.get("stop", 0),
                "finish_length": finish.get("length", 0),
                "mean_response_tokens": fixed(Fraction(sum(tokens), total), 0),
                "max_response_tokens": max(tokens),
                "tie_breaks_generated": sum(
                    r["tie_break"] is not None for r in ev.rows
                ),
            }
        )
    return rows


def paired_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    """Every Socratic-minus-non-Socratic comparison, each rule; Holm on the primary family."""
    by_key: dict[tuple[str, ...], dict[str, Evaluation]] = {}
    for ev in evaluations:
        if ev.arm != "base":
            by_key.setdefault(ev.key, {})[ev.arm] = ev
    family = load_protocol()["statistics"]["primary_family"]
    rows = []
    for key, arms in by_key.items():
        if set(arms) != {"socratic", "non_socratic"}:
            raise EvidenceError(f"unmatched arm for {key}")
        left, right = arms["socratic"], arms["non_socratic"]
        if [r["example_id"] for r in left.rows] != [
            r["example_id"] for r in right.rows
        ]:
            raise EvidenceError(f"items are not aligned for {key}")
        model, data, benchmark, prompt, mode = key
        in_family = (
            data == "full"
            and mode == family["decoding"]
            and prompt == family["prompt"]
            and model in family["models"]
            and benchmark in family["benchmarks"]
        )
        for rule in scoring.RULES:
            a, b, c, d = stats.paired_table(
                [x for _, x in left.flags[rule]], [x for _, x in right.flags[rule]]
            )
            n = a + b + c + d
            low, high = stats.newcombe_paired(a, b, c, d)
            p = stats.mcnemar_exact(b, c)
            if in_family:
                kind = "primary" if rule == PRIMARY else "sensitivity"
            else:
                kind = "secondary"
            rows.append(
                {
                    "family": kind,
                    "model": model,
                    "data": data,
                    "benchmark": benchmark,
                    "prompt": prompt,
                    "mode": mode,
                    "rule": rule,
                    "machine": left.machine,
                    "n": n,
                    "socratic_correct": a + b,
                    "non_socratic_correct": a + c,
                    "both_correct": a,
                    "socratic_only": b,
                    "non_socratic_only": c,
                    "both_wrong": d,
                    "difference_pp": 100 * (b - c) / n,
                    "newcombe_low_pp": 100 * low,
                    "newcombe_high_pp": 100 * high,
                    "mcnemar_exact_p": p,
                    "holm_p": None,
                    "socratic": fixed(percent(a + b, n)),
                    "non_socratic": fixed(percent(a + c, n)),
                    "difference": signed(percent(b - c, n)),
                    "ci_low": signed(100 * low),
                    "ci_high": signed(100 * high),
                    "p": p_text(p),
                    "holm": "",
                }
            )
    primary = [r for r in rows if r["family"] == "primary"]
    if len(primary) != family["tests"]:
        raise EvidenceError(
            f"expected {family['tests']} primary tests, found {len(primary)}"
        )
    for row, adjusted in zip(
        primary, stats.holm([r["mcnemar_exact_p"] for r in primary]), strict=True
    ):
        row["holm_p"], row["holm"] = adjusted, p_text(adjusted)
    return rows


def check_primary_against_stats(
    paired: list[dict[str, Any]], runs_root: Path, peak: str
) -> None:
    """Check that the primary family equals src.revision_v2.stats on the stored flags."""
    peaks = {"qwen3_0.6b": peak, "qwen3_1.7b": "1e-4", "llama3.2_1b": peak}
    reference = stats.primary_family(runs_root / "evaluation", peaks)
    ours = {(r["model"], r["benchmark"]): r for r in paired if r["family"] == "primary"}
    for row in reference:
        mine = ours[(row["model"], row["benchmark"])]
        same = (
            mine["socratic_only"] == row["socratic_only"]
            and mine["non_socratic_only"] == row["non_socratic_only"]
            and mine["n"] == row["n"]
            and math.isclose(mine["mcnemar_exact_p"], row["mcnemar_exact_p"])
            and math.isclose(mine["holm_p"], row["holm_p"])
        )
        if not same:
            raise EvidenceError(
                f"primary family differs from stats for {row['model']} {row['benchmark']}"
            )


def length_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    """Compare generated tokens of each Socratic arm with its non-Socratic match."""
    by_key: dict[tuple[str, ...], dict[str, Evaluation]] = {}
    for ev in evaluations:
        if ev.arm != "base":
            by_key.setdefault(ev.key, {})[ev.arm] = ev
    rows = []
    for (model, data, benchmark, prompt, mode), arms in sorted(by_key.items()):
        tokens = {
            arm: sum(s["response_tokens"] for s in responses(ev))
            for arm, ev in arms.items()
        }
        shorter = 1 - Fraction(tokens["non_socratic"], tokens["socratic"])
        rows.append(
            {
                "model": model,
                "data": data,
                "benchmark": benchmark,
                "prompt": prompt,
                "mode": mode,
                "socratic_tokens": tokens["socratic"],
                "non_socratic_tokens": tokens["non_socratic"],
                "non_socratic_shorter_pct": float(100 * shorter),
                "non_socratic_shorter": fixed(100 * shorter, 0),
            }
        )
    return rows


def timing_rows(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    """Wall time per evaluation."""
    from datetime import datetime

    rows = []
    for ev in evaluations:
        manifest = ev.manifest
        started = datetime.fromisoformat(manifest["started_at"])
        completed = datetime.fromisoformat(manifest["completed_at"])
        rows.append(
            {
                "condition_id": ev.condition_id,
                "benchmark": ev.benchmark,
                "prompt": ev.prompt,
                "mode": ev.mode,
                "machine": ev.machine,
                "items": len(ev.rows),
                "responses": len(responses(ev)),
                "elapsed_seconds": manifest["elapsed_seconds_this_invocation"],
                "wall_seconds": (completed - started).total_seconds(),
                "elapsed_minutes": fixed(
                    manifest["elapsed_seconds_this_invocation"] / SECONDS_PER_MINUTE, 0
                ),
            }
        )
    return rows


def dataset_record(training: list[dict[str, Any]]) -> dict[str, Any]:
    """Guiding-question density of the full training set, from the files the runs used."""
    by_arm = {
        r["arm"]: r
        for r in training
        if r["data"] == "full" and r["model"] == "qwen3_0.6b"
    }
    rows = {}
    for arm in ("socratic", "non_socratic"):
        manifest = json.loads(
            (
                REPOSITORY_ROOT
                / "runs/revision_v2/training"
                / by_arm[arm]["run_id"]
                / "manifest.json"
            ).read_text("utf-8")
        )
        train = manifest["dataset"]["train"]
        path = REPOSITORY_ROOT / train["path"]
        if file_sha256(path) != train["sha256"]:
            raise EvidenceError(f"{path} is not the training file the runs used")
        rows[arm] = list(read_jsonl(path))
    socratic, plain = rows["socratic"], rows["non_socratic"]
    if [r["example_id"] for r in socratic] != [r["example_id"] for r in plain]:
        raise EvidenceError("training files are not aligned")

    def questions(text: str) -> int:
        return sum(line.strip().endswith("?") for line in text.splitlines())

    # Guiding questions are the question lines a Socratic solution has beyond its
    # match. A few reasoning lines also end in "?"; they are shared and cancel out.
    counts = [
        questions(s["answer"]) - questions(n["answer"])
        for s, n in zip(socratic, plain, strict=True)
    ]
    shared = sum(questions(n["answer"]) > 0 for n in plain)
    socratic_chars = sum(len(r["answer"]) for r in socratic)
    plain_chars = sum(len(r["answer"]) for r in plain)
    question_share = Fraction(socratic_chars - plain_chars, socratic_chars)
    return {
        "training_rows": len(socratic),
        "questions_per_socratic_solution": {
            "mean": float(Fraction(sum(counts), len(counts))),
            "mean_display": fixed(Fraction(sum(counts), len(counts))),
            "min": min(counts),
            "max": max(counts),
            "distribution": dict(sorted(Counter(counts).items())),
        },
        "solution_characters": {
            "socratic": socratic_chars,
            "non_socratic": plain_chars,
        },
        "non_socratic_solutions_with_a_question_line": shared,
        "question_share_of_socratic_characters_pct": float(100 * question_share),
        "question_share_display": fixed(100 * question_share, 0),
        "definition": "guiding questions per solution are the lines ending in '?' that a "
        "Socratic solution has beyond its non-Socratic match; the share counts the "
        "characters the Socratic solutions have beyond their non-Socratic match",
    }


def determinism_record(runs_root: Path) -> dict[str, Any]:
    """Cross-machine comparison of the determinism runs, plus each machine's v1 check."""
    from src.revision_v2.queue import determinism

    check = load_protocol()["matrix"]["determinism_check"]
    relative = (
        Path(check["condition"])
        / check["benchmark"]
        / f"{check['prompt']}_{check['mode']}"
    )
    machines = sorted(
        p.name for p in (runs_root / "determinism").iterdir() if p.is_dir()
    )
    paths = {
        m: runs_root / "determinism" / m / relative / "predictions.jsonl"
        for m in machines
    }
    record: dict[str, Any] = {"condition": check, "machines": machines}
    record["across_machines"] = determinism(paths[machines[0]], paths[machines[1]])
    record["v1"] = {
        m: json.loads(
            (runs_root / "determinism" / m / "v1_comparison.json").read_text("utf-8")
        )
        for m in machines
    }
    same_machine = runs_root / "evaluation" / relative / "predictions.jsonl"
    if same_machine.exists():
        record["determinism_run_vs_core_run"] = determinism(paths["isik"], same_machine)
    for name, comparison in [
        ("across_machines", record["across_machines"]),
        *[(f"v1 on {m}", v) for m, v in record["v1"].items()],
    ]:
        if not comparison["identical"]:
            raise EvidenceError(f"determinism check {name} is not identical")
    return record


# --------------------------------------------------------------------------
# LaTeX tables. Each file is a complete tabular; the manuscript supplies the
# float, caption and label.


def model_blocks(rows: list[dict[str, Any]], render: Any) -> list[str]:
    """Render rows grouped by model with a rule between models."""
    lines: list[str] = []
    for index, model in enumerate(MODELS):
        group = [r for r in rows if r["model"] == model]
        if not group:
            continue
        if index and lines:
            lines.append(r"\midrule")
        lines.extend(render(row, position) for position, row in enumerate(group))
    return lines


def tabular(spec: str, header: list[str], body: list[str]) -> str:
    """Assemble a booktabs tabular."""
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            *header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def lookup(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    """Return the single row matching every field."""
    found = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(found) != 1:
        raise EvidenceError(f"expected one row for {where}, found {len(found)}")
    return found[0]


def condition_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model, then base / Socratic / non-Socratic."""
    arms = list(ARMS)
    return sorted(
        rows, key=lambda r: (list(MODELS).index(r["model"]), arms.index(r["arm"]))
    )


def conditions(
    accuracy: list[dict[str, Any]], prompt: str, mode: str
) -> list[dict[str, Any]]:
    """Distinct full-data conditions evaluated with this prompt and mode."""
    seen = {}
    for r in accuracy:
        if (
            r["prompt"] == prompt
            and r["mode"] == mode
            and r["data"] in ("full", "none")
        ):
            seen[r["condition_id"]] = {
                "model": r["model"],
                "arm": r["arm"],
                "condition_id": r["condition_id"],
            }
    return condition_order(list(seen.values()))


def label(row: dict[str, Any], position: int) -> str:
    """Model name on the first row of a block, then the condition."""
    return (MODELS[row["model"]] if position == 0 else "") + " & " + ARMS[row["arm"]]


def table_primary(paired: list[dict[str, Any]]) -> str:
    """Render the main table of the 12 pre-registered matched comparisons."""
    rows = [r for r in paired if r["family"] == "primary"]
    benchmarks = list(BENCHMARKS)
    rows.sort(
        key=lambda r: (list(MODELS).index(r["model"]), benchmarks.index(r["benchmark"]))
    )

    def render(r: dict[str, Any], position: int) -> str:
        name = MODELS[r["model"]] if position == 0 else ""
        return (
            f"{name} & {BENCHMARKS[r['benchmark']]} & {r['socratic']} & {r['non_socratic']} & "
            f"{tex_number(r['difference'])} & {tex_interval(r['ci_low'], r['ci_high'])} & "
            f"{r['p']} & {r['holm']} \\\\"
        )

    header = [
        r"\textbf{Model} & \textbf{Benchmark} & \textbf{Socratic} & \textbf{Non-Socratic} & "
        r"\textbf{Difference} & \textbf{95\% CI} & \textbf{$p$} & \textbf{Holm $p$} \\"
    ]
    return tabular("llrrrcrr", header, model_blocks(rows, render))


def table_accuracy(
    accuracy: list[dict[str, Any]], prompt: str, mode: str, *, intervals: bool
) -> str:
    """Accuracy (primary rule) by condition and benchmark, optionally with Wilson intervals."""
    benchmarks = [
        b
        for b in BENCHMARKS
        if any(
            r["benchmark"] == b and r["prompt"] == prompt and r["mode"] == mode
            for r in accuracy
        )
    ]

    def render(c: dict[str, Any], position: int) -> str:
        cells = []
        for b in benchmarks:
            r = lookup(
                accuracy,
                condition_id=c["condition_id"],
                benchmark=b,
                prompt=prompt,
                mode=mode,
                rule=PRIMARY,
            )
            cells.append(
                r["accuracy"]
                + (
                    f" {tex_interval(r['wilson_low'], r['wilson_high'])}"
                    if intervals
                    else ""
                )
            )
        return label(c, position) + " & " + " & ".join(cells) + r" \\"

    header = [
        r"\textbf{Model} & \textbf{Condition} & "
        + " & ".join(rf"\textbf{{{BENCHMARKS[b]}}}" for b in benchmarks)
        + r" \\"
    ]
    spec = "ll" + ("c" if intervals else "r") * len(benchmarks)
    return tabular(
        spec, header, model_blocks(conditions(accuracy, prompt, mode), render)
    )


def table_format(accuracy: list[dict[str, Any]], prompt: str, mode: str) -> str:
    """Invalid answers (primary rule) and fallback-rule accuracy by condition and benchmark."""
    benchmarks = [
        b
        for b in BENCHMARKS
        if any(
            r["benchmark"] == b and r["prompt"] == prompt and r["mode"] == mode
            for r in accuracy
        )
    ]

    def render(c: dict[str, Any], position: int) -> str:
        cells = []
        for b in benchmarks:
            where = {
                "condition_id": c["condition_id"],
                "benchmark": b,
                "prompt": prompt,
                "mode": mode,
            }
            cells.append(str(lookup(accuracy, rule=PRIMARY, **where)["invalid"]))
            cells.append(lookup(accuracy, rule=FALLBACK, **where)["accuracy"])
        return label(c, position) + " & " + " & ".join(cells) + r" \\"

    header = [
        " & & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{\textbf{{{BENCHMARKS[b]}}}}}" for b in benchmarks
        )
        + r" \\",
        "".join(
            rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(benchmarks))
        ),
        r"\textbf{Model} & \textbf{Condition} & "
        + " & ".join(r"\textbf{Invalid} & \textbf{Fallback}" for _ in benchmarks)
        + r" \\",
    ]
    return tabular(
        "ll" + "rr" * len(benchmarks),
        header,
        model_blocks(conditions(accuracy, prompt, mode), render),
    )


def table_resources(training: list[dict[str, Any]]) -> str:
    """Training resources for every run, with the machine that produced it."""
    data_order = ["full", "10k", "5k"]
    rows = sorted(
        training,
        key=lambda r: (
            data_order.index(r["data"]),
            list(MODELS).index(r["model"]),
            list(ARMS).index(r["arm"]),
        ),
    )

    def render(r: dict[str, Any], _position: int) -> str:
        return (
            f"{MODELS[r['model']]} & {ARMS[r['arm']]} & {DATA_SIZES[r['data']]} & {MACHINE_LABELS[r['machine']]} & "
            f"{r['precision']} & {r['elapsed_minutes']} & {r['peak_mlx_memory_gb']} & "
            f"{r['trainable_millions']} & {r['adapter_mb']} \\\\"
        )

    body = []
    for size in data_order:
        group = [r for r in rows if r["data"] == size]
        if body and group:
            body.append(r"\midrule")
        body.extend(render(r, i) for i, r in enumerate(group))
    header = [
        r"\textbf{Model} & \textbf{Arm} & \textbf{Rows} & \textbf{Machine} & \textbf{Precision} & "
        r"\textbf{Time (min)} & \textbf{Peak (GB)} & \textbf{Trainable (M)} & \textbf{LoRA (MB)} \\"
    ]
    return tabular("llrllrrrr", header, body)


def table_configurations(training: list[dict[str, Any]]) -> str:
    """Per-model settings; both arms and every data size share them."""
    rows = []
    for model, name in MODELS.items():
        group = [r for r in training if r["model"] == model]
        keys = (
            "precision",
            "layers",
            "lora_scale",
            "peak_lr",
            "warmup_updates",
            "iterations",
            "optimizer_updates",
        )
        values = {k: {r[k] for r in group} for k in keys}
        if any(len(v) != 1 for v in values.values()):
            raise EvidenceError(f"{model} runs do not share one configuration")
        v = {k: next(iter(s)) for k, s in values.items()}
        mantissa, exponent = f"{v['peak_lr']:.0e}".split("e")
        peak = (
            rf"$10^{{{int(exponent)}}}$"
            if mantissa == "1"
            else rf"${mantissa}\times10^{{{int(exponent)}}}$"
        )
        rows.append(
            f"{name} & {v['precision']} & {v['layers']} & {v['lora_scale']:g} & {peak} & "
            f"{v['warmup_updates']} & {v['iterations']:,} & {v['optimizer_updates']} & {len(group)} \\\\"
        )
    header = [
        r"\textbf{Model} & \textbf{Precision} & \textbf{Layers} & \textbf{LoRA scale} & \textbf{Peak LR} & "
        r"\textbf{Warm-up} & \textbf{Iterations} & \textbf{Updates} & \textbf{Runs} \\"
    ]
    return tabular("llrrrrrrr", header, rows)


def table_second_tier(accuracy: list[dict[str, Any]]) -> str:
    """Qwen3-0.6B greedy P0 accuracy by training-set size and arm."""
    rows = []
    for size in ("5k", "10k", "full"):
        for arm in ("socratic", "non_socratic"):
            cells = [
                lookup(
                    accuracy,
                    model="qwen3_0.6b",
                    arm=arm,
                    data=size,
                    benchmark=b,
                    prompt="P0",
                    mode="greedy",
                    rule=PRIMARY,
                )["accuracy"]
                for b in BENCHMARKS
            ]
            rows.append(
                f"{DATA_SIZES[size]} & {ARMS[arm]} & " + " & ".join(cells) + r" \\"
            )
    header = [
        r"\textbf{Training rows} & \textbf{Arm} & "
        + " & ".join(rf"\textbf{{{v}}}" for v in BENCHMARKS.values())
        + r" \\"
    ]
    return tabular("llrrrr", header, rows)


def table_secondary_paired(paired: list[dict[str, Any]]) -> str:
    """Secondary matched comparisons (primary rule): SC@5, P4 and the data-size runs."""
    rows = [r for r in paired if r["family"] == "secondary" and r["rule"] == PRIMARY]
    order = {
        ("full", "P0", "sc5"): 0,
        ("full", "P4", "greedy"): 1,
        ("10k", "P0", "greedy"): 2,
        ("5k", "P0", "greedy"): 3,
    }
    names = {
        0: "SC@5, zero-shot",
        1: "Greedy, four-shot",
        2: "Greedy, 10,000 rows",
        3: "Greedy, 5,000 rows",
    }
    rows.sort(
        key=lambda r: (
            order[(r["data"], r["prompt"], r["mode"])],
            list(MODELS).index(r["model"]),
            list(BENCHMARKS).index(r["benchmark"]),
        )
    )
    body, previous = [], None
    for r in rows:
        group = order[(r["data"], r["prompt"], r["mode"])]
        if previous is not None and group != previous:
            body.append(r"\midrule")
        body.append(
            f"{names[group] if group != previous else ''} & {MODELS[r['model']]} & {BENCHMARKS[r['benchmark']]} & "
            f"{r['socratic']} & {r['non_socratic']} & {tex_number(r['difference'])} & "
            f"{tex_interval(r['ci_low'], r['ci_high'])} & {r['p']} \\\\"
        )
        previous = group
    header = [
        r"\textbf{Setting} & \textbf{Model} & \textbf{Benchmark} & \textbf{Socratic} & \textbf{Non-Socratic} & "
        r"\textbf{Difference} & \textbf{95\% CI} & \textbf{$p$} \\"
    ]
    return tabular("lllrrrcr", header, body)


def table_sensitivity(paired: list[dict[str, Any]]) -> str:
    """Render the 12 primary differences under each of the three rules."""
    rules = list(scoring.RULES)
    keys = sorted(
        {(r["model"], r["benchmark"]) for r in paired if r["family"] == "primary"},
        key=lambda k: (list(MODELS).index(k[0]), list(BENCHMARKS).index(k[1])),
    )
    body, previous = [], None
    for model, benchmark in keys:
        if previous and previous != model:
            body.append(r"\midrule")
        cells = []
        for rule in rules:
            r = lookup(
                paired,
                model=model,
                data="full",
                benchmark=benchmark,
                prompt="P0",
                mode="greedy",
                rule=rule,
            )
            cells.append(f"{tex_number(r['difference'])} ({r['p']})")
        body.append(
            f"{MODELS[model] if previous != model else ''} & {BENCHMARKS[benchmark]} & "
            + " & ".join(cells)
            + r" \\"
        )
        previous = model
    header = [
        r"\textbf{Model} & \textbf{Benchmark} & "
        + " & ".join(rf"\textbf{{{RULE_SHORT[r]}}}" for r in rules)
        + r" \\"
    ]
    return tabular("llccc", header, body)


def table_outcomes(accuracy: list[dict[str, Any]]) -> str:
    """Greedy P0 outcomes under the primary rule: correct / valid but wrong / invalid."""

    def render(c: dict[str, Any], position: int) -> str:
        cells = []
        for b in BENCHMARKS:
            r = lookup(
                accuracy,
                condition_id=c["condition_id"],
                benchmark=b,
                prompt="P0",
                mode="greedy",
                rule=PRIMARY,
            )
            cells.append(f"{r['correct']:,}/{r['valid_wrong']:,}/{r['invalid']:,}")
        return label(c, position) + " & " + " & ".join(cells) + r" \\"

    header = [
        r"\textbf{Model} & \textbf{Condition} & "
        + " & ".join(rf"\textbf{{{v}}}" for v in BENCHMARKS.values())
        + r" \\"
    ]
    return tabular(
        "llrrrr", header, model_blocks(conditions(accuracy, "P0", "greedy"), render)
    )


def table_diagnostics(diagnostics: list[dict[str, Any]], prompt: str, mode: str) -> str:
    """Per-response format diagnostics by condition and benchmark."""
    chosen = [
        d
        for d in diagnostics
        if d["prompt"] == prompt and d["mode"] == mode and d["data"] in ("full", "none")
    ]
    body, previous = [], None
    for c in conditions([{**d, "rule": None} for d in chosen], prompt, mode):
        for b, benchmark_name in BENCHMARKS.items():
            match = [
                d
                for d in chosen
                if d["condition_id"] == c["condition_id"] and d["benchmark"] == b
            ]
            if not match:
                continue
            d = match[0]
            if previous is not None and previous != c["model"]:
                body.append(r"\midrule")
            previous = c["model"]
            body.append(
                f"{MODELS[c['model']]} {ARMS[c['arm']]} & {benchmark_name} & {d['responses']:,} & "
                f"{d['invalid_primary_responses']} & {d['bare_answer']} & {d['limit_hit']} & "
                f"{d['repetition_loop']} & {d['question_lines_pct']} & {d['mean_response_tokens']} \\\\"
            )
    header = [
        r"\textbf{Condition} & \textbf{Benchmark} & \textbf{$n$} & \textbf{Invalid} & \textbf{Bare} & "
        r"\textbf{Limit} & \textbf{Loop} & \textbf{Questions (\%)} & \textbf{Tokens} \\"
    ]
    return tabular("llrrrrrrr", header, body)


def table_p4(accuracy: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> str:
    """Render four-shot GSM8K accuracy beside its format diagnostics."""

    def render(c: dict[str, Any], position: int) -> str:
        where = {
            "condition_id": c["condition_id"],
            "benchmark": "gsm8k",
            "prompt": "P4",
            "mode": "greedy",
        }
        r = lookup(accuracy, rule=PRIMARY, **where)
        d = lookup(diagnostics, **where)
        return (
            f"{label(c, position)} & {r['accuracy']} {tex_interval(r['wilson_low'], r['wilson_high'])} & "
            f"{r['invalid']} & {d['bare_answer']} & {d['limit_hit']} & {d['repetition_loop']} \\\\"
        )

    header = [
        r"\textbf{Model} & \textbf{Condition} & \textbf{Accuracy (\%)} & \textbf{Invalid} & \textbf{Bare} & "
        r"\textbf{Limit} & \textbf{Loop} \\"
    ]
    return tabular(
        "llcrrrr", header, model_blocks(conditions(accuracy, "P4", "greedy"), render)
    )


# --------------------------------------------------------------------------
# Build.


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with a stable column order."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), "utf-8")


def build(runs_root: Path, output: Path) -> dict[str, Any]:
    """Check the returned runs and write every evidence file."""
    commit = tag_commit()
    code = frozen_code_hashes()
    decision = json.loads((runs_root / "pilot_decision.json").read_text("utf-8"))
    training = load_training(runs_root, commit)
    adapters = {r["run_id"]: r["adapter_sha256"] for r in training}
    machines_by_run = {r["run_id"]: r["machine"] for r in training}

    evaluations = []
    for run_dir in sorted((runs_root / "evaluation").glob("*/*/*")):
        evaluation = load_evaluation(run_dir)
        check_summary(evaluation)
        check_provenance(evaluation, commit, code, adapters)
        if (
            evaluation.arm != "base"
            and evaluation.machine != machines_by_run[evaluation.condition_id]
        ):
            raise EvidenceError(
                f"{run_dir}: evaluated on a different machine from training"
            )
        evaluations.append(evaluation)

    accuracy = accuracy_rows(evaluations)
    diagnostics = diagnostics_rows(evaluations)
    paired = paired_rows(evaluations)
    check_primary_against_stats(paired, runs_root, decision["peak"])
    determinism_result = determinism_record(runs_root)

    tables = {
        "primary_family": table_primary(paired),
        "accuracy_p0_greedy": table_accuracy(accuracy, "P0", "greedy", intervals=True),
        "accuracy_p0_greedy_plain": table_accuracy(
            accuracy, "P0", "greedy", intervals=False
        ),
        "format_p0_greedy": table_format(accuracy, "P0", "greedy"),
        "resources": table_resources(training),
        "configurations": table_configurations(training),
        "accuracy_p0_sc5": table_accuracy(accuracy, "P0", "sc5", intervals=True),
        "accuracy_p4_greedy": table_accuracy(accuracy, "P4", "greedy", intervals=True),
        "format_p4_greedy": table_format(accuracy, "P4", "greedy"),
        "second_tier": table_second_tier(accuracy),
        "secondary_paired": table_secondary_paired(paired),
        "sensitivity": table_sensitivity(paired),
        "outcomes_p0_greedy": table_outcomes(accuracy),
        "diagnostics_p0_greedy": table_diagnostics(diagnostics, "P0", "greedy"),
        "diagnostics_p4_greedy": table_diagnostics(diagnostics, "P4", "greedy"),
        "diagnostics_p0_sc5": table_diagnostics(diagnostics, "P0", "sc5"),
        "p4_summary": table_p4(accuracy, diagnostics),
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    write_csv(output / "accuracy.csv", accuracy)
    write_csv(output / "diagnostics.csv", diagnostics)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "training.csv", training)
    write_csv(output / "evaluation_timing.csv", timing_rows(evaluations))
    write_csv(output / "response_length.csv", length_rows(evaluations))
    write_json(output / "determinism.json", determinism_result)
    write_json(output / "dataset.json", dataset_record(training))
    for name, text in tables.items():
        (output / "tables" / f"{name}.tex").write_text(text, "utf-8")

    inputs = {}
    for path in sorted(runs_root.rglob("*")):
        if path.is_file() and path.suffix in (
            ".json",
            ".jsonl",
            ".log",
            ".yaml",
            ".safetensors",
        ):
            inputs[str(path.relative_to(REPOSITORY_ROOT))] = file_sha256(path)
    packets = (
        {p.name: file_sha256(p) for p in sorted(PACKET_DIRECTORY.glob("*.tar.gz"))}
        if PACKET_DIRECTORY.exists()
        else {}
    )
    outputs = {
        str(p.relative_to(output)): file_sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    manifest = {
        "created_at": utc_now(),
        "generator": "src.revision_v2.tables",
        "generator_sha256": file_sha256(Path(__file__)),
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_tracked_changes": git(
            "status", "--porcelain", "--untracked-files=no"
        ),
        "freeze_tag": FREEZE_TAG,
        "freeze_commit": commit,
        "frozen_code_sha256": code,
        "protocol": protocol_identity(),
        "pilot_decision": decision,
        "counts": {
            "training_runs": len(training),
            "evaluations": len(evaluations),
            "items_rescored": sum(len(e.rows) for e in evaluations),
            "responses_rescored": sum(len(responses(e)) for e in evaluations),
            "paired_comparisons": len(paired),
        },
        "checks_passed": [
            "every response re-scored from raw text equals its stored extraction, diagnostics and decision",
            "every evaluation's counts equal its manifest summary",
            "prediction, adapter and train.log hashes equal their manifests",
            "every run used the frozen protocol, the freeze-tag commit, and the frozen evaluator and scorer",
            "every Llama prompt carries the fixed date; every Qwen prompt has thinking off",
            "every adapter evaluated is the adapter trained, on the same machine",
            "logged learning rates at updates 16 and 188 equal the schedule replay",
            "matched arms are evaluated on identical items",
            "the primary family equals src.revision_v2.stats",
            "determinism runs are identical across machines and to each machine's v1 run",
        ],
        "return_packets_sha256": packets,
        "inputs_sha256": inputs,
        "outputs_sha256": outputs,
    }
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY_ROOT / RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the evidence directory."""
    args = _parser().parse_args(argv)
    manifest = build(args.runs_root.resolve(), args.output.resolve())
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote evidence to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

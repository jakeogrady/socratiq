"""Rescore immutable reviewer-rerun responses with the submitted-paper rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluate_v2 import (
    answers_equal,
    extract_marked_number,
    extract_terminal_marked_number,
    majority_vote,
    normalize_reference_answer,
)
from src.rerun_utils import (
    file_sha256,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)
from src.summarize_results import wilson_interval

SCORER_ID = "submitted-last-marked-decimal-v1"
SCORER_RULE = (
    "Select the last ####-marked integer or decimal anywhere in the response; "
    "ignore trailing text; do not fall back to unmarked numbers."
)
TERMINAL_DIAGNOSTIC_ID = "terminal-marked-decimal-diagnostic-v1"
PAIRED_COMPARISONS = (
    ("qwen3-0.6b-socratic-vs-base", "qwen3-0.6b-socratic", "qwen3-0.6b-base"),
    (
        "qwen3-0.6b-non-socratic-vs-base",
        "qwen3-0.6b-non-socratic",
        "qwen3-0.6b-base",
    ),
    (
        "qwen3-0.6b-socratic-vs-non-socratic",
        "qwen3-0.6b-socratic",
        "qwen3-0.6b-non-socratic",
    ),
    ("qwen3-1.7b-socratic-vs-base", "qwen3-1.7b-socratic", "qwen3-1.7b-base"),
    (
        "qwen3-1.7b-non-socratic-vs-base",
        "qwen3-1.7b-non-socratic",
        "qwen3-1.7b-base",
    ),
    (
        "qwen3-1.7b-socratic-vs-non-socratic",
        "qwen3-1.7b-socratic",
        "qwen3-1.7b-non-socratic",
    ),
    ("llama3.2-1b-socratic-vs-base", "llama3.2-1b-socratic", "llama3.2-1b-base"),
)


class RescoringError(RuntimeError):
    """Raised when source evidence is incomplete or unsafe to rescore."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RescoringError(f"expected a JSON object in {path}")
    return value


def _prediction_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    """Resolve a prediction file, preferring the copy beside its manifest."""
    configured = manifest.get("prediction_path")
    if not isinstance(configured, str) or not configured:
        raise RescoringError(f"prediction_path is missing in {manifest_path}")
    adjacent = manifest_path.parent / Path(configured).name
    if adjacent.is_file():
        return adjacent
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_file():
        return candidate
    raise RescoringError(f"prediction file does not exist for {manifest_path}")


def _extract_answers(
    samples: Any,
    *,
    extractor: Any,
    context: str,
) -> list[str | None]:
    if not isinstance(samples, list) or not samples:
        raise RescoringError(f"{context}: samples must be a non-empty list")
    answers: list[str | None] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise RescoringError(f"{context}: sample {sample_index} is not an object")
        response = sample.get("response")
        if not isinstance(response, str):
            raise RescoringError(
                f"{context}: sample {sample_index} has no raw response"
            )
        answers.append(extractor(response))
    return answers


def _greedy_tie_answer(
    row: Mapping[str, Any],
    *,
    extractor: Any,
    required: bool,
    context: str,
) -> str | None:
    tie_break = row.get("greedy_tie_break")
    if not isinstance(tie_break, Mapping):
        if required:
            raise RescoringError(
                f"{context}: corrected self-consistency vote requires a raw "
                "greedy tie-break response that was not recorded"
            )
        return None
    response = tie_break.get("response")
    if not isinstance(response, str):
        raise RescoringError(f"{context}: tie-break has no raw response")
    return extractor(response)


def rescore_prediction_row(
    row: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Recompute one decision from raw responses without model inference."""
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise RescoringError("prediction row has no example_id")
    context = example_id
    primary_answers = _extract_answers(
        row.get("samples"), extractor=extract_marked_number, context=context
    )
    terminal_answers = _extract_answers(
        row.get("samples"), extractor=extract_terminal_marked_number, context=context
    )

    primary_prediction, primary_tied = majority_vote(primary_answers)
    if mode == "self_consistency" and primary_tied:
        primary_tie_answer = _greedy_tie_answer(
            row,
            extractor=extract_marked_number,
            required=True,
            context=context,
        )
        primary_prediction, _ = majority_vote(
            primary_answers, greedy_answer=primary_tie_answer
        )

    terminal_prediction, terminal_tied = majority_vote(terminal_answers)
    if mode == "self_consistency" and terminal_tied:
        terminal_tie_answer = _greedy_tie_answer(
            row,
            extractor=extract_terminal_marked_number,
            required=False,
            context=context,
        )
        terminal_prediction, _ = majority_vote(
            terminal_answers, greedy_answer=terminal_tie_answer
        )

    reference = normalize_reference_answer(row.get("reference_answer"))
    if reference is None:
        raise RescoringError(f"{context}: reference answer is invalid")
    corrected_valid = primary_prediction is not None
    corrected_correct = answers_equal(primary_prediction, reference)
    terminal_valid = terminal_prediction is not None
    terminal_correct = answers_equal(terminal_prediction, reference)

    return {
        "schema_version": "1.0",
        "scorer_id": SCORER_ID,
        "experiment_id": row.get("experiment_id"),
        "model": row.get("model"),
        "model_revision": row.get("model_revision"),
        "adapter": row.get("adapter"),
        "benchmark": row.get("benchmark"),
        "example_id": example_id,
        "example_index": row.get("example_index"),
        "reference_answer": reference,
        "sample_extracted_answers": primary_answers,
        "terminal_sample_extracted_answers": terminal_answers,
        "previous_predicted_answer": row.get("predicted_answer"),
        "previous_valid_answer": row.get("valid_answer") is True,
        "previous_correct": row.get("correct") is True,
        "corrected_predicted_answer": primary_prediction,
        "corrected_valid_answer": corrected_valid,
        "corrected_correct": corrected_correct,
        "terminal_predicted_answer": terminal_prediction,
        "terminal_valid_answer": terminal_valid,
        "terminal_correct": terminal_correct,
        "decision_changed": (
            row.get("predicted_answer") != primary_prediction
            or (row.get("valid_answer") is True) != corrected_valid
            or (row.get("correct") is True) != corrected_correct
        ),
        "format_only_recovery": (
            row.get("valid_answer") is not True and corrected_valid
        ),
    }


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    source_manifest_path: Path,
    source_prediction_path: Path,
    rescored_path: Path,
) -> dict[str, Any]:
    if not rows:
        raise RescoringError(f"no prediction rows in {source_prediction_path}")
    ids = [row.get("example_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise RescoringError(f"duplicate example IDs in {source_prediction_path}")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RescoringError(f"configuration is missing in {source_manifest_path}")
    benchmark_config = configuration.get("benchmark")
    if not isinstance(benchmark_config, Mapping):
        raise RescoringError(
            f"benchmark configuration is missing in {source_manifest_path}"
        )
    expected_rows = benchmark_config.get("expected_rows")
    if not isinstance(expected_rows, int) or len(rows) != expected_rows:
        raise RescoringError(
            f"expected {expected_rows} rows, found {len(rows)} in "
            f"{source_prediction_path}"
        )

    previous_correct = sum(row.get("previous_correct") is True for row in rows)
    previous_valid = sum(row.get("previous_valid_answer") is True for row in rows)
    corrected_correct = sum(row.get("corrected_correct") is True for row in rows)
    corrected_valid = sum(row.get("corrected_valid_answer") is True for row in rows)
    terminal_correct = sum(row.get("terminal_correct") is True for row in rows)
    terminal_valid = sum(row.get("terminal_valid_answer") is True for row in rows)
    changed = sum(row.get("decision_changed") is True for row in rows)
    recovered = sum(row.get("format_only_recovery") is True for row in rows)
    lower, upper = wilson_interval(corrected_correct, len(rows))

    return {
        "experiment_id": configuration.get("experiment_id"),
        "model": configuration.get("model"),
        "model_revision": configuration.get("model_revision"),
        "benchmark": benchmark_config.get("key"),
        "mode": configuration.get("mode"),
        "samples": configuration.get("samples"),
        "observed_rows": len(rows),
        "previous_valid_answers": previous_valid,
        "previous_correct_answers": previous_correct,
        "previous_accuracy": previous_correct / len(rows),
        "corrected_valid_answers": corrected_valid,
        "corrected_invalid_answers": len(rows) - corrected_valid,
        "corrected_correct_answers": corrected_correct,
        "corrected_accuracy": corrected_correct / len(rows),
        "corrected_accuracy_percent": 100 * corrected_correct / len(rows),
        "corrected_wilson_lower": lower,
        "corrected_wilson_upper": upper,
        "terminal_valid_answers": terminal_valid,
        "terminal_correct_answers": terminal_correct,
        "terminal_accuracy": terminal_correct / len(rows),
        "decision_changes": changed,
        "format_only_recoveries": recovered,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_prediction_path": str(source_prediction_path),
        "source_prediction_sha256": file_sha256(source_prediction_path),
        "rescored_path": str(rescored_path),
        "rescored_sha256": file_sha256(rescored_path),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant pairs."""
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2)
        for index in range(min(left_only, right_only) + 1)
    ]
    largest = max(log_probabilities)
    tail = math.exp(largest) * sum(
        math.exp(log_probability - largest) for log_probability in log_probabilities
    )
    return min(1.0, 2 * tail)


def paired_comparisons(
    decisions: Mapping[tuple[str, str], Mapping[str, bool]],
) -> list[dict[str, Any]]:
    """Build predeclared paired comparisons from corrected per-example decisions."""
    benchmarks = sorted({benchmark for _, benchmark in decisions})
    rows: list[dict[str, Any]] = []
    for comparison_id, left_experiment, right_experiment in PAIRED_COMPARISONS:
        for benchmark in benchmarks:
            left = decisions.get((left_experiment, benchmark))
            right = decisions.get((right_experiment, benchmark))
            if left is None or right is None:
                continue
            if set(left) != set(right):
                raise RescoringError(
                    f"paired example IDs differ for {comparison_id} on {benchmark}"
                )
            example_ids = sorted(left)
            both_correct = sum(left[item] and right[item] for item in example_ids)
            left_only = sum(left[item] and not right[item] for item in example_ids)
            right_only = sum(not left[item] and right[item] for item in example_ids)
            both_wrong = len(example_ids) - both_correct - left_only - right_only
            left_correct = both_correct + left_only
            right_correct = both_correct + right_only
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "benchmark": benchmark,
                    "left_experiment": left_experiment,
                    "right_experiment": right_experiment,
                    "observed_pairs": len(example_ids),
                    "left_correct_answers": left_correct,
                    "right_correct_answers": right_correct,
                    "left_accuracy": left_correct / len(example_ids),
                    "right_accuracy": right_correct / len(example_ids),
                    "left_minus_right_percentage_points": (
                        100 * (left_correct - right_correct) / len(example_ids)
                    ),
                    "both_correct": both_correct,
                    "left_only_correct": left_only,
                    "right_only_correct": right_only,
                    "both_wrong": both_wrong,
                    "mcnemar_exact_two_sided_p": exact_mcnemar_p(left_only, right_only),
                }
            )
    return rows


def rescore_evaluations(evaluation_root: Path, output_root: Path) -> dict[str, Any]:
    """Rescore every complete evaluation below ``evaluation_root``."""
    if output_root.exists():
        raise RescoringError(f"refusing to overwrite output root: {output_root}")
    manifest_paths = sorted(evaluation_root.rglob("manifest.json"))
    if not manifest_paths:
        raise RescoringError(f"no evaluation manifests found below {evaluation_root}")

    sources: list[tuple[Path, dict[str, Any], Path]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        configuration = manifest.get("configuration")
        if not isinstance(configuration, Mapping):
            continue
        if configuration.get("evaluator") != "evaluate_v2":
            continue
        if manifest.get("status") != "completed":
            raise RescoringError(f"evaluation is not completed: {manifest_path}")
        prediction_path = _prediction_path(manifest_path, manifest)
        actual_hash = file_sha256(prediction_path)
        expected_hash = manifest.get("prediction_sha256")
        if expected_hash != actual_hash:
            raise RescoringError(
                f"prediction hash mismatch for {prediction_path}: "
                f"manifest={expected_hash}, actual={actual_hash}"
            )
        sources.append((manifest_path, manifest, prediction_path))
    if not sources:
        raise RescoringError(
            f"no completed evaluate_v2 manifests found below {evaluation_root}"
        )

    output_root.mkdir(parents=True)
    summary_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str], dict[str, bool]] = {}
    for manifest_path, manifest, prediction_path in sources:
        configuration = manifest["configuration"]
        mode = configuration.get("mode")
        if mode not in {"greedy", "self_consistency"}:
            raise RescoringError(f"unsupported evaluation mode in {manifest_path}")
        relative_run = manifest_path.parent.relative_to(evaluation_root)
        run_output = output_root / "runs" / relative_run
        rescored_path = run_output / "rescored.jsonl"
        rescored_rows = [
            rescore_prediction_row(row, mode=mode)
            for row in read_jsonl(prediction_path)
        ]
        write_jsonl(rescored_path, rescored_rows)
        experiment_id = str(configuration.get("experiment_id"))
        benchmark_config = configuration.get("benchmark")
        if not isinstance(benchmark_config, Mapping):
            raise RescoringError(
                f"benchmark configuration is missing in {manifest_path}"
            )
        benchmark = str(benchmark_config.get("key"))
        decision_key = (experiment_id, benchmark)
        if decision_key in decisions:
            raise RescoringError(f"duplicate evaluation run for {decision_key}")
        decisions[decision_key] = {
            str(row["example_id"]): bool(row["corrected_correct"])
            for row in rescored_rows
        }
        summary = _summary(
            rescored_rows,
            manifest=manifest,
            source_manifest_path=manifest_path,
            source_prediction_path=prediction_path,
            rescored_path=rescored_path,
        )
        summary_rows.append(summary)
        run_manifest = {
            "schema_version": "1.0",
            "scorer_id": SCORER_ID,
            "scorer_rule": SCORER_RULE,
            "terminal_diagnostic_id": TERMINAL_DIAGNOSTIC_ID,
            "generated_at": utc_now(),
            **summary,
        }
        run_manifest_path = run_output / "rescore_manifest.json"
        write_json(run_manifest_path, run_manifest)
        run_records.append(
            {
                **summary,
                "rescore_manifest_path": str(run_manifest_path),
                "rescore_manifest_sha256": file_sha256(run_manifest_path),
            }
        )

    summary_rows.sort(
        key=lambda row: (str(row["experiment_id"]), str(row["benchmark"]))
    )
    summary_path = output_root / "summary.csv"
    _write_csv(summary_path, summary_rows)
    comparison_rows = paired_comparisons(decisions)
    comparison_path = output_root / "paired_comparisons.csv"
    _write_csv(comparison_path, comparison_rows)
    aggregate = {
        "schema_version": "1.0",
        "scorer_id": SCORER_ID,
        "scorer_rule": SCORER_RULE,
        "terminal_diagnostic_id": TERMINAL_DIAGNOSTIC_ID,
        "generated_at": utc_now(),
        "evaluation_root": str(evaluation_root),
        "output_root": str(output_root),
        "run_count": len(run_records),
        "total_rows": sum(int(row["observed_rows"]) for row in summary_rows),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "paired_comparisons_path": str(comparison_path),
        "paired_comparisons_sha256": file_sha256(comparison_path),
        "paired_comparison_count": len(comparison_rows),
        "scorer_code": {
            "rescore_predictions_sha256": file_sha256(Path(__file__)),
            "evaluate_v2_sha256": file_sha256(
                Path(__file__).with_name("evaluate_v2.py")
            ),
        },
        "runs": run_records,
    }
    write_json(output_root / "rescore_manifest.json", aggregate)
    return aggregate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the immutable-response rescoring command."""
    args = _build_parser().parse_args(argv)
    manifest = rescore_evaluations(args.evaluation_root, args.output_root)
    print(
        f"Rescored {manifest['total_rows']} rows across "
        f"{manifest['run_count']} evaluation runs."
    )
    print(f"Summary: {manifest['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

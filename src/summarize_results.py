"""Generate reviewer-rerun result, resource, and reproducibility summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from src.evaluate_v2 import MODEL_REVISIONS, PROMPT_VERSION, TASK_INSTRUCTION
from src.rerun_utils import (
    file_sha256,
    protocol_identity,
    read_jsonl,
    utc_now,
    write_json,
)


class ReportingError(RuntimeError):
    """Raised when raw evidence is incomplete or internally inconsistent."""


MANDATORY_TRAINING_EXPERIMENTS = frozenset(
    {
        "qwen3-0.6b-socratic",
        "qwen3-0.6b-non-socratic",
        "llama3.2-1b-socratic",
    }
)
MANDATORY_EVALUATION_EXPERIMENTS = frozenset(
    {
        "qwen3-0.6b-base",
        "qwen3-0.6b-socratic",
        "qwen3-0.6b-non-socratic",
        "llama3.2-1b-base",
        "llama3.2-1b-socratic",
    }
)
MANDATORY_BENCHMARKS = frozenset({"gsm8k", "multiarith", "svamp"})
GSMHARD_EVALUATION_EXPERIMENTS = frozenset(
    {
        "qwen3-0.6b-base",
        "qwen3-0.6b-socratic",
        "qwen3-0.6b-non-socratic",
        "qwen3-1.7b-base",
        "qwen3-1.7b-socratic",
        "qwen3-1.7b-non-socratic",
        "llama3.2-1b-base",
        "llama3.2-1b-socratic",
    }
)
GSMHARD_MODES = frozenset({"greedy", "self_consistency"})


def validate_mandatory_matrix(
    evaluation_rows: Sequence[Mapping[str, Any]],
    resource_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the three training and 15 greedy evaluation reviewer runs."""
    expected_evaluations = {
        (experiment, benchmark, "greedy")
        for experiment in MANDATORY_EVALUATION_EXPERIMENTS
        for benchmark in MANDATORY_BENCHMARKS
    }
    observed_evaluations = [
        (
            str(row.get("experiment_id")),
            str(row.get("benchmark")),
            str(row.get("mode")),
        )
        for row in evaluation_rows
    ]
    duplicate_evaluations = sorted(
        {
            identity
            for identity in observed_evaluations
            if observed_evaluations.count(identity) > 1
        }
    )
    missing_evaluations = sorted(expected_evaluations - set(observed_evaluations))

    completed_training = {
        str(row.get("experiment_id"))
        for row in resource_rows
        if row.get("status") == "completed"
    }
    missing_training = sorted(MANDATORY_TRAINING_EXPERIMENTS - completed_training)
    if missing_training or missing_evaluations or duplicate_evaluations:
        raise ReportingError(
            "mandatory reviewer matrix is incomplete or duplicated: "
            f"missing_training={missing_training}, "
            f"missing_evaluations={missing_evaluations}, "
            f"duplicate_evaluations={duplicate_evaluations}"
        )
    return {
        "status": "passed",
        "mandatory_training_runs": len(MANDATORY_TRAINING_EXPERIMENTS),
        "mandatory_evaluation_runs": len(expected_evaluations),
    }


def validate_gsmhard_matrix(
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the isolated eight-condition, two-decoding GSM-Hard matrix."""
    expected = {
        (experiment, "gsm_hard", mode)
        for experiment in GSMHARD_EVALUATION_EXPERIMENTS
        for mode in GSMHARD_MODES
    }
    observed = [
        (
            str(row.get("experiment_id")),
            str(row.get("benchmark")),
            str(row.get("mode")),
        )
        for row in evaluation_rows
    ]
    duplicates = sorted(
        {identity for identity in observed if observed.count(identity) > 1}
    )
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected or duplicates:
        raise ReportingError(
            "GSM-Hard extension matrix is incomplete, unexpected, or duplicated: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    return {
        "status": "passed",
        "benchmark": "gsm_hard",
        "conditions": len(GSMHARD_EVALUATION_EXPERIMENTS),
        "modes": len(GSMHARD_MODES),
        "evaluation_runs": len(expected),
    }


def wilson_interval(
    correct: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be greater than zero")
    if correct < 0 or correct > total:
        raise ValueError("correct must be between zero and total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = correct / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int | None,
    require_full: bool,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate and summarize one homogeneous collection of prediction rows."""
    if not rows:
        raise ReportingError("prediction file is empty")
    example_ids = [row.get("example_id") for row in rows]
    if any(not isinstance(example_id, str) for example_id in example_ids):
        raise ReportingError("one or more predictions lack an example_id")
    if len(set(example_ids)) != len(example_ids):
        raise ReportingError("prediction file contains duplicate example IDs")

    benchmarks = {row.get("benchmark") for row in rows}
    experiments = {row.get("experiment_id") for row in rows}
    models = {row.get("model") for row in rows}
    model_revisions = {row.get("model_revision") for row in rows}
    if (
        len(benchmarks) != 1
        or len(experiments) != 1
        or len(models) != 1
        or len(model_revisions) != 1
    ):
        raise ReportingError("prediction rows are not one homogeneous evaluation")
    if require_full and expected_rows is not None and len(rows) != expected_rows:
        raise ReportingError(
            f"expected {expected_rows} prediction rows, found {len(rows)}"
        )

    correct = sum(row.get("correct") is True for row in rows)
    valid = sum(row.get("valid_answer") is True for row in rows)
    generation_seconds = 0.0
    sample_count = 0
    for row in rows:
        samples = row.get("samples", [])
        if not isinstance(samples, list):
            raise ReportingError("samples must be a list in every prediction row")
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ReportingError("sample entries must be objects")
            generation_seconds += float(sample.get("generation_seconds", 0.0))
            sample_count += 1
        tie_break = row.get("greedy_tie_break")
        if isinstance(tie_break, Mapping):
            generation_seconds += float(tie_break.get("generation_seconds", 0.0))
            sample_count += 1

    lower, upper = wilson_interval(correct, len(rows), confidence=confidence)
    return {
        "experiment_id": next(iter(experiments)),
        "model": next(iter(models)),
        "model_revision": next(iter(model_revisions)),
        "benchmark": next(iter(benchmarks)),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "valid_answers": valid,
        "invalid_answers": len(rows) - valid,
        "correct_answers": correct,
        "accuracy": correct / len(rows),
        "accuracy_percent": 100 * correct / len(rows),
        "wilson_lower": lower,
        "wilson_upper": upper,
        "wilson_lower_percent": 100 * lower,
        "wilson_upper_percent": 100 * upper,
        "generation_count": sample_count,
        "generation_seconds": generation_seconds,
        "seconds_per_generation": (
            generation_seconds / sample_count if sample_count else None
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReportingError(f"expected a JSON object in {path}")
    return value


def summarize_evaluation_run(
    manifest_path: Path,
    *,
    require_full: bool = True,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize one evaluation manifest and its raw prediction file."""
    manifest = _read_json(manifest_path)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ReportingError(f"not an evaluation manifest: {manifest_path}")
    benchmark = configuration.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ReportingError("evaluation manifest lacks benchmark configuration")
    expected_rows = benchmark.get("expected_rows")
    if not isinstance(expected_rows, int):
        raise ReportingError("benchmark expected_rows is missing or invalid")
    prediction_path = Path(manifest.get("prediction_path", ""))
    if not prediction_path.is_absolute():
        repository_relative = Path.cwd() / prediction_path
        run_relative = manifest_path.parent / prediction_path.name
        prediction_path = (
            repository_relative if repository_relative.exists() else run_relative
        )
    if not prediction_path.exists():
        raise ReportingError(f"prediction file does not exist: {prediction_path}")
    rows = list(read_jsonl(prediction_path))
    summary = summarize_prediction_rows(
        rows,
        expected_rows=expected_rows,
        require_full=require_full,
        confidence=confidence,
    )
    base_model = manifest.get("base_model")
    if configuration.get("evaluator") == "evaluate_v2":
        if not isinstance(base_model, Mapping):
            raise ReportingError(
                "corrected evaluation manifest lacks base_model identity"
            )
        manifested_identity = base_model.get("resolved_revision") or base_model.get(
            "content_sha256"
        )
        if summary["model_revision"] != manifested_identity:
            raise ReportingError(
                "raw prediction model revision differs from its evaluation manifest"
            )
    summary.update(
        {
            "mode": configuration.get("mode"),
            "samples": configuration.get("samples"),
            "answer_scorer_id": configuration.get("answer_scorer_id"),
            "evaluator_code_sha256": configuration.get("evaluator_code_sha256"),
            "adapter_path": (
                configuration.get("adapter", {}).get("path")
                if isinstance(configuration.get("adapter"), Mapping)
                else None
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
            "completed_at": manifest.get("completed_at"),
            "protocol_extension_path": (
                configuration.get("protocol_extension", {}).get("path")
                if isinstance(configuration.get("protocol_extension"), Mapping)
                else None
            ),
            "protocol_extension_sha256": (
                configuration.get("protocol_extension", {}).get("sha256")
                if isinstance(configuration.get("protocol_extension"), Mapping)
                else None
            ),
        }
    )
    return summary


def summarize_training_run(manifest_path: Path) -> dict[str, Any]:
    """Extract one normalized resource row from a training manifest."""
    manifest = _read_json(manifest_path)
    if "configuration" not in manifest or "environment" not in manifest:
        raise ReportingError(f"not a training manifest: {manifest_path}")
    configuration = manifest["configuration"]
    environment = manifest["environment"]
    metrics = manifest.get("training_metrics") or {}
    trainable = metrics.get("trainable_parameters") or {}
    adapter = manifest.get("adapter") or {}
    hardware = environment.get("hardware") or {}
    packages = environment.get("packages") or {}
    return {
        "experiment_id": manifest.get("experiment_id"),
        "status": manifest.get("status"),
        "model": configuration.get("model"),
        "model_revision": (manifest.get("model") or {}).get("resolved_revision"),
        "precision_or_quantization": configuration.get("model_precision"),
        "hardware_model": hardware.get("model"),
        "hardware_chip": hardware.get("chip"),
        "hardware_memory_bytes": hardware.get("memory_bytes"),
        "python": environment.get("python"),
        "mlx": packages.get("mlx"),
        "mlx_lm": packages.get("mlx-lm"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "peak_mlx_memory_gb": metrics.get("peak_mlx_memory_gb"),
        "peak_process_bytes": manifest.get("peak_process_bytes"),
        "trainable_parameters": trainable.get("trainable_count"),
        "total_parameters": trainable.get("total_count"),
        "adapter_path": adapter.get("path") or manifest.get("adapter_path"),
        "adapter_bytes": adapter.get("bytes"),
        "adapter_sha256": adapter.get("sha256"),
        "selected_weights_path": adapter.get("selected_weights_path"),
        "selected_weights_bytes": adapter.get("selected_weights_bytes"),
        "selected_weights_sha256": adapter.get("selected_weights_sha256"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_manifests(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return sorted(root.rglob("manifest.json"))


def _reproducibility_markdown(
    evaluation_rows: Sequence[Mapping[str, Any]],
    resource_rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
) -> str:
    observed_benchmarks = {str(row.get("benchmark")) for row in evaluation_rows}
    lines = [
        "# Reviewer-rerun reproducibility note",
        "",
        f"Generated: {generated_at}",
        "",
        "This note is generated from run manifests and raw prediction records. "
        "Unknown fields are left blank rather than reconstructed.",
        "",
        "## Evaluation protocol",
        "",
        "- Evaluator: `src/evaluate_v2.py`",
        f"- Prompt version: `{PROMPT_VERSION}`.",
        f'- Task instruction: "{TASK_INSTRUCTION}"',
        "- Prompt layout: one user chat message containing the task instruction, "
        "then each demonstration as `Question: ...\\nAnswer: ...`, then the target "
        "as `Question: ...\\nAnswer:`.",
        "- GSM8K: 4 fixed examples at train indices 0, 1, 2, and 3; all 1,319 "
        "test rows are targets.",
        "- MultiArith: zero-shot; all 180 test rows are targets.",
        "- SVAMP: zero-shot; all 300 rows in the pinned ChilleD test split are "
        "targets.",
        *(
            [
                "- GSM-Hard: all 1,319 target rows from the pinned "
                "`reasoning-machines/gsm-hard` train-labelled split; 4 fixed "
                "demonstrations at indices 0, 1, 2, and 3 come from the pinned "
                "GSM8K training split.",
                "- GSM-Hard interpretation: an in-distribution large-number "
                "perturbation for numerical robustness, not a fully independent "
                "out-of-distribution corpus; automatically perturbed questions "
                "can contain awkward quantities.",
            ]
            if "gsm_hard" in observed_benchmarks
            else []
        ),
        "- Chat template: the selected model tokenizer's template, with Qwen "
        "thinking explicitly disabled for the frozen primary protocol.",
        "- Required output: final `#### <number>` marker.",
        "- Extraction: last marked number only; no arbitrary-number fallback.",
        "- Comparison: comma-stripped, finite, normalized `Decimal` equality; "
        "missing or malformed marked answers are incorrect.",
        "- Primary decoding: greedy, 1 sample, temperature 0, maximum 512 new tokens.",
        "- Secondary decoding: SC@5 only after the frozen timing gate, temperature "
        "0.7, top-p 0.95, top-k 20, maximum 512 new tokens; ties use a separate "
        "greedy sample and then numeric/lexical order.",
        "- Randomness: base seed 42; each sample seed is the first 32 bits of "
        "SHA-256 over the seed, experiment ID, benchmark, example index, and "
        "sample index.",
        "- Base-model provenance: reviewer-rerun base numbers are re-run locally "
        "with this evaluator; submitted/legacy base numbers are not copied into "
        "the corrected tables.",
        "",
        "## Frozen model identifiers",
        "",
        "| Model | Revision |",
        "|---|---|",
        *(
            f"| `{model}` | `{revision}` |"
            for model, revision in MODEL_REVISIONS.items()
        ),
        "",
        "## Evaluation runs",
        "",
    ]
    if not evaluation_rows:
        lines.append("No completed evaluation manifests were found.")
    else:
        lines.extend(
            [
                "| Experiment | Model | Revision | Benchmark | Mode | n | Accuracy | Wilson 95% CI |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            (
                "| {experiment_id} | {model} | {model_revision} | {benchmark} | {mode} | "
                "{observed_rows} | {accuracy_percent:.2f}% | "
                "[{wilson_lower_percent:.2f}%, {wilson_upper_percent:.2f}%] |".format(
                    **row
                )
            )
            for row in evaluation_rows
        )
    lines.extend(["", "## Training runs", ""])
    if not resource_rows:
        lines.append("No completed training manifests were found.")
    else:
        lines.extend(
            [
                "| Experiment | Model revision | Precision | Hardware | Time (s) | "
                "Peak MLX (GB) | Trainable / total | Adapter bytes | Adapter path |",
                "|---|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        lines.extend(
            (
                "| {experiment_id} | {model}@{model_revision} | "
                "{precision_or_quantization} | {hardware_chip} / "
                "{hardware_memory_bytes} bytes | {elapsed_seconds} | "
                "{peak_mlx_memory_gb} | {trainable_parameters} / "
                "{total_parameters} | {adapter_bytes} | {adapter_path} |".format(**row)
            )
            for row in resource_rows
        )
    lines.extend(
        [
            "",
            "## Provenance rule",
            "",
            "Only results produced by the frozen reviewer-rerun protocol are "
            "directly comparable. Submitted/legacy evaluator rows are historical "
            "context and are not merged into these tables.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_reports(
    evaluation_root: Path,
    training_root: Path,
    output_root: Path,
    *,
    require_full: bool = True,
    require_mandatory_matrix: bool = False,
    require_gsmhard_matrix: bool = False,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Discover run manifests and generate canonical compact reports."""
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_errors: list[dict[str, str]] = []
    for manifest_path in _find_manifests(evaluation_root):
        try:
            evaluation_rows.append(
                summarize_evaluation_run(
                    manifest_path,
                    require_full=require_full,
                    confidence=confidence,
                )
            )
        except ReportingError as exc:
            evaluation_errors.append({"path": str(manifest_path), "error": str(exc)})

    resource_rows: list[dict[str, Any]] = []
    resource_errors: list[dict[str, str]] = []
    for manifest_path in _find_manifests(training_root):
        try:
            resource_rows.append(summarize_training_run(manifest_path))
        except ReportingError as exc:
            resource_errors.append({"path": str(manifest_path), "error": str(exc)})

    evaluation_rows.sort(
        key=lambda row: (
            str(row.get("experiment_id")),
            str(row.get("benchmark")),
            str(row.get("mode")),
        )
    )
    resource_rows.sort(key=lambda row: str(row.get("experiment_id")))
    matrix_validation = None
    if require_mandatory_matrix and require_gsmhard_matrix:
        raise ReportingError("select only one matrix validation rule")
    if require_mandatory_matrix:
        matrix_validation = validate_mandatory_matrix(evaluation_rows, resource_rows)
    if require_gsmhard_matrix:
        matrix_validation = validate_gsmhard_matrix(evaluation_rows)
    summary_path = output_root / "summaries" / "summary.csv"
    resource_path = output_root / "summaries" / "resource_summary.csv"
    reproducibility_path = output_root / "reproducibility" / "reproducibility.md"
    _write_csv(summary_path, evaluation_rows)
    _write_csv(resource_path, resource_rows)
    generated_at = utc_now()
    reproducibility_path.parent.mkdir(parents=True, exist_ok=True)
    reproducibility_path.write_text(
        _reproducibility_markdown(
            evaluation_rows,
            resource_rows,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    report_manifest = {
        "generated_at": generated_at,
        "protocol": protocol_identity(),
        "protocol_extensions": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(
                {
                    (
                        str(row["protocol_extension_path"]),
                        str(row["protocol_extension_sha256"]),
                    )
                    for row in evaluation_rows
                    if row.get("protocol_extension_path")
                    and row.get("protocol_extension_sha256")
                }
            )
        ],
        "require_full": require_full,
        "require_mandatory_matrix": require_mandatory_matrix,
        "require_gsmhard_matrix": require_gsmhard_matrix,
        "matrix_validation": matrix_validation,
        "confidence": confidence,
        "evaluation_runs": len(evaluation_rows),
        "training_runs": len(resource_rows),
        "evaluation_errors": evaluation_errors,
        "training_errors": resource_errors,
        "outputs": {
            "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
            "resources": {
                "path": str(resource_path),
                "sha256": file_sha256(resource_path),
            },
            "reproducibility": {
                "path": str(reproducibility_path),
                "sha256": file_sha256(reproducibility_path),
            },
        },
    }
    write_json(output_root / "report_manifest.json", report_manifest)
    if require_full and evaluation_errors:
        raise ReportingError(
            f"refused incomplete evaluation manifests: {len(evaluation_errors)}"
        )
    return report_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("runs/reviewer_rerun/evaluation"),
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("runs/reviewer_rerun/training"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/reviewer_rerun"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-mandatory-matrix", action="store_true")
    parser.add_argument("--require-gsmhard-matrix", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all compact rerun reports."""
    args = _build_parser().parse_args(argv)
    manifest = generate_reports(
        args.evaluation_root,
        args.training_root,
        args.output_root,
        require_full=not args.allow_partial,
        require_mandatory_matrix=args.require_mandatory_matrix,
        require_gsmhard_matrix=args.require_gsmhard_matrix,
        confidence=args.confidence,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

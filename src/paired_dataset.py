"""Build and validate matched Socratic/non-Socratic training datasets.

The canonical record stores each guiding question separately from the shared
declarative reasoning. Both experimental arms are rendered from that one
record, making the presence of guiding questions the only content difference.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.rerun_utils import (
    file_sha256,
    object_sha256,
    protocol_identity,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = "1.1"
MIN_SOLUTION_STEPS = 2
MAX_SOLUTION_STEPS = 6
MIN_SYNTHETIC_QUESTION_CHARS = 20
MIN_REASONING_CHARS = 60
MAX_REASONING_CHARS = 300
DEFAULT_MIN_SOLUTION_CHARS = 120
DEFAULT_MAX_SOLUTION_CHARS = 2000
DEFAULT_NGRAM_SIZE = 5
DEFAULT_JACCARD_THRESHOLD = 0.85
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 42
FINAL_POSITIVE_INTEGER = re.compile(r"####\s*([1-9]\d*)\s*$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
CALCULATION = re.compile(
    r"(?<![\w.])(?:[-+]?(?:[$€£]\s*)?(?:\d|\())"
    r"[\d,\s.$€£()+\-−*/×÷=]*="
    r"[\d,\s.$€£()+\-−*/×÷=]*(?:\d|\))"
)
EQUALITY_RESULT = re.compile(rf"=\s*(?:[$€£]\s*)?({NUMBER})(?![\d,.])(?!\s*[+\-−*/×÷])")


class DatasetValidationError(ValueError):
    """Raised when a canonical or paired dataset violates an invariant."""


@dataclass(frozen=True)
class SolutionStep:
    """One separately renderable guiding question and reasoning statement."""

    guiding_question: str
    reasoning: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> SolutionStep:
        """Validate and construct a solution step."""
        guiding_question = _required_text(value, "guiding_question")
        reasoning = _required_text(value, "reasoning")
        if not guiding_question.endswith("?"):
            msg = f"solution_steps[{index}].guiding_question must end with '?'"
            raise DatasetValidationError(msg)
        if not MIN_REASONING_CHARS <= len(reasoning) <= MAX_REASONING_CHARS:
            msg = (
                f"solution_steps[{index}].reasoning must contain between "
                f"{MIN_REASONING_CHARS} and {MAX_REASONING_CHARS} characters"
            )
            raise DatasetValidationError(msg)
        return cls(guiding_question=guiding_question, reasoning=reasoning)

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable step."""
        return {
            "guiding_question": self.guiding_question,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class CanonicalExample:
    """Validated canonical synthetic example used to render both arms."""

    schema_version: str
    example_id: str
    source_id: str
    variant_id: int
    source_question: str
    source_solution: str
    synthetic_question: str
    solution_steps: tuple[SolutionStep, ...]
    final_answer: str
    generation: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CanonicalExample:
        """Validate and construct a canonical example from decoded JSON."""
        schema_version = _required_text(value, "schema_version")
        if schema_version != SCHEMA_VERSION:
            msg = f"Unsupported schema_version {schema_version!r}"
            raise DatasetValidationError(msg)

        example_id = _required_identifier(value, "example_id")
        source_id = _required_identifier(value, "source_id")
        variant_id = value.get("variant_id")
        if isinstance(variant_id, bool) or not isinstance(variant_id, int):
            raise DatasetValidationError("variant_id must be an integer")
        if variant_id < 1:
            raise DatasetValidationError("variant_id must be at least 1")

        synthetic_question = _required_text(value, "synthetic_question")
        if (
            len(synthetic_question) < MIN_SYNTHETIC_QUESTION_CHARS
            or "?" not in synthetic_question
        ):
            raise DatasetValidationError(
                "synthetic_question must contain at least "
                f"{MIN_SYNTHETIC_QUESTION_CHARS} characters and a direct question "
                "ending with '?'"
            )

        raw_steps = value.get("solution_steps")
        if not isinstance(raw_steps, list) or not (
            MIN_SOLUTION_STEPS <= len(raw_steps) <= MAX_SOLUTION_STEPS
        ):
            raise DatasetValidationError(
                "solution_steps must contain between "
                f"{MIN_SOLUTION_STEPS} and {MAX_SOLUTION_STEPS} steps"
            )
        steps: list[SolutionStep] = []
        step_signatures: set[tuple[str, str]] = set()
        for index, step in enumerate(raw_steps):
            if not isinstance(step, Mapping):
                msg = f"solution_steps[{index}] must be an object"
                raise DatasetValidationError(msg)
            parsed_step = SolutionStep.from_mapping(step, index=index)
            signature = (
                normalize_text(parsed_step.guiding_question),
                normalize_text(parsed_step.reasoning),
            )
            if signature in step_signatures:
                raise DatasetValidationError(f"solution_steps[{index}] is duplicated")
            step_signatures.add(signature)
            steps.append(parsed_step)

        final_answer = _required_text(value, "final_answer")
        if FINAL_POSITIVE_INTEGER.fullmatch(final_answer) is None:
            raise DatasetValidationError(
                "final_answer must be one positive integer in final '#### N' format"
            )
        _validate_explicit_calculations(steps, final_answer)

        raw_generation = value.get("generation", {})
        if not isinstance(raw_generation, Mapping):
            raise DatasetValidationError("generation must be an object")

        return cls(
            schema_version=schema_version,
            example_id=example_id,
            source_id=source_id,
            variant_id=variant_id,
            source_question=_required_text(value, "source_question"),
            source_solution=_required_text(value, "source_solution"),
            synthetic_question=synthetic_question,
            solution_steps=tuple(steps),
            final_answer=final_answer,
            generation=dict(raw_generation),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable canonical record."""
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "source_id": self.source_id,
            "variant_id": self.variant_id,
            "source_question": self.source_question,
            "source_solution": self.source_solution,
            "synthetic_question": self.synthetic_question,
            "solution_steps": [step.as_dict() for step in self.solution_steps],
            "final_answer": self.final_answer,
            "generation": self.generation,
        }

    @property
    def shared_reasoning(self) -> str:
        """Return exactly the declarative reasoning shared by both arms."""
        return "\n".join(step.reasoning for step in self.solution_steps)


def _required_text(value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        msg = f"{key} must be a non-empty string"
        raise DatasetValidationError(msg)
    return raw.strip()


def _required_identifier(value: Mapping[str, Any], key: str) -> str:
    identifier = _required_text(value, key)
    if ID_PATTERN.fullmatch(identifier) is None:
        msg = f"{key} contains unsupported characters: {identifier!r}"
        raise DatasetValidationError(msg)
    return identifier


def _evaluate_arithmetic_expression(expression: str) -> Fraction | None:
    """Safely evaluate a generated arithmetic expression.

    Returning ``None`` keeps the validator conservative when a candidate uses
    unsupported notation. Only numeric literals, parentheses, unary signs,
    and the four arithmetic operators are accepted.
    """
    normalized = (
        expression.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .strip()
    )
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    try:
        parsed = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise ZeroDivisionError
                result = left / right
            return result
        raise ValueError("unsupported arithmetic expression")

    try:
        return evaluate(parsed)
    except (ValueError, ZeroDivisionError):
        return None


def _contains_arithmetic_operator(expression: str) -> bool:
    """Return whether an equality segment contains an actual operation."""
    unsigned = expression.strip().lstrip("+-").strip()
    return any(operator in unsigned for operator in ("+", "-", "−", "*", "/", "×", "÷"))


def _validate_explicit_calculations(
    steps: Sequence[SolutionStep], final_answer: str
) -> None:
    """Verify explicit arithmetic equalities and their link to the final answer."""
    final_result: Fraction | None = None
    for step_index, step in enumerate(steps):
        result_candidates: list[tuple[int, Fraction]] = []
        for match in CALCULATION.finditer(step.reasoning):
            expressions = [part.strip() for part in match.group(0).split("=")]
            # Require an operation on the left-hand side. This avoids treating
            # prose such as "length minus 2 = 2*5" as the false equation
            # ``2 = 2*5`` while still validating explicit arithmetic.
            if not _contains_arithmetic_operator(expressions[0]):
                continue
            evaluated = [_evaluate_arithmetic_expression(part) for part in expressions]
            if any(value is None for value in evaluated):
                continue
            values = [value for value in evaluated if value is not None]
            if any(value != values[0] for value in values[1:]):
                raise DatasetValidationError(
                    f"solution_steps[{step_index}] contains an incorrect equality: "
                    f"{match.group(0)}"
                )
            result_candidates.append((match.end(), values[-1]))

        # Unit-bearing calculations are intentionally not evaluated (for
        # example, ``30 foxes * 3 rabbits/fox = 90 rabbits``), but their
        # explicit numeric result still links the worked trace to the final
        # answer. Ignore a numeric token when it begins another expression,
        # such as the ``2`` in ``= 2*5``.
        for match in EQUALITY_RESULT.finditer(step.reasoning):
            value = _evaluate_arithmetic_expression(match.group(1))
            if value is not None:
                result_candidates.append((match.end(), value))

        if result_candidates:
            final_result = max(result_candidates, key=lambda candidate: candidate[0])[1]

    if final_result is not None:
        final_match = FINAL_POSITIVE_INTEGER.fullmatch(final_answer)
        if final_match is None or final_result != Fraction(final_match.group(1)):
            raise DatasetValidationError(
                "the last explicit equality does not match final_answer"
            )


def normalize_text(text: str) -> str:
    """Normalize text for exact and n-gram duplicate checks."""
    lowered = text.casefold()
    collapsed = re.sub(r"[^\w.+-]+", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", collapsed).strip()


def generate_ngrams(
    text: str, n: int = DEFAULT_NGRAM_SIZE
) -> frozenset[tuple[str, ...]]:
    """Return a reusable set of token n-grams."""
    if n <= 0:
        raise ValueError("n must be greater than zero")
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
    )


def jaccard_similarity(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
) -> float:
    """Calculate exact set Jaccard similarity."""
    if not left and not right:
        return 1.0
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0


def filter_and_deduplicate(
    raw_records: Iterable[Mapping[str, Any]],
    *,
    min_solution_chars: int = DEFAULT_MIN_SOLUTION_CHARS,
    max_solution_chars: int = DEFAULT_MAX_SOLUTION_CHARS,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> tuple[list[CanonicalExample], list[dict[str, Any]]]:
    """Validate, filter, and exactly apply the near-duplicate policy.

    An inverted n-gram index restricts exact Jaccard calculations to records
    that share at least one n-gram. The final threshold decision always uses
    exact set Jaccard similarity.
    """
    if min_solution_chars < 0 or max_solution_chars < min_solution_chars:
        raise ValueError("invalid solution character bounds")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")

    accepted: list[CanonicalExample] = []
    accepted_ngrams: list[frozenset[tuple[str, ...]]] = []
    seen_questions: dict[str, str] = {}
    seen_ids: set[str] = set()
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    rejected: list[dict[str, Any]] = []

    for input_index, raw in enumerate(raw_records):
        raw_example_id = raw.get("example_id", f"input-{input_index}")
        try:
            record = CanonicalExample.from_mapping(raw)
        except DatasetValidationError as exc:
            rejected.append(
                {
                    "input_index": input_index,
                    "example_id": str(raw_example_id),
                    "reason": "schema_validation",
                    "detail": str(exc),
                }
            )
            continue

        if record.example_id in seen_ids:
            rejected.append(
                {
                    "input_index": input_index,
                    "example_id": record.example_id,
                    "reason": "duplicate_example_id",
                }
            )
            continue

        solution_length = len(record.shared_reasoning) + 1 + len(record.final_answer)
        if solution_length < min_solution_chars or solution_length > max_solution_chars:
            rejected.append(
                {
                    "input_index": input_index,
                    "example_id": record.example_id,
                    "reason": "solution_length",
                    "solution_characters": solution_length,
                    "minimum": min_solution_chars,
                    "maximum": max_solution_chars,
                }
            )
            continue

        normalized_question = normalize_text(record.synthetic_question)
        if normalized_question in seen_questions:
            rejected.append(
                {
                    "input_index": input_index,
                    "example_id": record.example_id,
                    "reason": "exact_duplicate_question",
                    "matched_example_id": seen_questions[normalized_question],
                    "similarity": 1.0,
                }
            )
            continue

        ngrams = generate_ngrams(record.synthetic_question, ngram_size)
        candidate_indices: set[int] = set()
        for ngram in ngrams:
            candidate_indices.update(inverted.get(ngram, ()))

        duplicate_index: int | None = None
        duplicate_score = 0.0
        for candidate_index in sorted(candidate_indices):
            score = jaccard_similarity(ngrams, accepted_ngrams[candidate_index])
            if score >= jaccard_threshold and score > duplicate_score:
                duplicate_index = candidate_index
                duplicate_score = score

        if duplicate_index is not None:
            rejected.append(
                {
                    "input_index": input_index,
                    "example_id": record.example_id,
                    "reason": "near_duplicate_question",
                    "matched_example_id": accepted[duplicate_index].example_id,
                    "similarity": duplicate_score,
                    "threshold": jaccard_threshold,
                    "ngram_size": ngram_size,
                }
            )
            continue

        accepted_index = len(accepted)
        accepted.append(record)
        accepted_ngrams.append(ngrams)
        seen_ids.add(record.example_id)
        seen_questions[normalized_question] = record.example_id
        for ngram in ngrams:
            inverted[ngram].add(accepted_index)

    return accepted, rejected


def deterministic_select(
    records: Sequence[CanonicalExample], target_count: int, *, seed: int
) -> list[CanonicalExample]:
    """Select an exact-size deterministic subset and restore stable input order."""
    if target_count <= 0:
        raise ValueError("target_count must be greater than zero")
    if len(records) < target_count:
        msg = f"Need {target_count} accepted examples but only have {len(records)}"
        raise DatasetValidationError(msg)
    if len(records) == target_count:
        return list(records)

    ranked = sorted(
        enumerate(records),
        key=lambda item: (
            object_sha256({"seed": seed, "id": item[1].example_id}),
            item[0],
        ),
    )
    selected_indices = {index for index, _ in ranked[:target_count]}
    return [record for index, record in enumerate(records) if index in selected_indices]


def split_by_source(
    records: Sequence[CanonicalExample],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[CanonicalExample], list[CanonicalExample], dict[str, Any]]:
    """Split by source ID while approaching the requested row fraction."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not records:
        raise DatasetValidationError("cannot split an empty dataset")

    groups: dict[str, list[CanonicalExample]] = defaultdict(list)
    for record in records:
        groups[record.source_id].append(record)

    source_ids = sorted(groups)
    random.Random(seed).shuffle(source_ids)
    target_validation_rows = round(len(records) * validation_fraction)

    validation_sources: set[str] = set()
    validation_rows = 0
    remaining: list[str] = []
    for source_id in source_ids:
        group_size = len(groups[source_id])
        if validation_rows + group_size <= target_validation_rows:
            validation_sources.add(source_id)
            validation_rows += group_size
        else:
            remaining.append(source_id)

    if remaining:
        best_source = min(
            remaining,
            key=lambda source_id: (
                abs(
                    target_validation_rows - (validation_rows + len(groups[source_id]))
                ),
                source_id,
            ),
        )
        current_distance = abs(target_validation_rows - validation_rows)
        candidate_distance = abs(
            target_validation_rows - (validation_rows + len(groups[best_source]))
        )
        if candidate_distance < current_distance:
            validation_sources.add(best_source)
            validation_rows += len(groups[best_source])

    train = [record for record in records if record.source_id not in validation_sources]
    validation = [
        record for record in records if record.source_id in validation_sources
    ]
    train_sources = {record.source_id for record in train}
    if train_sources & validation_sources:
        raise AssertionError("source leakage detected after group split")

    split_manifest = {
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "target_validation_rows": target_validation_rows,
        "actual_train_rows": len(train),
        "actual_validation_rows": len(validation),
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "validation_source_ids": sorted(validation_sources),
    }
    return train, validation, split_manifest


def render_answer(record: CanonicalExample, *, socratic: bool) -> str:
    """Render a completion while keeping declarative reasoning byte-identical."""
    lines: list[str] = []
    for step in record.solution_steps:
        if socratic:
            lines.append(step.guiding_question)
        lines.append(step.reasoning)
    lines.append(record.final_answer)
    return "\n".join(lines)


def render_record(record: CanonicalExample, *, socratic: bool) -> dict[str, Any]:
    """Render one MLX-LM prompt/completion record with provenance IDs."""
    return {
        "example_id": record.example_id,
        "source_id": record.source_id,
        "variant_id": record.variant_id,
        "question": record.synthetic_question,
        "answer": render_answer(record, socratic=socratic),
    }


def validate_rendered_pair(
    canonical: Sequence[CanonicalExample],
    socratic_rows: Sequence[Mapping[str, Any]],
    non_socratic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the rendered arms exactly match the canonical source."""
    if len(socratic_rows) != len(non_socratic_rows):
        raise DatasetValidationError("rendered arm row counts differ")
    if len(canonical) != len(socratic_rows):
        raise DatasetValidationError("rendered row count differs from canonical count")

    for index, (record, socratic, non_socratic) in enumerate(
        zip(canonical, socratic_rows, non_socratic_rows, strict=True)
    ):
        for key, expected in (
            ("example_id", record.example_id),
            ("source_id", record.source_id),
            ("variant_id", record.variant_id),
            ("question", record.synthetic_question),
        ):
            if socratic.get(key) != expected or non_socratic.get(key) != expected:
                msg = f"row {index} differs for {key}"
                raise DatasetValidationError(msg)

        expected_socratic = render_answer(record, socratic=True)
        expected_non_socratic = render_answer(record, socratic=False)
        if socratic.get("answer") != expected_socratic:
            raise DatasetValidationError(f"row {index} has invalid Socratic rendering")
        if non_socratic.get("answer") != expected_non_socratic:
            raise DatasetValidationError(
                f"row {index} has invalid non-Socratic rendering"
            )
        for step in record.solution_steps:
            if step.guiding_question in expected_non_socratic:
                raise DatasetValidationError(
                    f"row {index} leaked a guiding question into non-Socratic output"
                )
            if (
                step.reasoning not in expected_socratic
                or step.reasoning not in expected_non_socratic
            ):
                raise DatasetValidationError(f"row {index} lost shared reasoning")

    return {
        "status": "passed",
        "rows": len(canonical),
        "canonical_sequence_sha256": object_sha256(
            [record.example_id for record in canonical]
        ),
        "checked_at": utc_now(),
    }


def build_paired_dataset(
    input_path: Path,
    output_root: Path,
    *,
    target_count: int | None = None,
    min_solution_chars: int = DEFAULT_MIN_SOLUTION_CHARS,
    max_solution_chars: int = DEFAULT_MAX_SOLUTION_CHARS,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Filter canonical input, split by source, render both arms, and audit."""
    raw_records = list(read_jsonl(input_path))
    accepted, rejected = filter_and_deduplicate(
        raw_records,
        min_solution_chars=min_solution_chars,
        max_solution_chars=max_solution_chars,
        ngram_size=ngram_size,
        jaccard_threshold=jaccard_threshold,
    )
    preselection_count = len(accepted)
    if target_count is not None:
        accepted = deterministic_select(accepted, target_count, seed=seed)

    train, validation, split_manifest = split_by_source(
        accepted,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    canonical_dir = output_root / "canonical"
    audit_dir = output_root / "audits"
    manifest_dir = output_root / "manifests"
    socratic_dir = output_root / "socratic"
    non_socratic_dir = output_root / "non_socratic"

    accepted_path = canonical_dir / "accepted.jsonl"
    rejected_path = audit_dir / "rejections.jsonl"
    write_jsonl(accepted_path, (record.as_dict() for record in accepted))
    write_jsonl(rejected_path, rejected)

    split_rows = {"train": train, "valid": validation}
    rendered_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, records in split_rows.items():
        socratic_rows = [render_record(record, socratic=True) for record in records]
        non_socratic_rows = [
            render_record(record, socratic=False) for record in records
        ]
        pair_report = validate_rendered_pair(records, socratic_rows, non_socratic_rows)
        write_jsonl(socratic_dir / f"{split}.jsonl", socratic_rows)
        write_jsonl(non_socratic_dir / f"{split}.jsonl", non_socratic_rows)
        rendered_by_split[split] = {
            "socratic": socratic_rows,
            "non_socratic": non_socratic_rows,
            "pair_report": [pair_report],
        }

    output_files = {
        "canonical": accepted_path,
        "rejections": rejected_path,
        "socratic_train": socratic_dir / "train.jsonl",
        "socratic_valid": socratic_dir / "valid.jsonl",
        "non_socratic_train": non_socratic_dir / "train.jsonl",
        "non_socratic_valid": non_socratic_dir / "valid.jsonl",
    }
    pair_report = {
        "status": "passed",
        "protocol": protocol_identity(),
        "total_rows": len(accepted),
        "train": rendered_by_split["train"]["pair_report"][0],
        "valid": rendered_by_split["valid"]["pair_report"][0],
    }
    write_json(manifest_dir / "pairing_report.json", pair_report)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_rows": len(raw_records),
        "accepted_before_selection": preselection_count,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "target_count": target_count,
        "filter": {
            "min_solution_chars": min_solution_chars,
            "max_solution_chars": max_solution_chars,
            "ngram_size": ngram_size,
            "jaccard_threshold": jaccard_threshold,
        },
        "split": split_manifest,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in output_files.items()
        },
        "pairing_report": pair_report,
    }
    write_json(manifest_dir / "dataset_manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate canonical JSONL")
    validate.add_argument("input", type=Path)

    render = subparsers.add_parser(
        "render", help="filter, split, and render both matched arms"
    )
    render.add_argument("input", type=Path)
    render.add_argument("--output-root", type=Path, default=Path("data/reviewer_rerun"))
    render.add_argument("--target-count", type=int)
    render.add_argument(
        "--min-solution-chars", type=int, default=DEFAULT_MIN_SOLUTION_CHARS
    )
    render.add_argument(
        "--max-solution-chars", type=int, default=DEFAULT_MAX_SOLUTION_CHARS
    )
    render.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    render.add_argument(
        "--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD
    )
    render.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    render.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical validator or paired renderer."""
    args = _build_parser().parse_args(argv)
    if args.command == "validate":
        rows = list(read_jsonl(args.input))
        validated = [CanonicalExample.from_mapping(row) for row in rows]
        print(json.dumps({"status": "passed", "rows": len(validated)}, indent=2))
        return 0

    manifest = build_paired_dataset(
        args.input,
        args.output_root,
        target_count=args.target_count,
        min_solution_chars=args.min_solution_chars,
        max_solution_chars=args.max_solution_chars,
        ngram_size=args.ngram_size,
        jaccard_threshold=args.jaccard_threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

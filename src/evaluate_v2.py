"""Correct, resumable evaluation for the reviewer-rerun experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.rerun_utils import (
    append_jsonl,
    directory_sha256,
    file_sha256,
    object_sha256,
    protocol_identity,
    read_jsonl,
    utc_now,
    write_json,
)

PROMPT_VERSION = "arithmetic-eval-v2"
ANSWER_SCORER_ID = "submitted-last-marked-decimal-v1"
ANSWER_SCORER_RULE = (
    "Select the last ####-marked integer or decimal anywhere in the response; "
    "ignore trailing text; do not fall back to unmarked numbers."
)
TERMINAL_DIAGNOSTIC_ID = "terminal-marked-decimal-diagnostic-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
ANSWER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MARKED_ANSWER = re.compile(rf"####\s*({ANSWER_PATTERN})")
TERMINAL_MARKED_ANSWER = re.compile(rf"####\s*({ANSWER_PATTERN})(?=\s*(?:[.!])?\s*$)")
PLAIN_ANSWER = re.compile(rf"^\s*({ANSWER_PATTERN})\s*$")

TASK_INSTRUCTION = (
    "Solve the arithmetic word problem using concise step-by-step reasoning. "
    "End with the final numeric answer on its own line in exactly this format: "
    "#### <number>."
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Pinned logical definition of one evaluation benchmark."""

    key: str
    dataset: str
    config: str | None
    target_split: str
    few_shot_split: str | None
    few_shot_indices: tuple[int, ...]
    expected_rows: int
    question_column: str
    answer_column: str
    revision: str | None = None


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        key="gsm8k",
        dataset="openai/gsm8k",
        config="main",
        target_split="test",
        few_shot_split="train",
        few_shot_indices=(0, 1, 2, 3),
        expected_rows=1319,
        question_column="question",
        answer_column="answer",
        revision="740312add88f781978c0658806c59bc2815b9866",
    ),
    "multiarith": BenchmarkSpec(
        key="multiarith",
        dataset="ChilleD/MultiArith",
        config=None,
        target_split="test",
        few_shot_split=None,
        few_shot_indices=(),
        expected_rows=180,
        question_column="question",
        answer_column="final_ans",
        revision="144d44c3fb87c0b9097ac9593c789e716a282e3e",
    ),
    "svamp": BenchmarkSpec(
        key="svamp",
        dataset="ChilleD/SVAMP",
        config=None,
        target_split="test",
        few_shot_split=None,
        few_shot_indices=(),
        expected_rows=300,
        question_column="question_concat",
        answer_column="Answer",
        revision="5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
    ),
}

MODEL_REVISIONS = {
    "mlx-community/Qwen3-0.6B-bf16": "42096995f6402fde107068cf530136fe64b604f8",
    "mlx-community/Qwen3-1.7B-4bit": "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
    "mlx-community/Llama-3.2-1B-Instruct-MLXTuned": (
        "7247cd8c176bbc558293c9b4750e9f97b5beb319"
    ),
}


class EvaluationError(RuntimeError):
    """Raised when evaluation provenance or completeness is unsafe."""


def evaluator_identity() -> dict[str, str]:
    """Return the scorer and source-code identity used for safe resume."""
    return {
        "evaluator": "evaluate_v2",
        "evaluator_code_sha256": file_sha256(Path(__file__)),
        "answer_scorer_id": ANSWER_SCORER_ID,
        "answer_scorer_rule": ANSWER_SCORER_RULE,
    }


def normalize_numeric(value: str) -> str | None:
    """Return a canonical finite decimal string, or None for invalid input."""
    compact = value.strip().replace(",", "")
    try:
        number = Decimal(compact)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    # Decimal.normalize()/quantize(Decimal(1)) use the active context precision
    # and can raise InvalidOperation for otherwise valid integers >= 10**28.
    # Fixed-point formatting preserves every parsed digit without consulting the
    # context; stripping fractional zeroes then gives one canonical vote key.
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_marked_number(text: str) -> str | None:
    """Extract and normalize the last explicitly marked numeric answer.

    The submitted-paper rule selects the last ``####``-marked decimal anywhere
    in the completed response. Text after that marked number (including a stray
    question mark) does not invalidate it. No unmarked-number fallback is used.
    """
    matches = list(MARKED_ANSWER.finditer(text))
    if not matches:
        return None
    return normalize_numeric(matches[-1].group(1))


def extract_terminal_marked_number(text: str) -> str | None:
    """Extract a marked answer only when it terminates the response.

    This stricter rule is retained as a format-adherence diagnostic. It is not
    the primary exact-match rule declared in the submitted manuscript.
    """
    matches = list(TERMINAL_MARKED_ANSWER.finditer(text))
    if not matches:
        return None
    return normalize_numeric(matches[-1].group(1))


def normalize_reference_answer(value: Any) -> str | None:
    """Normalize a benchmark reference, accepting marked or bare numeric data."""
    text = str(value).strip()
    if "####" in text:
        return extract_marked_number(text)
    match = PLAIN_ANSWER.fullmatch(text)
    return normalize_numeric(match.group(1)) if match else None


def answers_equal(predicted: str | None, reference: str | None) -> bool:
    """Compare already-normalized decimal answers exactly."""
    if predicted is None or reference is None:
        return False
    try:
        return Decimal(predicted) == Decimal(reference)
    except InvalidOperation:
        return False


def majority_vote(
    answers: Sequence[str | None], *, greedy_answer: str | None = None
) -> tuple[str | None, bool]:
    """Vote over valid answers with a predeclared deterministic tie policy.

    A tied greedy answer wins when supplied. Otherwise tied values are sorted
    numerically and then lexically, making the result independent of sample
    order. The returned boolean reports whether a tie occurred.
    """
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return None, False
    counts = Counter(valid)
    highest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest]
    tied = len(winners) > 1
    if greedy_answer in winners:
        return greedy_answer, tied
    winners.sort(key=lambda answer: (Decimal(answer), answer))
    return winners[0], tied


def derive_seed(
    base_seed: int,
    experiment_id: str,
    benchmark: str,
    example_index: int,
    sample_index: int,
) -> int:
    """Derive a stable unsigned 32-bit seed for one generated sample."""
    material = (
        f"{base_seed}\0{experiment_id}\0{benchmark}\0{example_index}\0{sample_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def build_prompt(
    target_question: str,
    few_shot_rows: Sequence[Mapping[str, Any]],
    *,
    question_column: str,
    answer_column: str,
) -> str:
    """Construct the exact semantic prompt using only supplied shot rows."""
    sections = [TASK_INSTRUCTION]
    for row in few_shot_rows:
        question = _row_text(row, question_column)
        answer = str(row.get(answer_column, "")).strip()
        if not answer:
            raise EvaluationError(f"few-shot field {answer_column!r} is empty")
        sections.append(f"Question: {question}\nAnswer: {answer}")
    sections.append(f"Question: {target_question.strip()}\nAnswer:")
    return "\n\n".join(sections)


def _row_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"dataset field {key!r} must be a non-empty string")
    return value.strip()


def render_model_prompt(
    tokenizer: Any, semantic_prompt: str, *, enable_thinking: bool
) -> str:
    """Apply the model chat template and explicitly pass thinking mode."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        return semantic_prompt
    messages = [{"role": "user", "content": semantic_prompt}]
    try:
        return apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        if enable_thinking:
            raise EvaluationError(
                "Tokenizer does not support requested enable_thinking=True"
            ) from None
        return apply_template(messages, tokenize=False, add_generation_prompt=True)


def _load_dataset_split(spec: BenchmarkSpec, split: str, revision: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for evaluation; install the locked environment."
        ) from exc
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if spec.config is None:
        return load_dataset(spec.dataset, **kwargs)
    return load_dataset(spec.dataset, spec.config, **kwargs)


def load_benchmark(
    spec: BenchmarkSpec, *, revision: str | None = None
) -> tuple[Any, list[Mapping[str, Any]], dict[str, Any]]:
    """Load the target and fixed few-shot rows from separate declared splits."""
    target = _load_dataset_split(spec, spec.target_split, revision or spec.revision)
    if len(target) != spec.expected_rows:
        raise EvaluationError(
            f"{spec.key} expected {spec.expected_rows} target rows, found {len(target)}"
        )

    shots: list[Mapping[str, Any]] = []
    shot_fingerprint = None
    if spec.few_shot_indices:
        if spec.few_shot_split is None or spec.few_shot_split == spec.target_split:
            raise EvaluationError(
                "few-shot examples must come from a separate train split"
            )
        shot_dataset = _load_dataset_split(
            spec, spec.few_shot_split, revision or spec.revision
        )
        shots = [shot_dataset[index] for index in spec.few_shot_indices]
        shot_fingerprint = getattr(shot_dataset, "_fingerprint", None)

    provenance = {
        "dataset": spec.dataset,
        "config": spec.config,
        "requested_revision": revision or spec.revision,
        "target_split": spec.target_split,
        "target_rows": len(target),
        "target_fingerprint": getattr(target, "_fingerprint", None),
        "few_shot_split": spec.few_shot_split,
        "few_shot_indices": list(spec.few_shot_indices),
        "few_shot_fingerprint": shot_fingerprint,
    }
    return target, shots, provenance


def _load_model(model_name: str, adapter_path: Path | None) -> tuple[Any, Any]:
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is required for generation; install the locked project environment."
        ) from exc
    if adapter_path is None:
        return load(model_name)
    return load(model_name, adapter_path=str(adapter_path))


def requested_model_revision(
    model_name: str, revision_override: str | None = None
) -> str | None:
    """Return the required immutable revision for a remote model identifier.

    Local directories are identified by a content hash instead. Supplying a
    Hugging Face revision alongside a local path is rejected because it would
    create two conflicting provenance identities for the same evaluation.
    """
    if Path(model_name).expanduser().exists():
        if revision_override is not None:
            raise EvaluationError(
                "--model-revision cannot be combined with a local model path"
            )
        return None
    revision = revision_override or MODEL_REVISIONS.get(model_name)
    if revision is None:
        raise EvaluationError(
            "remote model is not in the frozen registry; provide "
            "--model-revision with a 40-character commit SHA"
        )
    if REVISION_RE.fullmatch(revision) is None:
        raise EvaluationError(
            "model revision must be a 40-character lowercase commit SHA"
        )
    return revision


def materialize_evaluation_model(
    model_name: str, revision_override: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Materialize one pinned model and return its complete provenance."""
    local_path = Path(model_name).expanduser()
    requested_revision = requested_model_revision(model_name, revision_override)
    if requested_revision is None:
        resolved_path = local_path.resolve()
        return resolved_path, {
            "identifier": model_name,
            "requested_revision": None,
            "resolved_revision": None,
            "materialized_path": str(resolved_path),
            "content_sha256": directory_sha256(resolved_path),
        }

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to materialize pinned model weights"
        ) from exc

    resolved_revision = HfApi().model_info(model_name, revision=requested_revision).sha
    if resolved_revision != requested_revision:
        raise EvaluationError(
            "model revision changed: "
            f"expected {requested_revision}, resolved {resolved_revision}"
        )
    materialized_path = Path(
        snapshot_download(repo_id=model_name, revision=requested_revision)
    ).resolve()
    return materialized_path, {
        "identifier": model_name,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "materialized_path": str(materialized_path),
        "content_sha256": None,
    }


def _generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, float]:
    try:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("mlx and mlx-lm are required for generation") from exc

    mx.random.seed(seed)
    if temperature == 0:
        sampler = make_sampler(temp=0.0)
    else:
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=0.0,
        )
    started = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    return str(response), time.perf_counter() - started


def _adapter_identity(adapter_path: Path | None) -> dict[str, Any] | None:
    if adapter_path is None:
        return None
    if not adapter_path.exists():
        raise EvaluationError(f"adapter path does not exist: {adapter_path}")
    manifest: dict[str, Any] = {"path": str(adapter_path)}
    adapter_file = adapter_path / "adapters.safetensors"
    if adapter_file.exists():
        manifest["weights_sha256"] = file_sha256(adapter_file)
        manifest["weights_bytes"] = adapter_file.stat().st_size
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        manifest["config_sha256"] = file_sha256(config_file)
    return manifest


def _completed_ids(prediction_path: Path) -> set[str]:
    if not prediction_path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(prediction_path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            raise EvaluationError("prediction row is missing example_id")
        if example_id in completed:
            raise EvaluationError(f"duplicate prediction example_id: {example_id}")
        completed.add(example_id)
    return completed


def _configuration(args: argparse.Namespace, spec: BenchmarkSpec) -> dict[str, Any]:
    model_revision = requested_model_revision(
        args.model, getattr(args, "model_revision", None)
    )
    local_model_path = Path(args.model).expanduser()
    return {
        **evaluator_identity(),
        "prompt_version": PROMPT_VERSION,
        "experiment_id": args.experiment_id,
        "model": args.model,
        "model_revision": model_revision,
        "local_model_content_sha256": (
            directory_sha256(local_model_path) if local_model_path.exists() else None
        ),
        "adapter": _adapter_identity(args.adapter_path),
        "benchmark": asdict(spec),
        "dataset_revision_override": args.dataset_revision,
        "mode": args.mode,
        "samples": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "base_seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "start_index": args.start_index,
        "limit": args.limit,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run or safely resume one benchmark evaluation."""
    spec = BENCHMARKS[args.benchmark]
    if args.mode == "self_consistency" and (args.samples <= 0 or args.samples % 2 == 0):
        raise EvaluationError("self-consistency samples must be a positive odd number")
    if args.mode == "greedy" and args.samples != 1:
        raise EvaluationError("greedy evaluation requires --samples 1")

    configuration = _configuration(args, spec)
    configuration_hash = object_sha256(configuration)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "manifest.json"
    prediction_path = args.run_dir / "predictions.jsonl"

    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("configuration_sha256") != configuration_hash:
            raise EvaluationError(
                "refusing to resume: existing evaluation configuration hash differs"
            )

    target, shots, dataset_provenance = load_benchmark(
        spec, revision=args.dataset_revision
    )
    start = args.start_index
    stop = len(target) if args.limit is None else min(len(target), start + args.limit)
    if start < 0 or start >= len(target) or stop <= start:
        raise EvaluationError("requested evaluation range is empty or invalid")

    model_path, model_provenance = materialize_evaluation_model(
        args.model, getattr(args, "model_revision", None)
    )
    model, tokenizer = _load_model(str(model_path), args.adapter_path)
    completed = _completed_ids(prediction_path)
    started_at = utc_now()
    run_started = time.perf_counter()

    manifest = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": started_at,
        "protocol": protocol_identity(),
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "base_model": model_provenance,
        "dataset": dataset_provenance,
        "prediction_path": str(prediction_path),
    }
    write_json(manifest_path, manifest)

    for index in range(start, stop):
        example_id = f"{spec.key}-{spec.target_split}-{index:06d}"
        if example_id in completed:
            continue
        row = target[index]
        target_question = _row_text(row, spec.question_column)
        semantic_prompt = build_prompt(
            target_question,
            shots,
            question_column=spec.question_column,
            answer_column=spec.answer_column,
        )
        model_prompt = render_model_prompt(
            tokenizer,
            semantic_prompt,
            enable_thinking=args.enable_thinking,
        )
        reference = normalize_reference_answer(row.get(spec.answer_column))
        if reference is None:
            raise EvaluationError(f"invalid reference answer for {example_id}")

        samples: list[dict[str, Any]] = []
        sample_count = 1 if args.mode == "greedy" else args.samples
        for sample_index in range(sample_count):
            seed = derive_seed(
                args.seed,
                args.experiment_id,
                spec.key,
                index,
                sample_index,
            )
            temperature = 0.0 if args.mode == "greedy" else args.temperature
            response, duration = _generate_one(
                model,
                tokenizer,
                model_prompt,
                seed=seed,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            samples.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "response": response,
                    "extracted_answer": extract_marked_number(response),
                    "generation_seconds": duration,
                }
            )

        greedy_tie_break: dict[str, Any] | None = None
        answers = [sample["extracted_answer"] for sample in samples]
        prediction, tied = majority_vote(answers)
        if args.mode == "self_consistency" and tied:
            greedy_seed = derive_seed(
                args.seed,
                args.experiment_id,
                spec.key,
                index,
                args.samples,
            )
            greedy_response, greedy_duration = _generate_one(
                model,
                tokenizer,
                model_prompt,
                seed=greedy_seed,
                max_tokens=args.max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            greedy_answer = extract_marked_number(greedy_response)
            prediction, _ = majority_vote(answers, greedy_answer=greedy_answer)
            greedy_tie_break = {
                "seed": greedy_seed,
                "response": greedy_response,
                "extracted_answer": greedy_answer,
                "generation_seconds": greedy_duration,
            }

        append_jsonl(
            prediction_path,
            {
                "schema_version": "1.0",
                "experiment_id": args.experiment_id,
                "model": args.model,
                "model_revision": (
                    model_provenance["resolved_revision"]
                    or model_provenance["content_sha256"]
                ),
                "adapter": configuration["adapter"],
                "benchmark": spec.key,
                "example_id": example_id,
                "example_index": index,
                "semantic_prompt": semantic_prompt,
                "model_prompt": model_prompt,
                "reference_answer": reference,
                "samples": samples,
                "vote_tied": tied,
                "greedy_tie_break": greedy_tie_break,
                "predicted_answer": prediction,
                "valid_answer": prediction is not None,
                "correct": answers_equal(prediction, reference),
            },
        )

    rows = list(read_jsonl(prediction_path))
    in_range = [row for row in rows if start <= int(row["example_index"]) < stop]
    elapsed = time.perf_counter() - run_started
    correct = sum(bool(row.get("correct")) for row in in_range)
    valid = sum(bool(row.get("valid_answer")) for row in in_range)
    complete = len(in_range) == stop - start
    manifest.update(
        {
            "status": "completed" if complete else "incomplete",
            "completed_at": utc_now(),
            "range": {"start": start, "stop": stop, "expected_rows": stop - start},
            "observed_rows": len(in_range),
            "valid_answers": valid,
            "correct_answers": correct,
            "accuracy": correct / len(in_range) if in_range else math.nan,
            "elapsed_seconds_this_invocation": elapsed,
            "prediction_sha256": file_sha256(prediction_path),
        }
    )
    write_json(manifest_path, manifest)
    if not complete:
        raise EvaluationError("evaluation ended without the expected number of rows")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision",
        help=(
            "40-character Hugging Face commit SHA; required for remote models "
            "outside the frozen registry"
        ),
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--mode", choices=("greedy", "self_consistency"), default="greedy"
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one declared evaluation condition."""
    args = _build_parser().parse_args(argv)
    manifest = run_evaluation(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

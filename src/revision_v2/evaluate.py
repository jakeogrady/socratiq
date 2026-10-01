"""Run or resume one revision-v2 evaluation (condition x benchmark x prompt x mode).

All decoding settings come from configs/revision_v2/protocol.yaml. Every
response is saved with its finish_reason and token counts and is scored under
the primary rule and both sensitivity rules (src.revision_v2.scoring).

--dry-run loads only the tokenizer and writes a placeholder response per
sample, so the whole pipeline (data, prompts, chat template, scoring,
manifest) can be checked without loading weights or generating text.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.evaluate_v2 import (
    BENCHMARKS,
    build_prompt,
    derive_seed,
    load_benchmark,
    normalize_reference_answer,
    render_model_prompt,
)
from src.rerun_utils import (
    append_jsonl,
    file_sha256,
    object_sha256,
    read_jsonl,
    utc_now,
    write_json,
)
from src.revision_v2 import scoring
from src.revision_v2.llama_base import FIXED_DATE, verify_checkpoint
from src.revision_v2.protocol import (
    load_protocol,
    machine_label,
    protocol_identity,
    require_v2_output,
    tracked_git_state,
)
from src.run_training import collect_environment

TOKENIZER_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]
DRY_RUN_RESPONSE = "DRY RUN: no model was loaded and no text was generated.\n#### 0"


class EvaluationError(RuntimeError):
    """Raised when an evaluation would leave the protocol or is unsafe to resume."""


def evaluation_plan(benchmark: str, prompt: str, mode: str) -> dict[str, Any]:
    """Validate the combination against the protocol and return decoding settings."""
    protocol = load_protocol()["evaluation"]
    if benchmark not in protocol["prompts"][prompt]["benchmarks"]:
        msg = f"prompt {prompt} is not defined for {benchmark}"
        raise EvaluationError(msg)
    if mode == "sc5":
        sc = protocol["self_consistency"]
        if prompt != sc["prompt"] or benchmark not in sc["benchmarks"]:
            msg = f"SC@5 is defined only for {sc['prompt']} on {sc['benchmarks']}"
            raise EvaluationError(msg)
        decoding = {
            "samples": sc["samples"],
            "temperature": sc["temperature"],
            "top_p": sc["top_p"],
            "top_k": sc["top_k"],
        }
    elif mode == "greedy":
        decoding = dict(protocol["greedy"])
    else:
        msg = f"unknown mode {mode}"
        raise EvaluationError(msg)
    spec = BENCHMARKS[benchmark]
    if spec.revision != protocol["benchmark_revisions"][benchmark]:
        msg = f"{benchmark} revision differs from the protocol"
        raise EvaluationError(msg)
    if prompt == "P0":
        spec = dataclasses.replace(spec, few_shot_indices=())
    elif len(spec.few_shot_indices) != protocol["prompts"][prompt]["shots"]:
        msg = f"{prompt} shot count differs from the benchmark specification"
        raise EvaluationError(msg)
    return {
        "spec": spec,
        "decoding": decoding,
        "max_tokens": protocol["max_new_tokens"],
        "base_seed": protocol["base_seed"],
        "enable_thinking": protocol["qwen_enable_thinking"],
    }


def model_registry() -> dict[str, dict[str, Any]]:
    """Return protocol model entries keyed by identifier or local path."""
    registry = {}
    for key, model in load_protocol()["models"].items():
        registry[model.get("local_path", model["identifier"])] = {"key": key, **model}
    return registry


def materialize_model(model: str, *, dry_run: bool) -> tuple[Path, dict[str, Any]]:
    """Verify the base model against the protocol and return its local path."""
    entry = model_registry().get(model)
    if entry is None:
        msg = f"{model} is not a protocol model"
        raise EvaluationError(msg)
    if "local_path" in entry:
        check = verify_checkpoint(Path(entry["local_path"]))
        if (
            check["status"] != "passed"
            or check["directory_sha256"] != entry["directory_sha256"]
        ):
            msg = f"local checkpoint failed verification: {check['problems']}"
            raise EvaluationError(msg)
        return Path(entry["local_path"]), {
            "identifier": entry["identifier"],
            "key": entry["key"],
            "local_path": entry["local_path"],
            "directory_sha256": check["directory_sha256"],
        }
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(model, revision=entry["revision"]).sha
    if resolved != entry["revision"]:
        msg = (
            f"model revision changed: expected {entry['revision']}, resolved {resolved}"
        )
        raise EvaluationError(msg)
    path = Path(
        snapshot_download(
            repo_id=model,
            revision=entry["revision"],
            allow_patterns=TOKENIZER_PATTERNS if dry_run else None,
        )
    )
    return path, {
        "identifier": model,
        "key": entry["key"],
        "requested_revision": entry["revision"],
        "resolved_revision": resolved,
        "materialized_path": str(path),
    }


def adapter_identity(
    adapter_path: Path | None, expected_sha256: str | None
) -> dict[str, Any] | None:
    """Verify the final adapter weights against the training manifest's hash."""
    if adapter_path is None:
        if expected_sha256:
            msg = "--adapter-sha256 given without --adapter-path"
            raise EvaluationError(msg)
        return None
    weights = adapter_path / "adapters.safetensors"
    if not weights.is_file():
        msg = f"adapter weights missing: {weights}"
        raise EvaluationError(msg)
    actual = file_sha256(weights)
    if not expected_sha256 or actual != expected_sha256:
        msg = f"adapter weights hash {actual} does not match expected {expected_sha256}"
        raise EvaluationError(msg)
    config = adapter_path / "adapter_config.json"
    return {
        "path": str(adapter_path),
        "weights_sha256": actual,
        "weights_bytes": weights.stat().st_size,
        "config_sha256": file_sha256(config) if config.is_file() else None,
    }


def load_tokenizer_only(path: Path) -> Any:
    """Load the tokenizer exactly as mlx_lm.load would, without weights."""
    from mlx_lm.utils import load_tokenizer

    config = json.loads((path / "config.json").read_text("utf-8"))
    return load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))


def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seed: int,
    max_tokens: int,
    decoding: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Generate one response and record its stop reason and token counts."""
    started = time.perf_counter()
    if dry_run:
        text = DRY_RUN_RESPONSE
        return {
            "seed": seed,
            "response": text,
            "finish_reason": "dry_run",
            "response_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "generation_seconds": time.perf_counter() - started,
        }
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    if decoding["temperature"] == 0:
        sampler = make_sampler(temp=0.0)
    else:
        sampler = make_sampler(
            temp=decoding["temperature"],
            top_p=decoding["top_p"],
            top_k=decoding["top_k"],
            min_p=0.0,
        )
    text = ""
    last = None
    for last in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    ):
        text += last.text
    if last is None:
        msg = "generation produced no response object"
        raise EvaluationError(msg)
    finish = last.finish_reason
    # mlx-lm counts the EOS token in generation_tokens when it stops on EOS.
    response_tokens = last.generation_tokens - (1 if finish == "stop" else 0)
    return {
        "seed": seed,
        "response": text,
        "finish_reason": finish,
        "response_tokens": response_tokens,
        "prompt_tokens": last.prompt_tokens,
        "generation_seconds": time.perf_counter() - started,
    }


def score_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Attach rule extractions and diagnostics to one generated sample."""
    sample["extracted"] = scoring.extract_all(sample["response"])
    sample["diagnostics"] = scoring.diagnose(
        sample["response"], sample["finish_reason"]
    )
    return sample


def decide(
    samples: list[dict[str, Any]], tie_break: dict[str, Any] | None, reference: str
) -> dict[str, Any]:
    """Apply each rule's vote (a single sample's answer for greedy) and score it."""
    result = {}
    for rule in scoring.RULES:
        answers = [s["extracted"][rule] for s in samples]
        greedy_answer = tie_break["extracted"][rule] if tie_break else None
        predicted, tied = scoring.majority_vote(answers, greedy_answer=greedy_answer)
        result[rule] = {
            "predicted_answer": predicted,
            "valid_answer": predicted is not None,
            "correct": scoring.answers_equal(predicted, reference),
            "vote_tied": tied,
        }
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-rule accuracy and per-response diagnostics for the manifest."""
    responses = [s for r in rows for s in r["samples"]]
    summary: dict[str, Any] = {
        "items": len(rows),
        "responses": len(responses),
        "rules": {},
    }
    for rule in scoring.RULES:
        correct = sum(r["decisions"][rule]["correct"] for r in rows)
        valid = sum(r["decisions"][rule]["valid_answer"] for r in rows)
        summary["rules"][rule] = {
            "correct": correct,
            "valid": valid,
            "invalid": len(rows) - valid,
            "accuracy": correct / len(rows) if rows else None,
        }
    summary["diagnostics_responses"] = {
        key: sum(s["diagnostics"][key] for s in responses)
        for key in ("bare_answer", "limit_hit", "repetition_loop", "question_lines")
    }
    summary["diagnostics_responses"]["invalid_primary"] = sum(
        s["extracted"][scoring.PRIMARY_RULE] is None for s in responses
    )
    summary["finish_reasons"] = dict(Counter(s["finish_reason"] for s in responses))
    tokens = [s["response_tokens"] for s in responses]
    summary["response_tokens"] = {
        "mean": statistics.fmean(tokens) if tokens else None,
        "max": max(tokens) if tokens else None,
    }
    summary["tie_breaks_generated"] = sum(r["tie_break"] is not None for r in rows)
    return summary


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume one evaluation and write predictions and manifest."""
    plan = evaluation_plan(args.benchmark, args.prompt, args.mode)
    spec, decoding = plan["spec"], plan["decoding"]
    run_dir = require_v2_output(args.run_dir)
    model_path, model_provenance = materialize_model(args.model, dry_run=args.dry_run)
    adapter = (
        None
        if args.dry_run and args.adapter_path is None
        else adapter_identity(args.adapter_path, args.adapter_sha256)
    )
    configuration = {
        "evaluator": "src.revision_v2.evaluate",
        "evaluator_code_sha256": file_sha256(Path(__file__)),
        "scorer_version": scoring.SCORER_VERSION,
        "scorer_code_sha256": file_sha256(Path(scoring.__file__)),
        "primary_rule": scoring.PRIMARY_RULE,
        "rules": list(scoring.RULES),
        "protocol_sha256": protocol_identity()["sha256"],
        "condition_id": args.condition_id,
        "model": model_provenance["identifier"],
        "model_revision": model_provenance.get("resolved_revision"),
        "model_directory_sha256": model_provenance.get("directory_sha256"),
        "adapter_weights_sha256": adapter and adapter["weights_sha256"],
        "benchmark": dataclasses.asdict(spec),
        "prompt": args.prompt,
        "mode": args.mode,
        "decoding": decoding,
        "max_tokens": plan["max_tokens"],
        "base_seed": plan["base_seed"],
        "enable_thinking": plan["enable_thinking"],
        "start_index": args.start_index,
        "limit": args.limit,
        "dry_run": args.dry_run,
    }
    configuration_sha256 = object_sha256(configuration)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, prediction_path = (
        run_dir / "manifest.json",
        run_dir / "predictions.jsonl",
    )
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text("utf-8"))
        if previous.get("configuration_sha256") != configuration_sha256:
            msg = "refusing to resume: configuration differs from the existing run"
            raise EvaluationError(msg)
    target, shots, dataset_provenance = load_benchmark(spec)
    start = args.start_index
    stop = len(target) if args.limit is None else min(len(target), start + args.limit)
    if args.dry_run:
        model, tokenizer = None, load_tokenizer_only(model_path)
    else:
        from mlx_lm import load

        model, tokenizer = load(
            str(model_path),
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
        )
    environment = collect_environment()
    environment["machine"] = machine_label()
    environment["git"] = tracked_git_state()
    manifest = {
        "schema_version": "2.0",
        "status": "running",
        "started_at": utc_now(),
        "protocol": protocol_identity(),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "base_model": model_provenance,
        "adapter": adapter,
        "dataset": dataset_provenance,
        "environment": environment,
        "seeds": {
            "base_seed": plan["base_seed"],
            "derivation": "sha256(base_seed, condition_id, benchmark, item_index, sample_index)[:4]",
        },
        "prediction_path": str(prediction_path),
    }
    write_json(manifest_path, manifest)
    completed = (
        {row["example_id"] for row in read_jsonl(prediction_path)}
        if prediction_path.exists()
        else set()
    )
    is_llama = model_provenance["key"].startswith("llama")
    run_started = time.perf_counter()
    for index in range(start, stop):
        example_id = f"{spec.key}-{spec.target_split}-{index:06d}"
        if example_id in completed:
            continue
        row = target[index]
        semantic = build_prompt(
            str(row[spec.question_column]),
            shots,
            question_column=spec.few_shot_question_column or spec.question_column,
            answer_column=spec.few_shot_answer_column or spec.answer_column,
        )
        prompt = render_model_prompt(
            tokenizer, semantic, enable_thinking=plan["enable_thinking"]
        )
        if is_llama and f"Today Date: {FIXED_DATE}\n" not in prompt:
            msg = "Llama prompt does not carry the fixed date; the checkpoint template is wrong"
            raise EvaluationError(msg)
        reference = normalize_reference_answer(row.get(spec.answer_column))
        if reference is None:
            msg = f"invalid reference answer for {example_id}"
            raise EvaluationError(msg)
        samples = []
        for sample_index in range(decoding["samples"]):
            seed = derive_seed(
                plan["base_seed"], args.condition_id, spec.key, index, sample_index
            )
            generated = generate_once(
                model,
                tokenizer,
                prompt,
                seed=seed,
                max_tokens=plan["max_tokens"],
                decoding=decoding,
                dry_run=args.dry_run,
            )
            samples.append(score_sample({"sample_index": sample_index, **generated}))
        tie_break = None
        if args.mode == "sc5" and any(
            scoring.majority_vote([s["extracted"][rule] for s in samples])[1]
            for rule in scoring.RULES
        ):
            seed = derive_seed(
                plan["base_seed"],
                args.condition_id,
                spec.key,
                index,
                decoding["samples"],
            )
            greedy = {"samples": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0}
            tie_break = score_sample(
                generate_once(
                    model,
                    tokenizer,
                    prompt,
                    seed=seed,
                    max_tokens=plan["max_tokens"],
                    decoding=greedy,
                    dry_run=args.dry_run,
                )
            )
        decisions = decide(samples, tie_break, reference)
        primary = decisions[scoring.PRIMARY_RULE]
        append_jsonl(
            prediction_path,
            {
                "schema_version": "2.0",
                "condition_id": args.condition_id,
                "benchmark": spec.key,
                "prompt": args.prompt,
                "mode": args.mode,
                "example_id": example_id,
                "example_index": index,
                "semantic_prompt": semantic,
                "model_prompt": prompt,
                "reference_answer": reference,
                "samples": samples,
                "tie_break": tie_break,
                "decisions": decisions,
                "predicted_answer": primary["predicted_answer"],
                "valid_answer": primary["valid_answer"],
                "correct": primary["correct"],
            },
        )
    rows = [
        r
        for r in read_jsonl(prediction_path)
        if start <= int(r["example_index"]) < stop
    ]
    complete = len(rows) == stop - start
    manifest.update(
        {
            "status": ("dry_run_completed" if args.dry_run else "completed")
            if complete
            else "incomplete",
            "completed_at": utc_now(),
            "range": {"start": start, "stop": stop, "expected_rows": stop - start},
            "observed_rows": len(rows),
            "summary": summarize(rows),
            "elapsed_seconds_this_invocation": time.perf_counter() - run_started,
            "prediction_sha256": file_sha256(prediction_path),
        }
    )
    write_json(manifest_path, manifest)
    if not complete:
        msg = "evaluation ended without the expected number of rows"
        raise EvaluationError(msg)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="protocol model identifier or local checkpoint path",
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--adapter-sha256",
        help="expected SHA-256 of adapters.safetensors from the training manifest",
    )
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--prompt", choices=("P0", "P4"), required=True)
    parser.add_argument("--mode", choices=("greedy", "sc5"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation and print its manifest."""
    manifest = run_evaluation(_parser().parse_args(argv))
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "environment"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

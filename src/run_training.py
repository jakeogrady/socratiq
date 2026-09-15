"""Validate and run MLX-LM LoRA jobs with reviewer-rerun manifests."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import (
    directory_sha256,
    directory_size,
    file_sha256,
    object_sha256,
    protocol_identity,
    utc_now,
    write_json,
)

REQUIRED_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
PAIRED_ALLOWED_DIFFERENCES = frozenset({"data", "adapter_path"})
PAIR_CONFIG_COUNT = 2
TRAINABLE_RE = re.compile(
    r"Trainable parameters:\s*([0-9.]+)%\s*\(([0-9.]+)([KMB]?)/([0-9.]+)([KMB]?)\)"
)
PEAK_MEMORY_RE = re.compile(r"Peak mem\s+([0-9.]+)\s+GB")
VAL_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9.]+)")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class TrainingConfigurationError(ValueError):
    """Raised when a training run would violate the frozen protocol."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping, importing the locked dependency lazily."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read MLX-LM configs; install the locked project environment."
        ) from exc
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"Expected a YAML mapping in {path}")
    return value


def validate_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fields required for a reviewer-rerun LoRA job."""
    errors: list[str] = []
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        errors.append("model must be a non-empty string")
    model_revision = config.get("model_revision")
    if (
        not isinstance(model_revision, str)
        or REVISION_RE.fullmatch(model_revision) is None
    ):
        errors.append("model_revision must be a 40-character lowercase commit SHA")
    if config.get("model_precision") not in {"bfloat16", "4-bit"}:
        errors.append("model_precision must be 'bfloat16' or '4-bit'")
    if config.get("fine_tune_type") != "lora":
        errors.append("fine_tune_type must be 'lora'")
    if config.get("train") is not True:
        errors.append("train must be true")
    if config.get("test") is not False:
        errors.append("test must be explicitly false")
    for field in (
        "seed",
        "iters",
        "save_every",
        "steps_per_report",
        "steps_per_eval",
        "val_batches",
        "batch_size",
        "grad_accumulation_steps",
        "max_seq_length",
        "num_layers",
    ):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{field} must be an integer")
    for field in ("data", "adapter_path", "prompt_feature", "completion_feature"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    learning_rate = config.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        errors.append("learning_rate must be a positive numeric YAML value")
    schedule = config.get("lr_schedule")
    if not isinstance(schedule, Mapping):
        errors.append("lr_schedule must be a mapping")
    else:
        arguments = schedule.get("arguments")
        if not isinstance(arguments, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in arguments
        ):
            errors.append("lr_schedule.arguments must contain only numeric YAML values")

    lora = config.get("lora_parameters")
    if not isinstance(lora, Mapping):
        errors.append("lora_parameters must be a mapping")
        keys: list[Any] = []
    else:
        raw_keys = lora.get("keys")
        keys = raw_keys if isinstance(raw_keys, list) else []
        if tuple(keys) != REQUIRED_TARGETS:
            errors.append(
                "lora_parameters.keys must contain the seven intended targets in frozen order"
            )
        rank = lora.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            errors.append("lora_parameters.rank must be a positive integer")
        scale = lora.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
            errors.append("lora_parameters.scale must be positive")
        dropout = lora.get("dropout")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            errors.append("lora_parameters.dropout must be in [0, 1)")

    if config.get("prompt_feature") != "question":
        errors.append("prompt_feature must be 'question'")
    if config.get("completion_feature") != "answer":
        errors.append("completion_feature must be 'answer'")
    if config.get("mask_prompt") is not False:
        errors.append(
            "mask_prompt must be explicitly false to preserve the selected recipe"
        )
    if errors:
        raise TrainingConfigurationError("; ".join(errors))

    return {
        "status": "passed",
        "model": config["model"],
        "num_layers": config["num_layers"],
        "targets": list(keys),
        "configuration_sha256": object_sha256(dict(config)),
    }


def compare_paired_configs(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a matched pair to differ only in data and adapter paths."""
    validate_training_config(left)
    validate_training_config(right)
    normalized_left = copy.deepcopy(dict(left))
    normalized_right = copy.deepcopy(dict(right))
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(normalized_left) | set(normalized_right)):
        if normalized_left.get(key) != normalized_right.get(key):
            differences[key] = {
                "left": normalized_left.get(key),
                "right": normalized_right.get(key),
            }
    prohibited = sorted(set(differences) - PAIRED_ALLOWED_DIFFERENCES)
    if prohibited:
        raise TrainingConfigurationError(
            f"paired configs differ in prohibited fields: {', '.join(prohibited)}"
        )
    if set(differences) != PAIRED_ALLOWED_DIFFERENCES:
        raise TrainingConfigurationError(
            "paired configs must use distinct data and adapter_path values"
        )
    return {
        "status": "passed",
        "allowed_differences": differences,
        "shared_configuration_sha256": object_sha256(
            {
                key: value
                for key, value in normalized_left.items()
                if key not in PAIRED_ALLOWED_DIFFERENCES
            }
        ),
    }


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def collect_environment() -> dict[str, Any]:
    """Collect non-secret machine, software, Git, and disk metadata."""
    disk = shutil.disk_usage(Path.cwd())
    environment: dict[str, Any] = {
        "collected_at": utc_now(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "disk": {
            "path": str(Path.cwd()),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"]),
            "branch": _command_output(["git", "branch", "--show-current"]),
            "status_porcelain": _command_output(["git", "status", "--porcelain"]),
        },
    }
    if platform.system() == "Darwin":
        hardware: dict[str, Any] = {}
        try:
            completed = subprocess.run(
                ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            profile = json.loads(completed.stdout) if completed.returncode == 0 else {}
            entries = profile.get("SPHardwareDataType", [])
            if entries:
                raw_hardware = entries[0]
                hardware = {
                    "machine_name": raw_hardware.get("machine_name"),
                    "model": raw_hardware.get("machine_model"),
                    "chip": raw_hardware.get("chip_type"),
                    "physical_memory": raw_hardware.get("physical_memory"),
                    "memory_bytes": _parse_memory_size(
                        raw_hardware.get("physical_memory")
                    ),
                    "processors": raw_hardware.get("number_processors"),
                }
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            hardware = {}
        if not hardware:
            hardware = {
                "model": _command_output(["sysctl", "-n", "hw.model"]),
                "chip": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "memory_bytes": _command_output(["sysctl", "-n", "hw.memsize"]),
            }
        environment["hardware"] = hardware
    try:
        from importlib.metadata import PackageNotFoundError, version

        packages: dict[str, str | None] = {}
        for package in (
            "mlx",
            "mlx-lm",
            "transformers",
            "datasets",
            "openai",
            "PyYAML",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        environment["packages"] = packages
    except ImportError:
        environment["packages"] = {}
    return environment


def _parse_memory_size(value: Any) -> int | None:
    """Convert a system-profiler memory label such as '64 GB' to bytes."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]B)\s*", value, re.IGNORECASE)
    if match is None:
        return None
    scale = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }[match.group(2).upper()]
    return round(float(match.group(1)) * scale)


def dataset_identity(data_path: Path) -> dict[str, Any]:
    """Hash the expected MLX-LM train and validation files."""
    result: dict[str, Any] = {"path": str(data_path), "exists": data_path.exists()}
    files: dict[str, Any] = {}
    for split in ("train", "valid"):
        path = data_path / f"{split}.jsonl"
        files[split] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": file_sha256(path) if path.exists() else None,
        }
    result["files"] = files
    return result


def resolve_model_revision(
    model_name: str, requested_revision: str | None = None
) -> str | None:
    """Resolve a requested Hugging Face revision to its immutable SHA."""
    path = Path(model_name)
    if path.exists():
        return directory_sha256(path)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    return HfApi().model_info(model_name, revision=requested_revision).sha


def materialize_model_snapshot(model_name: str, revision: str) -> Path:
    """Download and return one exact Hugging Face model snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to pin model weights") from exc
    return Path(snapshot_download(repo_id=model_name, revision=revision))


def preflight_model_targets(
    config: Mapping[str, Any], *, model_path: str | Path | None = None
) -> dict[str, Any]:
    """Load a model and prove all configured target suffixes exist."""
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError("mlx-lm is required for model target preflight") from exc
    model, _ = load(str(model_path or config["model"]))
    module_names = [name for name, _ in model.named_modules()]
    target_counts = {
        target: sum(name.endswith(target) for name in module_names)
        for target in config["lora_parameters"]["keys"]
    }
    missing = [target for target, count in target_counts.items() if count == 0]
    insufficient = [
        target
        for target, count in target_counts.items()
        if count < int(config["num_layers"])
    ]
    if missing or insufficient:
        raise TrainingConfigurationError(
            f"model target preflight failed; missing={missing}, insufficient={insufficient}"
        )
    return {
        "status": "passed",
        "named_module_count": len(module_names),
        "target_counts": target_counts,
    }


def parse_training_log(text: str) -> dict[str, Any]:
    """Extract resource and validation metrics from MLX-LM text output."""
    trainable = TRAINABLE_RE.search(text)
    peak_values = [float(value) for value in PEAK_MEMORY_RE.findall(text)]
    validation = [
        {"iteration": int(iteration), "loss": float(loss)}
        for iteration, loss in VAL_LOSS_RE.findall(text)
    ]
    metrics: dict[str, Any] = {
        "peak_mlx_memory_gb": max(peak_values) if peak_values else None,
        "validation": validation,
    }
    if trainable:
        metrics["trainable_parameters"] = {
            "percent": float(trainable.group(1)),
            "trainable_display": f"{trainable.group(2)}{trainable.group(3)}",
            "total_display": f"{trainable.group(4)}{trainable.group(5)}",
            "trainable_count": _scaled_number(trainable.group(2), trainable.group(3)),
            "total_count": _scaled_number(trainable.group(4), trainable.group(5)),
        }
    else:
        metrics["trainable_parameters"] = None
    if validation:
        best = min(validation, key=lambda item: (item["loss"], item["iteration"]))
        metrics["best_validation"] = best
    else:
        metrics["best_validation"] = None
    return metrics


def _scaled_number(value: str, suffix: str) -> int:
    scale = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return round(float(value) * scale)


def _peak_process_bytes(usage: resource.struct_rusage) -> int:
    # macOS reports bytes; Linux reports KiB.
    value = int(usage.ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_training_job(
    config_path: Path,
    *,
    experiment_id: str,
    run_dir: Path | None,
    dry_run: bool,
    smoke_iters: int | None,
    minimum_free_gib: float,
    skip_model_preflight: bool,
    skip_revision_resolution: bool,
) -> dict[str, Any]:
    """Validate, manifest, and optionally invoke one MLX-LM training job."""
    config = load_yaml(config_path)
    validation = validate_training_config(config)
    configured_adapter = Path(config["adapter_path"])
    actual_run_dir = run_dir or configured_adapter.parent
    actual_adapter = configured_adapter
    if smoke_iters is not None:
        if smoke_iters <= 0:
            raise ValueError("smoke_iters must be greater than zero")
        actual_adapter = actual_run_dir / "smoke_adapter"

    environment = collect_environment()
    free_gib = environment["disk"]["free_bytes"] / (1024**3)
    if not dry_run and free_gib < minimum_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free; training gate requires {minimum_free_gib:.1f} GiB"
        )

    target_preflight: dict[str, Any] | None = None
    expected_revision = config["model_revision"]
    revision: str | None = expected_revision if dry_run else None
    model_path: str | Path = config["model"]
    if not dry_run:
        if not skip_revision_resolution:
            revision = resolve_model_revision(config["model"], expected_revision)
            if revision != expected_revision:
                raise RuntimeError(
                    "model revision changed: "
                    f"expected {expected_revision}, resolved {revision}"
                )
            model_path = materialize_model_snapshot(config["model"], expected_revision)
        if not skip_model_preflight:
            target_preflight = preflight_model_targets(config, model_path=model_path)

    local_executable = Path(sys.executable).with_name("mlx_lm.lora")
    executable = shutil.which("mlx_lm.lora")
    if executable is None and local_executable.is_file():
        executable = str(local_executable)
    command = [executable or "mlx_lm.lora", "--config", str(config_path)]
    if not dry_run and not skip_revision_resolution:
        command.extend(["--model", str(model_path)])
    if smoke_iters is not None:
        command.extend(
            ["--iters", str(smoke_iters), "--adapter-path", str(actual_adapter)]
        )
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": protocol_identity(),
        "experiment_id": experiment_id,
        "status": "dry_run" if dry_run else "preparing",
        "created_at": utc_now(),
        "configuration_path": str(config_path),
        "configuration_file_sha256": file_sha256(config_path),
        "configuration": config,
        "configuration_validation": validation,
        "model": {
            "identifier": config["model"],
            "expected_revision": expected_revision,
            "resolved_revision": revision,
            "materialized_path": str(model_path) if not dry_run else None,
            "target_preflight": target_preflight,
        },
        "dataset": dataset_identity(Path(config["data"])),
        "environment": environment,
        "command": command,
        "resolved_executable": executable,
        "run_dir": str(actual_run_dir),
        "adapter_path": str(actual_adapter),
        "smoke_iters": smoke_iters,
        "minimum_free_gib": minimum_free_gib,
    }
    if dry_run:
        return manifest
    if executable is None:
        raise RuntimeError("mlx_lm.lora executable is not available")
    if not all(details["exists"] for details in manifest["dataset"]["files"].values()):
        raise RuntimeError(
            "training dataset is incomplete; train.jsonl and valid.jsonl are required"
        )
    if actual_run_dir.exists() and any(actual_run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {actual_run_dir}")

    actual_run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, actual_run_dir / "config.yaml")
    manifest_path = actual_run_dir / "manifest.json"
    log_path = actual_run_dir / "train.log"
    manifest["status"] = "running"
    manifest["started_at"] = utc_now()
    write_json(manifest_path, manifest)

    started = time.perf_counter()
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("training process did not expose its output stream")
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    log_text = log_path.read_text(encoding="utf-8")
    metrics = parse_training_log(log_text)

    final_weights = actual_adapter / "adapters.safetensors"
    adapter_manifest = {
        "path": str(actual_adapter),
        "exists": actual_adapter.exists(),
        "bytes": directory_size(actual_adapter),
        "sha256": directory_sha256(actual_adapter),
        "selected_checkpoint_policy": "final_adapter_after_last_iteration",
        "selected_weights_path": str(final_weights),
        "selected_weights_exists": final_weights.is_file(),
        "selected_weights_bytes": (
            final_weights.stat().st_size if final_weights.is_file() else None
        ),
        "selected_weights_sha256": (
            file_sha256(final_weights) if final_weights.is_file() else None
        ),
    }
    completed = return_code == 0 and final_weights.is_file()
    manifest.update(
        {
            "status": "completed" if completed else "failed",
            "completed_at": utc_now(),
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "peak_process_bytes": max(
                _peak_process_bytes(before_usage), _peak_process_bytes(after_usage)
            ),
            "training_metrics": metrics,
            "train_log": {
                "path": str(log_path),
                "bytes": log_path.stat().st_size,
                "sha256": file_sha256(log_path),
            },
            "adapter": adapter_manifest,
        }
    )
    write_json(manifest_path, manifest)
    write_json(actual_run_dir / "adapter_manifest.json", adapter_manifest)
    if return_code != 0:
        raise RuntimeError(f"MLX-LM training failed with exit code {return_code}")
    if not final_weights.is_file():
        raise RuntimeError(
            "MLX-LM exited successfully but final adapter weights are missing"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate training YAML files")
    validate.add_argument("configs", nargs="+", type=Path)
    validate.add_argument(
        "--pair", action="store_true", help="require exactly two matched configs"
    )

    environment = commands.add_parser("environment", help="print environment metadata")
    environment.add_argument("--output", type=Path)

    run = commands.add_parser("run", help="manifest and run one MLX-LM job")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--smoke-iters", type=int)
    run.add_argument("--minimum-free-gib", type=float, default=40.0)
    run.add_argument("--skip-model-preflight", action="store_true")
    run.add_argument("--skip-revision-resolution", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configs, report the environment, or run training."""
    args = _build_parser().parse_args(argv)
    if args.command == "validate":
        configs = [load_yaml(path) for path in args.configs]
        result: dict[str, Any] = {
            "configs": [
                {
                    "path": str(path),
                    "validation": validate_training_config(config),
                }
                for path, config in zip(args.configs, configs, strict=True)
            ]
        }
        if args.pair:
            if len(configs) != PAIR_CONFIG_COUNT:
                raise TrainingConfigurationError("--pair requires exactly two configs")
            result["pair"] = compare_paired_configs(configs[0], configs[1])
    elif args.command == "environment":
        result = collect_environment()
        if args.output:
            write_json(args.output, result)
    else:
        result = run_training_job(
            args.config,
            experiment_id=args.experiment_id,
            run_dir=args.run_dir,
            dry_run=args.dry_run,
            smoke_iters=args.smoke_iters,
            minimum_free_gib=args.minimum_free_gib,
            skip_model_preflight=args.skip_model_preflight,
            skip_revision_resolution=args.skip_revision_resolution,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

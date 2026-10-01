"""Validate, run and check one revision-v2 LoRA training job.

`run` refuses to start unless the configuration is byte-identical to what the
protocol generates, the base model matches its pinned revision or checkpoint
hash, and every data file matches the tracked dataset manifest. After MLX-LM
finishes it checks every logged learning rate against the schedule replay and
records the result in manifest.json. `pilot-check` applies the protocol's
pilot acceptance rule to a finished run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import (
    directory_sha256,
    directory_size,
    file_sha256,
    object_sha256,
    utc_now,
    write_json,
)
from src.revision_v2 import configs as config_module
from src.revision_v2.data import MANIFEST_PATH as DATASET_MANIFEST
from src.revision_v2.llama_base import verify_checkpoint
from src.revision_v2.protocol import (
    REPOSITORY_ROOT,
    RUNS_ROOT,
    load_protocol,
    machine_label,
    protocol_identity,
    require_v2_output,
    tracked_git_state,
)
from src.run_training import (
    REQUIRED_TARGETS,
    _peak_process_bytes,
    collect_environment,
    materialize_model_snapshot,
    parse_training_log,
    preflight_model_targets,
    resolve_model_revision,
)

LR_LINE = re.compile(
    r"Iter (\d+): Train loss [0-9.]+, Learning Rate ([0-9.]+e[-+]\d+),"
)
VAL_LINE = re.compile(r"Iter (\d+): Val loss ([0-9.]+|nan|inf)", re.IGNORECASE)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")


class TrainingError(RuntimeError):
    """Raised when a run would leave the protocol or failed its checks."""


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML training configuration."""
    import yaml

    value = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(value, dict):
        msg = f"expected a mapping in {path}"
        raise TrainingError(msg)
    return value


def find_run(config_path: Path) -> dict[str, Any]:
    """Return the protocol's run entry for a configuration path."""
    for run in config_module.training_runs():
        if Path(run["config_path"]) == config_path:
            return run
    msg = f"{config_path} is not a protocol-defined revision-v2 configuration"
    raise TrainingError(msg)


def validate_config(config_path: Path) -> dict[str, Any]:
    """Check the configuration against the protocol without touching models."""
    run = find_run(config_path)
    expected_text = config_module.render_config(run)
    if config_path.read_text("utf-8") != expected_text:
        msg = f"{config_path} differs from the protocol-generated text; do not edit configs"
        raise TrainingError(msg)
    config = load_config(config_path)
    protocol = load_protocol()
    errors = []
    if config["iters"] % config["grad_accumulation_steps"]:
        errors.append("iters is not a multiple of grad_accumulation_steps")
    if (
        config["iters"] // config["grad_accumulation_steps"]
        != protocol["training"]["optimizer_updates"]
    ):
        errors.append("optimizer update count differs from the protocol")
    if tuple(config["lora_parameters"]["keys"]) != REQUIRED_TARGETS:
        errors.append("LoRA target keys differ from the frozen list")
    if "model_revision" in config and not SHA40.fullmatch(config["model_revision"]):
        errors.append("model_revision must be a 40-character commit SHA")
    if "model_directory_sha256" in config and not SHA64.fullmatch(
        config["model_directory_sha256"]
    ):
        errors.append("model_directory_sha256 must be a 64-character SHA-256")
    if not Path(config["adapter_path"]).is_relative_to(RUNS_ROOT / "training"):
        errors.append("adapter_path must be under runs/revision_v2/training")
    replay = config_module.replay_schedule(config)
    if (
        replay["updates"] != protocol["training"]["optimizer_updates"]
        or replay["discarded_microbatches"]
    ):
        errors.append("schedule replay does not give the protocol's update count")
    if errors:
        raise TrainingError("; ".join(errors))
    return {"status": "passed", "run": run, "config": config, "replay": replay}


def verify_data(config: dict[str, Any]) -> dict[str, Any]:
    """Check the run's train and valid files against the tracked dataset manifest."""
    tracked = json.loads(DATASET_MANIFEST.read_text("utf-8"))["files"]
    result: dict[str, Any] = {}
    for split in ("train", "valid"):
        path = Path(config["data"]) / f"{split}.jsonl"
        expected = tracked.get(path.as_posix())
        actual = file_sha256(path) if path.is_file() else None
        result[split] = {
            "path": path.as_posix(),
            "sha256": actual,
            "expected_sha256": expected and expected["sha256"],
            "rows": expected and expected["rows"],
        }
        if expected is None or actual != expected["sha256"]:
            msg = f"data file missing or hash mismatch: {path}; rebuild with `python -m src.revision_v2.data build`"
            raise TrainingError(msg)
    return result


def resolve_model(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Verify and materialise the base model; returns provenance and the path to load."""
    if "model_directory_sha256" in config:
        local = Path(config["model"])
        check = verify_checkpoint(local)
        if (
            check["status"] != "passed"
            or check["directory_sha256"] != config["model_directory_sha256"]
        ):
            msg = f"Llama base checkpoint failed verification: {check['problems']}"
            raise TrainingError(msg)
        return {
            "identifier": config["model"],
            "kind": "local",
            "directory_sha256": check["directory_sha256"],
            "load_path": str(local),
        }
    if dry_run:
        return {
            "identifier": config["model"],
            "kind": "huggingface",
            "expected_revision": config["model_revision"],
            "load_path": None,
        }
    resolved = resolve_model_revision(config["model"], config["model_revision"])
    if resolved != config["model_revision"]:
        msg = f"model revision changed: expected {config['model_revision']}, resolved {resolved}"
        raise TrainingError(msg)
    path = materialize_model_snapshot(config["model"], config["model_revision"])
    return {
        "identifier": config["model"],
        "kind": "huggingface",
        "expected_revision": config["model_revision"],
        "resolved_revision": resolved,
        "load_path": str(path),
    }


def check_logged_learning_rates(
    log_text: str, replay: dict[str, Any]
) -> dict[str, Any]:
    """Compare every logged learning rate with the replayed value at 4 significant digits."""
    logged = {int(it): value for it, value in LR_LINE.findall(log_text)}
    expected = {
        int(it): f"{lr:.3e}" for it, lr in replay["logged_lr_by_iteration"].items()
    }
    mismatches = sorted(it for it in expected if logged.get(it) != expected[it])
    update_16_iteration = (
        16 * 32 + 8
    )  # first report after update 16 (iteration 512) is iteration 520
    return {
        "status": "passed" if not mismatches and logged else "failed",
        "reports_checked": len(expected),
        "mismatched_iterations": mismatches[:20],
        "logged_at_update_16": logged.get(update_16_iteration),
        "expected_at_update_16": expected.get(update_16_iteration),
        "logged_at_update_188": logged.get(6016),
        "expected_at_update_188": expected.get(6016),
    }


def validation_losses(log_text: str) -> list[dict[str, float]]:
    """Return every validation loss in the order logged."""
    return [
        {"iteration": int(it), "loss": float(loss)}
        for it, loss in VAL_LINE.findall(log_text)
    ]


def run(
    config_path: Path, *, dry_run: bool, minimum_free_gib: float = 40.0
) -> dict[str, Any]:
    """Run one training job end to end and write its manifest."""
    validated = validate_config(config_path)
    config, run_entry, replay = (
        validated["config"],
        validated["run"],
        validated["replay"],
    )
    run_dir = require_v2_output(Path(run_entry["run_dir"]))
    data = verify_data(config)
    model = resolve_model(config, dry_run=dry_run)
    environment = collect_environment()
    environment["machine"] = machine_label()
    environment["git"] = tracked_git_state()
    free_gib = environment["disk"]["free_bytes"] / 1024**3
    if not dry_run and free_gib < minimum_free_gib:
        msg = f"only {free_gib:.1f} GiB free; {minimum_free_gib} GiB required"
        raise TrainingError(msg)
    executable = shutil.which("mlx_lm.lora") or str(
        Path(sys.executable).with_name("mlx_lm.lora")
    )
    command = [executable, "--config", str(config_path)]
    if model["kind"] == "huggingface" and model["load_path"]:
        command += ["--model", model["load_path"]]
    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "protocol": protocol_identity(),
        "run_id": run_entry["run_id"],
        "tier": run_entry["tier"],
        "arm": run_entry["arm"],
        "peak_variant": run_entry["peak_variant"],
        "seed": config["seed"],
        "status": "dry_run" if dry_run else "preparing",
        "created_at": utc_now(),
        "configuration_path": str(config_path),
        "configuration_file_sha256": file_sha256(config_path),
        "configuration": config,
        "configuration_sha256": object_sha256(config),
        "schedule_expectation": {
            "optimizer_updates": replay["updates"],
            "discarded_microbatches": replay["discarded_microbatches"],
            "peak_lr": replay["peak_lr"],
            "peak_updates": replay["peak_updates"],
            "lr_update_16": replay["lr_by_update"][15],
            "lr_update_188": replay["lr_by_update"][-1],
        },
        "model": model,
        "dataset": data,
        "environment": environment,
        "command": command,
        "run_dir": str(run_dir),
    }
    if dry_run:
        return manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        msg = f"run directory is not empty: {run_dir}"
        raise TrainingError(msg)
    if model["kind"] == "huggingface":
        manifest["model"]["target_preflight"] = preflight_model_targets(
            config, model_path=model["load_path"]
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    log_path = run_dir / "train.log"
    manifest["status"] = "running"
    manifest["started_at"] = utc_now()
    write_json(run_dir / "manifest.json", manifest)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=REPOSITORY_ROOT,
        )
        assert process.stdout is not None  # noqa: S101 - Popen with PIPE always sets stdout
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    log_text = log_path.read_text("utf-8")
    adapter = Path(config["adapter_path"])
    final_weights = adapter / "adapters.safetensors"
    lr_check = check_logged_learning_rates(log_text, replay)
    completed = (
        return_code == 0 and final_weights.is_file() and lr_check["status"] == "passed"
    )
    manifest.update(
        {
            "status": "completed" if completed else "failed",
            "completed_at": utc_now(),
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "peak_process_bytes": _peak_process_bytes(
                resource.getrusage(resource.RUSAGE_CHILDREN)
            ),
            "training_metrics": parse_training_log(log_text),
            "validation_losses": validation_losses(log_text),
            "post_run_checks": {"learning_rate": lr_check},
            "train_log": {
                "path": str(log_path),
                "bytes": log_path.stat().st_size,
                "sha256": file_sha256(log_path),
            },
            "adapter": {
                "path": str(adapter),
                "bytes": directory_size(adapter),
                "directory_sha256": directory_sha256(adapter),
                "selected_checkpoint_policy": "final_adapter_after_last_iteration",
                "selected_weights_path": str(final_weights),
                "selected_weights_bytes": final_weights.stat().st_size
                if final_weights.is_file()
                else None,
                "selected_weights_sha256": file_sha256(final_weights)
                if final_weights.is_file()
                else None,
                "adapter_config_sha256": file_sha256(adapter / "adapter_config.json")
                if (adapter / "adapter_config.json").is_file()
                else None,
            },
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    if not completed:
        msg = f"training failed or failed its checks; see {log_path} and {run_dir / 'manifest.json'}"
        raise TrainingError(msg)
    return manifest


def pilot_check(run_dir: Path) -> dict[str, Any]:
    """Apply the protocol's pilot acceptance rule to a finished run."""
    protocol = load_protocol()
    rule = protocol["training"]["pilot"]["accept"]
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    losses = manifest.get("validation_losses") or []
    values = [item["loss"] for item in losses]
    finite = bool(values) and all(math.isfinite(v) for v in values)
    criteria = {
        "run_completed": manifest.get("status") == "completed",
        "all_validation_losses_finite": finite,
        "final_below_initial": finite and values[-1] < values[0],
        "final_at_most_v1_reference": finite and values[-1] <= rule["final_at_most"],
    }
    result = {
        "run_id": manifest.get("run_id"),
        "peak_variant": manifest.get("peak_variant"),
        "initial_validation_loss": values[0] if values else None,
        "final_validation_loss": values[-1] if values else None,
        "final_iteration": losses[-1]["iteration"] if losses else None,
        "v1_reference_final_validation_loss": rule["final_at_most"],
        "criteria": criteria,
        "decision": "accept" if all(criteria.values()) else "reject",
        "checked_at": utc_now(),
        "note": "Decision on validation loss only. The PI approves before the freeze tag.",
    }
    write_json(run_dir / "pilot_check.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("configs", nargs="+", type=Path)
    run_cmd = commands.add_parser("run")
    run_cmd.add_argument("--config", type=Path, required=True)
    run_cmd.add_argument("--dry-run", action="store_true")
    run_cmd.add_argument("--minimum-free-gib", type=float, default=40.0)
    pilot = commands.add_parser("pilot-check")
    pilot.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print JSON."""
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result: dict[str, Any] = {}
        for path in args.configs:
            checked = validate_config(path)
            result[str(path)] = {
                "status": checked["status"],
                "updates": checked["replay"]["updates"],
                "peak_updates": checked["replay"]["peak_updates"],
            }
    elif args.command == "run":
        result = run(
            args.config, dry_run=args.dry_run, minimum_free_gib=args.minimum_free_gib
        )
    else:
        result = pilot_check(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

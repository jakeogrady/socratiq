"""Generate the revision-v2 training configurations and replay their schedules.

Every configuration is derived from configs/revision_v2/protocol.yaml so the
files cannot drift from the protocol. `generate --check` fails if any file on
disk differs from what the protocol implies.

The schedule replay uses mlx-lm's own build_schedule and AdamW and calls
apply_gradients only when iteration % grad_accumulation_steps == 0, exactly as
mlx_lm/tuner/trainer.py does, so it predicts every learning rate the trainer
will log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import write_json
from src.revision_v2.protocol import RUNS_ROOT, load_protocol

CONFIG_DIR = Path("configs/revision_v2/training")


def fmt(value: float) -> str:
    """Format a float so PyYAML reads it back as a float."""
    return f"{value:.1e}"


def peak_label(value: float) -> str:
    """Return the label used in run identifiers, e.g. 8e-5."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def training_runs(protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """List every training run the protocol defines, including halved-peak variants."""
    protocol = protocol or load_protocol()
    runs = []
    for tier in ("core", "second_tier"):
        for entry in protocol["matrix"][tier]:
            model_key = entry["model"]
            model = protocol["models"][model_key]
            schedule = protocol["training"]["schedules"][model["schedule"]]
            for variant, peak in schedule["peaks"].items():
                for arm in entry["arms"]:
                    data_suffix = (
                        ""
                        if entry["data"] == "full"
                        else f"_{entry['data'].removeprefix('subset_')}"
                    )
                    run_id = f"{model_key}_{arm}{data_suffix}_lr{peak_label(peak)}"
                    runs.append(
                        {
                            "run_id": run_id,
                            "pair_id": f"{model_key}{data_suffix}_lr{peak_label(peak)}",
                            "tier": tier,
                            "model_key": model_key,
                            "arm": arm,
                            "data": entry["data"],
                            "peak_variant": variant,
                            "peak": peak,
                            "config_path": (CONFIG_DIR / f"{run_id}.yaml").as_posix(),
                            "run_dir": (RUNS_ROOT / "training" / run_id).as_posix(),
                            "adapter_path": (
                                RUNS_ROOT / "training" / run_id / "adapter"
                            ).as_posix(),
                        }
                    )
    return runs


def render_config(run: dict[str, Any], protocol: dict[str, Any] | None = None) -> str:
    """Return the exact YAML text of one training configuration."""
    protocol = protocol or load_protocol()
    training = protocol["training"]
    model = protocol["models"][run["model_key"]]
    schedule = training["schedules"][model["schedule"]]
    peak = run["peak"]
    lines = [
        "# Generated from configs/revision_v2/protocol.yaml by src.revision_v2.configs. Do not edit.",
        f'run_id: "{run["run_id"]}"',
    ]
    if "local_path" in model:
        lines += [
            f'model: "{model["local_path"]}"',
            f'model_directory_sha256: "{model["directory_sha256"]}"',
        ]
    else:
        lines += [
            f'model: "{model["identifier"]}"',
            f'model_revision: "{model["revision"]}"',
        ]
    lines += [
        f'model_precision: "{model["precision"]}"',
        "train: true",
        "fine_tune_type: lora",
        f"optimizer: {training['optimizer']}",
        "optimizer_config:",
        "  adamw:",
        f"    betas: [{training['adamw']['betas'][0]}, {training['adamw']['betas'][1]}]",
        f"    eps: {fmt(training['adamw']['eps'])}",
        f"    weight_decay: {training['adamw']['weight_decay']}",
        f"learning_rate: {fmt(peak)}",
        "lr_schedule:",
        "  name: cosine_decay",
        f"  arguments: [{fmt(peak)}, {schedule['decay_steps']}, {fmt(schedule['end'])}]",
    ]
    if schedule["warmup"]:
        lines += [
            f"  warmup: {schedule['warmup']}",
            f"  warmup_init: {fmt(peak * schedule['warmup_init_fraction'])}",
        ]
    lines += [
        f"iters: {training['iters']}",
        f"seed: {training['seed']}",
        f"save_every: {training['save_every']}",
        f"steps_per_report: {training['steps_per_report']}",
        f"steps_per_eval: {training['steps_per_eval']}",
        f"val_batches: {training['val_batches']}",
        f"batch_size: {training['batch_size']}",
        f"grad_accumulation_steps: {training['grad_accumulation_steps']}",
        f"max_seq_length: {training['max_seq_length']}",
        f"num_layers: {model['num_layers']}",
        f"grad_checkpoint: {str(training['grad_checkpoint']).lower()}",
        f"mask_prompt: {str(training['mask_prompt']).lower()}",
        f'data: "{protocol["data"]["output_root"]}/{run["data"]}/{run["arm"]}"',
        'prompt_feature: "question"',
        'completion_feature: "answer"',
        "test: false",
        f'adapter_path: "{run["adapter_path"]}"',
        "lora_parameters:",
        "  keys:",
        *[f'    - "{key}"' for key in training["lora_keys"]],
        f"  rank: {training['lora_rank']}",
        f"  scale: {model['lora_scale']}",
        f"  dropout: {training['lora_dropout']}",
    ]
    return "\n".join(lines) + "\n"


def replay_schedule(config: dict[str, Any]) -> dict[str, Any]:
    """Replay mlx-lm's optimizer updates and logging for one configuration."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import build_schedule

    schedule = build_schedule(config["lr_schedule"])
    optimizer = optim.AdamW(
        learning_rate=schedule, **config["optimizer_config"]["adamw"]
    )
    param, grad = {"w": mx.zeros((1,))}, {"w": mx.ones((1,))}
    accumulation = config["grad_accumulation_steps"]
    per_update: list[float] = []
    logged: dict[int, float] = {}
    for iteration in range(1, config["iters"] + 1):
        if iteration % accumulation == 0:
            param = optimizer.apply_gradients(grad, param)
            per_update.append(optimizer.learning_rate.item())
        if iteration % config["steps_per_report"] == 0 or iteration == config["iters"]:
            logged[iteration] = optimizer.learning_rate.item()
    return {
        "updates": len(per_update),
        "discarded_microbatches": config["iters"] % accumulation,
        "lr_by_update": per_update,
        "logged_lr_by_iteration": logged,
        "peak_lr": max(per_update),
        "peak_updates": [
            k + 1 for k, v in enumerate(per_update) if v == max(per_update)
        ],
        "final_lr": per_update[-1],
    }


def generate(*, check: bool) -> dict[str, Any]:
    """Write (or check) every configuration file."""
    protocol = load_protocol()
    runs = training_runs(protocol)
    mismatched = []
    for run in runs:
        text = render_config(run, protocol)
        path = Path(run["config_path"])
        if check:
            if not path.is_file() or path.read_text("utf-8") != text:
                mismatched.append(path.as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    expected = {Path(r["config_path"]).name for r in runs}
    extra = sorted(p.name for p in CONFIG_DIR.glob("*.yaml") if p.name not in expected)
    return {
        "status": "passed" if not mismatched and not extra else "failed",
        "runs": len(runs),
        "mismatched": mismatched,
        "unexpected_files": extra,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--check", action="store_true")
    commands.add_parser("list")
    replay = commands.add_parser("replay")
    replay.add_argument("configs", nargs="+", type=Path)
    replay.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print JSON."""
    import yaml

    args = _parser().parse_args(argv)
    if args.command == "generate":
        result = generate(check=args.check)
    elif args.command == "list":
        result = {"runs": training_runs()}
    else:
        result = {}
        for path in args.configs:
            replayed = replay_schedule(yaml.safe_load(path.read_text("utf-8")))
            result[path.as_posix()] = replayed
        if args.output:
            write_json(args.output, result)
        result = {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in {"lr_by_update", "logged_lr_by_iteration"}
            }
            for k, v in result.items()
        }
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

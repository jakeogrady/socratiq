"""Resumable per-machine queue for the revision-v2 runs (Phase 2b).

Commands:
  plan / status   list the machine's items in order with status and expected minutes
  report          plain-text progress report with the learning-rate checks, to paste into an email
  run             run every unfinished item in order, stop on the first failure. On isik
                  it first runs the pilot and applies the acceptance rule itself (8e-5,
                  then 4e-5 once); the decision is saved and reused. On chee, --peak is
                  needed only for the second tier and the queue asks for it when it gets there.
  package         build the return packet (logs, manifests, predictions, final adapters, SHA-256)
  determinism     compare two prediction files response by response
  archive-v1      archive v1 runs and results before anything new starts

Finished items are skipped on restart. An interrupted evaluation resumes where
it stopped. An interrupted training run is moved aside (never deleted) and
restarted from the beginning, because MLX-LM cannot resume a schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import file_sha256, utc_now, write_json
from src.revision_v2 import configs as config_module
from src.revision_v2.protocol import (
    MACHINE_ENV,
    REPOSITORY_ROOT,
    RUNS_ROOT,
    load_protocol,
)

QUEUE_PATH = REPOSITORY_ROOT / "configs/revision_v2/queue.yaml"
PEAKS = ("8e-5", "4e-5")
PINNED = {"mlx": "0.30.3", "mlx-lm": "0.29.1"}
MIN_FREE_GIB = 60.0


def pilot_decision_path() -> Path:
    """Where the isik queue records the pilot decision."""
    return REPOSITORY_ROOT / RUNS_ROOT / "pilot_decision.json"


class QueueError(RuntimeError):
    """Raised when the queue must stop."""


def load_queue() -> dict[str, Any]:
    """Load the machine assignment."""
    import yaml

    return yaml.safe_load(QUEUE_PATH.read_text("utf-8"))


def base_name(condition_id: str) -> str:
    """Strip the learning-rate suffix: qwen3_0.6b_socratic_lr8e-5 -> qwen3_0.6b_socratic."""
    return condition_id.split("_lr")[0]


def condition_info(condition_id: str) -> dict[str, Any]:
    """Resolve a condition to its base model and, for adapters, its training run."""
    queue, protocol = load_queue(), load_protocol()
    if condition_id in queue["base_conditions"]:
        model = protocol["models"][queue["base_conditions"][condition_id]]
        return {
            "condition_id": condition_id,
            "model": model.get("local_path", model["identifier"]),
            "run": None,
        }
    runs = {r["run_id"]: r for r in config_module.training_runs(protocol)}
    if condition_id not in runs:
        msg = f"unknown condition {condition_id}"
        raise QueueError(msg)
    run = runs[condition_id]
    model = protocol["models"][run["model_key"]]
    return {
        "condition_id": condition_id,
        "model": model.get("local_path", model["identifier"]),
        "run": run,
    }


def expected_minutes(
    kind: str, key: str, machine: str, field: str | None = None
) -> int | None:
    """Look up the v1-derived duration estimate."""
    table = load_queue()["expected_minutes"]
    if kind == "train":
        return table["train"].get(key, {}).get(machine)
    return table["evaluate"].get(key, {}).get(field)


def eval_item(
    stage: str,
    condition_id: str,
    benchmark: str,
    prompt: str,
    mode: str,
    machine: str,
    root: Path,
) -> dict[str, Any]:
    """Build one evaluation item."""
    run_dir = root / condition_id / benchmark / f"{prompt}_{mode}"
    return {
        "id": f"{stage}/{condition_id}/{benchmark}/{prompt}_{mode}",
        "stage": stage,
        "kind": "eval",
        "condition_id": condition_id,
        "benchmark": benchmark,
        "prompt": prompt,
        "mode": mode,
        "run_dir": run_dir.as_posix(),
        "expected_minutes": expected_minutes(
            "eval", base_name(condition_id), machine, f"{benchmark}_{prompt}_{mode}"
        ),
    }


def train_item(stage: str, run_id: str, machine: str) -> dict[str, Any]:
    """Build one training item."""
    run = condition_info(run_id)["run"]
    return {
        "id": f"{stage}/{run_id}",
        "stage": stage,
        "kind": "train",
        "run_id": run_id,
        "config_path": run["config_path"],
        "run_dir": run["run_dir"],
        "expected_minutes": expected_minutes("train", base_name(run_id), machine),
    }


def resolve(templates: list[str], peak: str | None) -> list[str]:
    """Fill in the peak; without a peak, keep only the templates that do not need one."""
    return [
        t.format(peak=peak) for t in templates if peak is not None or "{peak}" not in t
    ]


def expand(machine: str, peak: str | None) -> list[dict[str, Any]]:
    """Return the machine's ordered item list (items needing an unknown peak are left out)."""
    if peak is not None and peak not in PEAKS:
        msg = f"--peak must be one of {PEAKS}"
        raise QueueError(msg)
    queue, protocol = load_queue(), load_protocol()
    assignment = queue["machines"][machine]
    evaluation_root = RUNS_ROOT / "evaluation"
    matrix = protocol["matrix"]["evaluation"]
    items = [
        train_item("1_core_training", run, machine)
        for run in resolve(assignment["train"], peak)
    ]
    det = protocol["matrix"]["determinism_check"]
    items.append(
        eval_item(
            "2_determinism",
            det["condition"],
            det["benchmark"],
            det["prompt"],
            det["mode"],
            machine,
            RUNS_ROOT / "determinism" / machine,
        )
    )
    conditions = resolve(assignment["core_conditions"], peak)
    greedy = matrix["core_and_base_greedy"]
    items += [
        eval_item("3_core_greedy", c, b, p, "greedy", machine, evaluation_root)
        for p, b in greedy
        if p == "P0"
        for c in conditions
    ]
    items += [
        eval_item("3_core_greedy", c, b, p, "greedy", machine, evaluation_root)
        for p, b in greedy
        if p != "P0"
        for c in conditions
    ]
    items += [
        eval_item("4_sc5", c, b, p, "sc5", machine, evaluation_root)
        for p, b in matrix["core_and_base_sc5"]
        for c in conditions
    ]
    second = resolve(assignment["second_tier"], peak)
    items += [train_item("5_second_tier_training", run, machine) for run in second]
    items += [
        eval_item("6_second_tier_greedy", c, b, p, "greedy", machine, evaluation_root)
        for p, b in matrix["second_tier_greedy"]
        for c in second
    ]
    return items


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Return a run's manifest, or None."""
    path = run_dir / "manifest.json"
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def item_status(item: dict[str, Any]) -> str:
    """Return done, partial or pending."""
    run_dir = REPOSITORY_ROOT / item["run_dir"]
    manifest = read_manifest(run_dir)
    if item["kind"] == "train":
        if (
            manifest
            and manifest.get("status") == "completed"
            and manifest.get("post_run_checks", {})
            .get("learning_rate", {})
            .get("status")
            == "passed"
        ):
            weights = REPOSITORY_ROOT / manifest["adapter"]["selected_weights_path"]
            if (
                weights.is_file()
                and file_sha256(weights)
                == manifest["adapter"]["selected_weights_sha256"]
            ):
                return "done"
    elif manifest and manifest.get("status") == "completed":
        predictions = run_dir / "predictions.jsonl"
        if predictions.is_file() and file_sha256(predictions) == manifest.get(
            "prediction_sha256"
        ):
            return "done"
    return "partial" if run_dir.exists() and any(run_dir.iterdir()) else "pending"


def git(*args: str) -> str:
    """Run a git command in the repository and return stdout."""
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def preconditions(machine: str, *, frozen: bool) -> dict[str, Any]:
    """Refuse to start unless the checkout, environment, data and checkpoint are right."""
    from importlib.metadata import version

    problems = []
    if os.environ.get(MACHINE_ENV) != machine:
        problems.append(
            f"{MACHINE_ENV} must be '{machine}'; use scripts/phase2_queue_{machine}.sh"
        )
    # macOS rewrites the tracked .DS_Store files on its own; they carry no code.
    modified = [
        line
        for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
        if not line.endswith(".DS_Store")
    ]
    if modified:
        problems.append(
            f"tracked files are modified ({modified}); do not edit code or configs"
        )
    head = git("rev-parse", "HEAD")
    tag = load_protocol()["freeze_tag"]
    if frozen:
        try:
            tagged = git("rev-parse", f"{tag}^{{commit}}")
        except subprocess.CalledProcessError:
            tagged = None
        if tagged != head:
            problems.append(
                f"HEAD is not the frozen tag {tag}; run `git checkout {tag}`"
            )
    for package, pinned in PINNED.items():
        if version(package) != pinned:
            problems.append(
                f"{package} {version(package)} installed, {pinned} required"
            )
    from src.revision_v2.data import verify_files

    if verify_files()["status"] != "passed":
        problems.append(
            "data files missing or not matching configs/revision_v2/dataset_manifest.json; see setup step 5"
        )
    if load_queue()["machines"][machine]["needs_llama_checkpoint"]:
        from src.revision_v2.llama_base import verify_checkpoint

        llama = load_protocol()["models"]["llama3.2_1b"]
        if (
            verify_checkpoint(REPOSITORY_ROOT / llama["local_path"])["status"]
            != "passed"
        ):
            problems.append(
                f"Llama checkpoint missing or wrong at {llama['local_path']}; see setup step 4"
            )
    free = shutil.disk_usage(REPOSITORY_ROOT).free / 1024**3
    if free < MIN_FREE_GIB:
        problems.append(f"only {free:.0f} GiB free; {MIN_FREE_GIB:.0f} GiB required")
    if problems:
        raise QueueError("preconditions failed:\n  - " + "\n  - ".join(problems))
    return {"head": head, "free_gib": round(free, 1)}


def log_event(machine: str, event: dict[str, Any]) -> None:
    """Append one event to the machine's queue log."""
    path = REPOSITORY_ROOT / RUNS_ROOT / "queue_logs" / machine / "queue_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), **event}, sort_keys=True) + "\n")


def command_for(item: dict[str, Any]) -> list[str]:
    """Build the subprocess command for one item."""
    if item["kind"] == "train":
        return [
            sys.executable,
            "-m",
            "src.revision_v2.train",
            "run",
            "--config",
            item["config_path"],
        ]
    info = condition_info(item["condition_id"])
    command = [
        sys.executable,
        "-m",
        "src.revision_v2.evaluate",
        "--condition-id",
        item["condition_id"],
        "--model",
        info["model"],
        "--benchmark",
        item["benchmark"],
        "--prompt",
        item["prompt"],
        "--mode",
        item["mode"],
        "--run-dir",
        item["run_dir"],
    ]
    if info["run"] is not None:
        manifest = read_manifest(REPOSITORY_ROOT / info["run"]["run_dir"])
        if not manifest or manifest.get("status") != "completed":
            msg = f"training {item['condition_id']} is not complete; it must finish before its evaluation"
            raise QueueError(msg)
        command += [
            "--adapter-path",
            info["run"]["adapter_path"],
            "--adapter-sha256",
            manifest["adapter"]["selected_weights_sha256"],
        ]
    return command


def run_item(item: dict[str, Any], machine: str) -> None:
    """Run one item, teeing its output to a per-item log."""
    run_dir = REPOSITORY_ROOT / item["run_dir"]
    if item["kind"] == "train" and item_status(item) == "partial":
        aside = run_dir.with_name(
            f"{run_dir.name}.incomplete-{utc_now().replace(':', '')}"
        )
        run_dir.rename(aside)
        log_event(
            machine,
            {
                "item": item["id"],
                "event": "moved_incomplete_training_aside",
                "to": str(aside),
            },
        )
    command = command_for(item)
    log_path = (
        REPOSITORY_ROOT
        / RUNS_ROOT
        / "queue_logs"
        / machine
        / (item["id"].replace("/", "__") + ".log")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_event(machine, {"item": item["id"], "event": "start", "command": command})
    print(
        f"\n=== {item['id']} (expected ~{item['expected_minutes']} min) ===", flush=True
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {utc_now()} start: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None  # noqa: S101 - PIPE always sets stdout
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        code = process.wait()
    status = item_status(item)
    log_event(
        machine,
        {
            "item": item["id"],
            "event": "finish",
            "return_code": code,
            "status": status,
            "log": str(log_path),
        },
    )
    if code != 0 or status != "done":
        msg = f"item {item['id']} failed (return code {code}, status {status}). Stop. Send {log_path} to the PI."
        raise QueueError(msg)


def print_plan(machine: str, requested_peak: str | None) -> list[dict[str, Any]]:
    """Print every item with its status and expected minutes."""
    peak = effective_peak(machine, requested_peak)
    if peak is None:
        print(
            "(peak not known yet: items that depend on the pilot's learning rate are not listed)"
        )
    items = expand(machine, peak)
    total = remaining = 0
    for number, item in enumerate(items, start=1):
        status = item_status(item)
        minutes = item["expected_minutes"] or 0
        total += minutes
        remaining += 0 if status == "done" else minutes
        print(f"{number:3d}  {status:8s} {minutes:5d} min  {item['id']}")
    print(
        f"\n{len(items)} items; expected {total / 60:.1f} h in total, {remaining / 60:.1f} h remaining"
    )
    return items


def produced_files(item: dict[str, Any]) -> str:
    """Describe the files an item writes."""
    if item["kind"] == "train":
        return f"`{item['run_dir']}/`: train.log, manifest.json, config.yaml, adapter/adapters.safetensors"
    return f"`{item['run_dir']}/`: manifest.json, predictions.jsonl"


def short_command(item: dict[str, Any]) -> str:
    """Return the module invocation the queue runs for an item (adapter hash flags omitted)."""
    if item["kind"] == "train":
        return f"`train run --config {item['config_path']}`"
    return f"`evaluate --condition-id {item['condition_id']} --benchmark {item['benchmark']} --prompt {item['prompt']} --mode {item['mode']}`"


def report(machine: str, requested_peak: str | None) -> str:
    """Return a plain-text progress report the students paste into an email."""
    head = git("rev-parse", "HEAD")
    peak = effective_peak(machine, requested_peak)
    expected_rows = {"gsm8k": 1319, "gsm_hard": 1319, "multiarith": 180, "svamp": 300}
    lines = []
    if machine == load_queue()["pilot"]["machine"]:
        decision = pilot_decision()
        if decision:
            tried = ", ".join(
                f"{a['peak']} {a['decision']} (final val loss {a['final_validation_loss']})"
                for a in decision["attempts"]
            )
            lines.append(f"pilot decision: {decision['peak']}  [{tried}]")
        else:
            lines.append("pilot decision: not yet (the pilot is the first item)")
            for candidate in PEAKS:
                item = train_item(
                    "0_pilot",
                    load_queue()["pilot"]["run"].format(peak=candidate),
                    machine,
                )
                lines.append(f"  pilot at {candidate}: {item_status(item)}")
            lines.append(
                "The full item list appears once the pilot has decided the learning rate."
            )
            return "\n".join(lines)
    elif load_queue()["machines"][machine]["second_tier"] and peak is None:
        lines.append(
            "peak: not given yet (needed only for the second tier, from isik's pilot decision)"
        )
    lines.append(f"machine={machine} peak={peak} commit={head[:12]} at={utc_now()}")
    counts = {"done": 0, "partial": 0, "pending": 0}
    for number, item in enumerate(expand(machine, peak), start=1):
        status = item_status(item)
        counts[status] += 1
        manifest = read_manifest(REPOSITORY_ROOT / item["run_dir"]) or {}
        line = f"{number:3d} {status:8s} {item['id']}"
        if status == "done":
            same_commit = (
                manifest.get("environment", {}).get("git", {}).get("commit") == head
            )
            if item["kind"] == "train":
                lr = manifest["post_run_checks"]["learning_rate"]
                losses = manifest.get("validation_losses") or [{"loss": None}]
                ok = lr["status"] == "passed" and same_commit
                line += (
                    f" | LR update16 {lr['logged_at_update_16']} (expected {lr['expected_at_update_16']})"
                    f" update188 {lr['logged_at_update_188']} (expected {lr['expected_at_update_188']})"
                    f" | val loss {losses[0]['loss']} -> {losses[-1]['loss']}"
                    f" | adapter {manifest['adapter']['selected_weights_sha256'][:12]}"
                )
            else:
                rows = manifest.get("observed_rows")
                ok = rows == expected_rows[item["benchmark"]] and same_commit
                reasons = manifest.get("summary", {}).get("finish_reasons", {})
                line += f" | rows {rows}/{expected_rows[item['benchmark']]} | finish {reasons}"
                comparison_path = v1_comparison_path(machine)
                if item["stage"] == "2_determinism" and comparison_path.is_file():
                    comparison = json.loads(comparison_path.read_text("utf-8"))
                    line += f" | vs v1: {comparison['status']}"
                    ok = ok and comparison["status"] != "different"
            line += " | OK" if ok else " | CHECK"
        lines.append(line)
    lines.append(
        f"done {counts['done']}, partial {counts['partial']}, pending {counts['pending']}"
    )
    return "\n".join(lines)


def markdown_table(machine: str, peak: str) -> str:
    """Return the runbook table: one row per item, in queue order."""
    lines = [
        "| # | Stage | Item | Command run by the queue | Expected (min) | Produces |",
        "|---:|---|---|---|---:|---|",
    ]
    total = 0
    for number, item in enumerate(expand(machine, peak), start=1):
        minutes = item["expected_minutes"] or 0
        total += minutes
        stage, _, name = item["id"].partition("/")
        lines.append(
            f"| {number} | {stage} | {name} | {short_command(item)} | {minutes} | {produced_files(item)} |"
        )
    lines.append(f"| | | **Total** | | **{total} ({total / 60:.1f} h)** | |")
    return "\n".join(lines)


def pilot_decision() -> dict[str, Any] | None:
    """Return the recorded pilot decision, if any."""
    path = pilot_decision_path()
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def effective_peak(machine: str, requested: str | None) -> str | None:
    """On the pilot machine the pilot decides the peak; elsewhere it comes from --peak."""
    if machine != load_queue()["pilot"]["machine"]:
        return requested
    decision = pilot_decision()
    decided = decision["peak"] if decision else None
    if requested and decided and requested != decided:
        msg = f"the pilot chose {decided}; do not pass --peak {requested} on {machine}"
        raise QueueError(msg)
    return decided


def decide_pilot(machine: str) -> str:
    """Run the pilot at 8e-5 and, only if it fails the rule, once at 4e-5."""
    from src.revision_v2.train import pilot_check

    decision = pilot_decision()
    if decision:
        return decision["peak"]
    template = load_queue()["pilot"]["run"]
    attempts = []
    for peak in PEAKS:
        item = train_item("0_pilot", template.format(peak=peak), machine)
        if item_status(item) != "done":
            print(f"\nPilot at peak {peak}: training Qwen3-0.6B Socratic.", flush=True)
            run_item(item, machine)
        check = pilot_check(REPOSITORY_ROOT / item["run_dir"])
        attempts.append(
            {
                "peak": peak,
                "decision": check["decision"],
                "initial_validation_loss": check["initial_validation_loss"],
                "final_validation_loss": check["final_validation_loss"],
                "limit": check["v1_reference_final_validation_loss"],
            }
        )
        log_event(machine, {"item": item["id"], "event": "pilot_check", **attempts[-1]})
        print(
            f"Pilot at peak {peak}: {check['decision']} (final validation loss "
            f"{check['final_validation_loss']}, limit {check['v1_reference_final_validation_loss']})",
            flush=True,
        )
        if check["decision"] == "accept":
            write_json(
                pilot_decision_path(),
                {
                    "peak": peak,
                    "decided_at": utc_now(),
                    "commit": git("rev-parse", "HEAD"),
                    "rule": "first of 8e-5, then 4e-5, that passes the protocol's pilot rule",
                    "attempts": attempts,
                },
            )
            return peak
    msg = (
        "both pilot runs (8e-5 and 4e-5) failed the acceptance rule. Stop and send the PI "
        "runs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5/pilot_check.json and "
        "runs/revision_v2/training/qwen3_0.6b_socratic_lr4e-5/pilot_check.json"
    )
    raise QueueError(msg)


def v1_comparison_path(machine: str) -> Path:
    """Where the determinism comparison with this machine's v1 run is written."""
    return REPOSITORY_ROOT / RUNS_ROOT / "determinism" / machine / "v1_comparison.json"


def compare_with_v1(machine: str, item: dict[str, Any]) -> dict[str, Any]:
    """Compare the determinism run with this machine's v1 run of the same prompt."""
    reference = (
        REPOSITORY_ROOT / load_queue()["machines"][machine]["v1_determinism_reference"]
    )
    new = REPOSITORY_ROOT / item["run_dir"] / "predictions.jsonl"
    if not reference.is_file():
        result: dict[str, Any] = {
            "status": "v1 reference not found",
            "reference": str(reference),
        }
    else:
        result = determinism(new, reference)
        result["status"] = "identical" if result["identical"] else "different"
    result["checked_at"] = utc_now()
    write_json(v1_comparison_path(machine), result)
    log_event(
        machine,
        {"item": item["id"], "event": "v1_comparison", "status": result["status"]},
    )
    return result


def run(machine: str, requested_peak: str | None) -> None:
    """Run every unfinished item in order; stop on the first failure."""
    state = preconditions(machine, frozen=True)
    if machine == load_queue()["pilot"]["machine"]:
        effective_peak(machine, requested_peak)
        peak: str | None = decide_pilot(machine)
    else:
        peak = requested_peak
    log_event(machine, {"event": "queue_start", "peak": peak, **state})
    for item in expand(machine, peak):
        if item_status(item) == "done":
            manifest = read_manifest(REPOSITORY_ROOT / item["run_dir"]) or {}
            if (
                manifest.get("environment", {}).get("git", {}).get("commit")
                != state["head"]
            ):
                msg = f"{item['run_dir']} is finished but was produced by a different commit; ask the PI"
                raise QueueError(msg)
            print(f"skip (done)  {item['id']}")
        else:
            run_item(item, machine)
        if (
            item["stage"] == "2_determinism"
            and not v1_comparison_path(machine).is_file()
        ):
            result = compare_with_v1(machine, item)
            print(
                f"Determinism check against your v1 run: {result['status']}", flush=True
            )
    if peak is None and load_queue()["machines"][machine]["second_tier"]:
        log_event(machine, {"event": "waiting_for_peak"})
        print(
            "\nCore items finished. The second tier needs the learning rate chosen by the "
            "pilot on isik (the first line of isik's report, for example 'pilot decision: 8e-5').\n"
            f"Start the queue again with: ./scripts/phase2_queue_{machine}.sh run --peak <that value>"
        )
        return
    log_event(machine, {"event": "queue_complete", "peak": peak})
    print("\nAll items finished. Run the package command next.")


def package(machine: str, requested_peak: str | None) -> Path:
    """Build the return packet with SHA-256 for every file."""
    peak = effective_peak(machine, requested_peak)
    items = expand(machine, peak)
    if machine == load_queue()["pilot"]["machine"]:
        pilots = [
            train_item("0_pilot", load_queue()["pilot"]["run"].format(peak=p), machine)
            for p in PEAKS
        ]
        items = [
            *pilots,
            *[i for i in items if i["run_dir"] not in {p["run_dir"] for p in pilots}],
        ]
    files: list[Path] = []
    statuses = {}
    for item in items:
        statuses[item["id"]] = item_status(item)
        run_dir = REPOSITORY_ROOT / item["run_dir"]
        names = (
            [
                "train.log",
                "manifest.json",
                "config.yaml",
                "pilot_check.json",
                "adapter/adapters.safetensors",
                "adapter/adapter_config.json",
            ]
            if item["kind"] == "train"
            else ["manifest.json", "predictions.jsonl"]
        )
        files += [run_dir / n for n in names if (run_dir / n).is_file()]
    log_dir = REPOSITORY_ROOT / RUNS_ROOT / "queue_logs" / machine
    files += sorted(p for p in log_dir.glob("*") if p.is_file())
    files += [
        p for p in (pilot_decision_path(), v1_comparison_path(machine)) if p.is_file()
    ]
    files = [f for f in files if not f.name.startswith(".env")]
    stamp = utc_now().replace(":", "")
    out_dir = REPOSITORY_ROOT / RUNS_ROOT / "return"
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_manifest = {
        "machine": machine,
        "peak": peak,
        "created_at": utc_now(),
        "git_commit": git("rev-parse", "HEAD"),
        "item_status": statuses,
        "files": {
            f.relative_to(REPOSITORY_ROOT).as_posix(): file_sha256(f) for f in files
        },
    }
    manifest_path = out_dir / f"{machine}-{stamp}-packet_manifest.json"
    write_json(manifest_path, packet_manifest)
    archive = out_dir / f"socratiq-v2-{machine}-{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in [*files, manifest_path]:
            tar.add(f, arcname=f.relative_to(REPOSITORY_ROOT).as_posix())
    archive.with_name(archive.name + ".sha256").write_text(
        f"{file_sha256(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return archive


def determinism(left: Path, right: Path) -> dict[str, Any]:
    """Compare two prediction files response by response, byte for byte."""

    def responses(path: Path) -> dict[str, list[str]]:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        return {r["example_id"]: [s["response"] for s in r["samples"]] for r in rows}

    a, b = responses(left), responses(right)
    shared = sorted(set(a) & set(b))
    differing = [k for k in shared if a[k] != b[k]]
    digest = {
        name: hashlib.sha256(
            "".join(json.dumps(x[k]) for k in shared).encode()
        ).hexdigest()
        for name, x in (("left", a), ("right", b))
    }
    return {
        "left": str(left),
        "right": str(right),
        "items_compared": len(shared),
        "items_differing": len(differing),
        "first_differing": differing[:10],
        "responses_sha256": digest,
        "identical": not differing and len(a) == len(b) == len(shared),
    }


def archive_v1(destination: Path) -> Path:
    """Archive v1 logs, adapters and predictions (read-only) before Phase 2b starts."""
    sources = [
        p
        for p in (
            REPOSITORY_ROOT / "runs/reviewer_rerun",
            REPOSITORY_ROOT / "results/reviewer_rerun",
        )
        if p.exists()
    ]
    if not sources:
        msg = "no runs/reviewer_rerun or results/reviewer_rerun directory found"
        raise QueueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as tar:
        for source in sources:
            tar.add(
                source,
                arcname=source.relative_to(REPOSITORY_ROOT).as_posix(),
                filter=lambda info: None
                if Path(info.name).name.startswith(".env")
                else info,
            )
    destination.with_name(destination.name + ".sha256").write_text(
        f"{file_sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "run", "package", "table", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--machine", choices=("chee", "isik"), required=True)
        sub.add_argument("--peak", choices=PEAKS, default=None)
    det = commands.add_parser("determinism")
    det.add_argument("left", type=Path)
    det.add_argument("right", type=Path)
    arch = commands.add_parser("archive-v1")
    arch.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one queue command."""
    args = _parser().parse_args(argv)
    try:
        if args.command in {"plan", "status"}:
            print_plan(args.machine, args.peak)
        elif args.command == "table":
            print(markdown_table(args.machine, args.peak or PEAKS[0]))
        elif args.command == "report":
            print(report(args.machine, args.peak))
        elif args.command == "run":
            run(args.machine, args.peak)
        elif args.command == "package":
            print(f"packet: {package(args.machine, args.peak)}")
        elif args.command == "determinism":
            result = determinism(args.left, args.right)
            print(json.dumps(result, indent=2))
            return 0 if result["identical"] else 1
        else:
            print(f"archive: {archive_v1(args.destination)}")
    except QueueError as error:
        print(f"\nSTOP: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

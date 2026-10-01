"""Load the frozen revision-v2 protocol and expose its identity."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

from src.rerun_utils import file_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs/revision_v2/protocol.yaml"
RUNS_ROOT = Path("runs/revision_v2")
FORBIDDEN_RUNS_ROOT = Path("runs/reviewer_rerun")
MACHINE_ENV = "SOCRATIQ_MACHINE"


class ProtocolError(RuntimeError):
    """Raised when an operation would leave the frozen revision-v2 protocol."""


@cache
def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Return the parsed protocol mapping."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ProtocolError(msg)
    return value


def protocol_identity(path: Path = PROTOCOL_PATH) -> dict[str, str]:
    """Return the path and SHA-256 that every v2 artifact must record."""
    return {"path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": file_sha256(path)}


def machine_label() -> dict[str, str | None]:
    """Identify the machine: queue-assigned label plus host details."""
    return {
        "label": os.environ.get(MACHINE_ENV),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "os": platform.platform(),
    }


def tracked_git_state() -> dict[str, str | None]:
    """Return the commit, branch and tracked-file changes; never untracked file names."""

    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() or None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain_tracked_only": git(
            "status", "--porcelain", "--untracked-files=no"
        ),
    }


def require_v2_output(path: Path) -> Path:
    """Refuse any output location outside runs/revision_v2 or inside v1 runs."""
    resolved = path.resolve()
    forbidden = (REPOSITORY_ROOT / FORBIDDEN_RUNS_ROOT).resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        msg = f"refusing to write under {FORBIDDEN_RUNS_ROOT}: {path}"
        raise ProtocolError(msg)
    return path

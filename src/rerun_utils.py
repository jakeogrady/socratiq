"""Small dependency-free helpers shared by reviewer-rerun commands."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs/reviewer_rerun/protocol.yaml"
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and manifests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def object_sha256(value: Any) -> str:
    """Hash a JSON-serializable value using canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_identity(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Return the immutable identity every official artifact must record."""
    if not path.is_file():
        raise FileNotFoundError(f"reviewer-rerun protocol is missing: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def directory_size(path: Path) -> int:
    """Return the total size of regular files below a directory."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def directory_sha256(path: Path) -> str:
    """Hash file names and contents below a directory deterministically."""
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from JSONL and include line context in parse errors."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {path} at line {line_number}: {exc}"
                raise ValueError(msg) from exc
            if not isinstance(value, dict):
                msg = f"Expected a JSON object in {path} at line {line_number}"
                raise ValueError(msg)
            yield value


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    """Write an indented JSON document atomically."""
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL atomically."""
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write(path, content)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object and flush it for resumable runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

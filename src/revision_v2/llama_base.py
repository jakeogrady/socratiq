"""Build and verify the revision-v2 Llama base checkpoint.

The base is Meta's exact Llama-3.2-1B-Instruct weights in MLX bf16. The
weights come from a converted repository only if every tensor is bitwise equal
to Meta's hash-verified model.safetensors. The single change to the checkpoint
is the chat template's date logic: the system block always reads
"Today Date: 26 Jul 2024" (the template's own fallback literal), so training
and every evaluation render the same prompt regardless of the run date.

Commands:
  compare  bitwise tensor comparison of two safetensors files
  build    assemble the checkpoint directory and write its manifest
  verify   check a placed checkpoint against the tracked manifest
"""

from __future__ import annotations

import argparse
import json
import mmap
import shutil
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import directory_sha256, file_sha256, utc_now, write_json

FIXED_DATE = "26 Jul 2024"
ORIGINAL_DATE_BLOCK = (
    "{%- if not date_string is defined %}\n"
    "    {%- if strftime_now is defined %}\n"
    '        {%- set date_string = strftime_now("%d %b %Y") %}\n'
    "    {%- else %}\n"
    '        {%- set date_string = "26 Jul 2024" %}\n'
    "    {%- endif %}\n"
    "{%- endif %}"
)
PATCHED_DATE_BLOCK = (
    "{#- revision-v2: fixed system date for training and evaluation #}\n"
    '{%- set date_string = "26 Jul 2024" %}'
)
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DEFAULT_MANIFEST = Path("configs/revision_v2/llama_base_manifest.json")


class LlamaBaseError(RuntimeError):
    """Raised when the checkpoint does not match the protocol."""


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return the safetensors header and the byte offset of the data block."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return header, 8 + length


def compare_safetensors(left: Path, right: Path) -> dict[str, Any]:
    """Compare two safetensors files tensor by tensor on raw bytes."""
    left_header, left_base = read_safetensors_header(left)
    right_header, right_base = read_safetensors_header(right)
    left_names = {k for k in left_header if k != "__metadata__"}
    right_names = {k for k in right_header if k != "__metadata__"}
    rows = []
    with (
        left.open("rb") as left_handle,
        right.open("rb") as right_handle,
        mmap.mmap(left_handle.fileno(), 0, access=mmap.ACCESS_READ) as left_map,
        mmap.mmap(right_handle.fileno(), 0, access=mmap.ACCESS_READ) as right_map,
    ):
        for name in sorted(left_names & right_names):
            a, b = left_header[name], right_header[name]
            a_start, a_end = (left_base + offset for offset in a["data_offsets"])
            b_start, b_end = (right_base + offset for offset in b["data_offsets"])
            same_meta = a["dtype"] == b["dtype"] and a["shape"] == b["shape"]
            same_bytes = (
                same_meta and left_map[a_start:a_end] == right_map[b_start:b_end]
            )
            rows.append(
                {
                    "tensor": name,
                    "dtype": a["dtype"],
                    "shape": a["shape"],
                    "metadata_equal": same_meta,
                    "bytes_equal": same_bytes,
                    "nbytes": a_end - a_start,
                }
            )
    return {
        "left": str(left),
        "right": str(right),
        "left_sha256": file_sha256(left),
        "right_sha256": file_sha256(right),
        "left_header_metadata": left_header.get("__metadata__"),
        "right_header_metadata": right_header.get("__metadata__"),
        "only_in_left": sorted(left_names - right_names),
        "only_in_right": sorted(right_names - left_names),
        "tensors_compared": len(rows),
        "tensors_bytes_equal": sum(r["bytes_equal"] for r in rows),
        "parameter_bytes_compared": sum(r["nbytes"] for r in rows),
        "all_tensors_bitwise_equal": (
            left_names == right_names and all(r["bytes_equal"] for r in rows)
        ),
        "tensors": rows,
    }


def patch_chat_template(tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose chat template always uses the fixed date."""
    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or template.count(ORIGINAL_DATE_BLOCK) != 1:
        msg = "chat template does not contain exactly one expected date block"
        raise LlamaBaseError(msg)
    patched = dict(tokenizer_config)
    patched["chat_template"] = template.replace(ORIGINAL_DATE_BLOCK, PATCHED_DATE_BLOCK)
    if "strftime_now" in patched["chat_template"]:
        msg = "patched template still references strftime_now"
        raise LlamaBaseError(msg)
    return patched


def build_checkpoint(
    source_dir: Path,
    meta_reference: Path,
    output_dir: Path,
    *,
    source: dict[str, Any],
    meta: dict[str, Any],
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Assemble the checkpoint, prove weight identity, and write its manifest."""
    if output_dir.exists() and any(output_dir.iterdir()):
        msg = f"output directory is not empty: {output_dir}"
        raise LlamaBaseError(msg)
    if file_sha256(meta_reference) != meta["model_safetensors_sha256"]:
        msg = "Meta reference file does not match Meta's published SHA-256"
        raise LlamaBaseError(msg)
    comparison = compare_safetensors(meta_reference, source_dir / "model.safetensors")
    if not comparison["all_tensors_bitwise_equal"]:
        msg = "source weights are not bitwise equal to Meta's weights; convert instead"
        raise LlamaBaseError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        if name == "tokenizer_config.json":
            continue
        shutil.copy2(source_dir / name, output_dir / name)
    original = json.loads((source_dir / "tokenizer_config.json").read_text("utf-8"))
    patched = patch_chat_template(original)
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(patched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    unchanged_keys = sorted(
        k for k in original if k != "chat_template" and original[k] == patched[k]
    )
    final = compare_safetensors(meta_reference, output_dir / "model.safetensors")
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "purpose": "revision-v2 Llama base: Meta weights in MLX bf16, fixed chat-template date",
        "local_path": str(output_dir),
        "meta_reference": meta,
        "weights_source": source,
        "weights_bitwise_equal_to_meta": final["all_tensors_bitwise_equal"],
        "tensors_compared": final["tensors_compared"],
        "chat_template_patch": {
            "fixed_date_string": FIXED_DATE,
            "original_block": ORIGINAL_DATE_BLOCK,
            "patched_block": PATCHED_DATE_BLOCK,
            "original_tokenizer_config_sha256": file_sha256(
                source_dir / "tokenizer_config.json"
            ),
            "other_keys_unchanged": len(unchanged_keys) == len(original) - 1,
        },
        "files": {name: file_sha256(output_dir / name) for name in CHECKPOINT_FILES},
        "directory_sha256": directory_sha256(output_dir),
    }
    write_json(manifest_path, manifest)
    return {"manifest": manifest, "comparison": final}


def verify_checkpoint(
    path: Path, manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, Any]:
    """Check every checkpoint file and the directory hash against the manifest."""
    manifest = json.loads(manifest_path.read_text("utf-8"))
    present = (
        sorted(p.name for p in path.iterdir() if p.is_file()) if path.is_dir() else []
    )
    problems = []
    if present != sorted(CHECKPOINT_FILES):
        problems.append(f"expected files {sorted(CHECKPOINT_FILES)}, found {present}")
    for name, expected in manifest["files"].items():
        candidate = path / name
        if not candidate.is_file() or file_sha256(candidate) != expected:
            problems.append(f"{name}: hash mismatch or missing")
    directory = directory_sha256(path) if path.is_dir() else None
    if directory != manifest["directory_sha256"]:
        problems.append("directory_sha256 mismatch")
    return {
        "path": str(path),
        "status": "passed" if not problems else "failed",
        "problems": problems,
        "directory_sha256": directory,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path)
    build = commands.add_parser("build")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument(
        "--source-json", required=True, help="JSON describing the weights source"
    )
    build.add_argument("--meta-reference", type=Path, required=True)
    build.add_argument(
        "--meta-json",
        required=True,
        help="JSON with Meta repo, revision and model_safetensors_sha256",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify = commands.add_parser("verify")
    verify.add_argument("path", type=Path)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print a JSON result."""
    args = _parser().parse_args(argv)
    if args.command == "compare":
        result = compare_safetensors(args.left, args.right)
        if args.output:
            write_json(args.output, result)
        result = {k: v for k, v in result.items() if k != "tensors"}
    elif args.command == "build":
        built = build_checkpoint(
            args.source_dir,
            args.meta_reference,
            args.output_dir,
            source=json.loads(args.source_json),
            meta=json.loads(args.meta_json),
            manifest_path=args.manifest,
        )
        result = built["manifest"]
    else:
        result = verify_checkpoint(args.path, args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Resumable matched synthetic-data generation for the reviewer rerun.

Offline stages (snapshot, build, estimate, assemble, retry, render) are kept
separate from API stages (preflight, submit, status, download). Full Batch
submission requires an explicit confirmation flag and is never implicit.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.paired_dataset import SCHEMA_VERSION, CanonicalExample, build_paired_dataset
from src.rerun_utils import (
    file_sha256,
    protocol_identity,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)

DEFAULT_SOURCE_DATASET = "openai/gsm8k"
DEFAULT_SOURCE_CONFIG = "main"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_SOURCE_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DEFAULT_TEACHER_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_VARIANTS_PER_SOURCE = 3
DEFAULT_MAX_OUTPUT_TOKENS = 3500
PROMPT_VERSION = "matched-pairs-v2"
PAID_CONFIRMATION = "SUBMIT_PAID_BATCH"
HTTP_OK = 200

SYSTEM_INSTRUCTIONS = """You create concise synthetic grade-school arithmetic training examples.
Return exactly three variations of the supplied source problem. Preserve the
source problem's mathematical skill and approximate difficulty while changing
the story, entities, and numbers. Every variation must contain one harmless
sentence of unused information. Keep all intermediate physical quantities
non-negative and make the final answer a positive integer.

For every solution step, provide two separate fields:
1. guiding_question: one short Socratic question ending with a question mark;
2. reasoning: a concise declarative explanation containing any calculation.

The reasoning field must stand on its own after the guiding question is
removed. Do not put questions, rhetorical checks, or instructions in the
reasoning field. Use correct explicit arithmetic equalities where possible,
and make the last explicit equality's result match the final answer. Do not
place a guiding question after the final reasoning step. End each variation
with exactly one final_answer in `#### N` format.
Return only data matching the supplied JSON schema."""


VARIANTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "variants": {
            "type": "array",
            "minItems": DEFAULT_VARIANTS_PER_SOURCE,
            "maxItems": DEFAULT_VARIANTS_PER_SOURCE,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "variant_id": {"type": "integer", "minimum": 1, "maximum": 3},
                    "synthetic_question": {"type": "string", "minLength": 1},
                    "solution_steps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "guiding_question": {"type": "string", "minLength": 2},
                                "reasoning": {"type": "string", "minLength": 1},
                            },
                            "required": ["guiding_question", "reasoning"],
                        },
                    },
                    "final_answer": {
                        "type": "string",
                        "pattern": r"^#### [1-9][0-9]*$",
                    },
                },
                "required": [
                    "variant_id",
                    "synthetic_question",
                    "solution_steps",
                    "final_answer",
                ],
            },
        }
    },
    "required": ["variants"],
}


def response_body(
    source: Mapping[str, Any],
    *,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build one Responses API body with strict structured output."""
    question = _required_source_text(source, "question")
    solution = _required_source_text(source, "answer")
    user_input = (
        "Source problem:\n"
        f"{question}\n\n"
        "Source worked solution:\n"
        f"{solution}\n\n"
        "Create exactly three valid variations. Use the source worked solution "
        "to preserve mathematical correctness."
    )
    return {
        "model": model,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "matched_socratic_variants",
                "schema": VARIANTS_SCHEMA,
                "strict": True,
            }
        },
        "store": False,
    }


def batch_request(
    source: Mapping[str, Any],
    *,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build one Batch JSONL request line."""
    source_id = _required_source_text(source, "source_id")
    return {
        "custom_id": source_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": response_body(
            source,
            model=model,
            max_output_tokens=max_output_tokens,
        ),
    }


def _required_source_text(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source field {key!r} must be a non-empty string")
    return value.strip()


def snapshot_source(
    output_path: Path,
    *,
    dataset_name: str = DEFAULT_SOURCE_DATASET,
    dataset_config: str = DEFAULT_SOURCE_CONFIG,
    split: str = DEFAULT_SOURCE_SPLIT,
    revision: str | None = DEFAULT_SOURCE_REVISION,
    expected_rows: int | None = 7473,
    limit: int | None = None,
) -> dict[str, Any]:
    """Download and normalize the source split; import datasets lazily."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for snapshot; install the locked project environment."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_name, dataset_config, **kwargs)
    total_rows = len(dataset)
    if expected_rows is not None and total_rows != expected_rows:
        msg = f"Expected {expected_rows} source rows, found {total_rows}"
        raise ValueError(msg)

    stop = total_rows if limit is None else min(limit, total_rows)
    rows: list[dict[str, Any]] = []
    for index in range(stop):
        row = dataset[index]
        rows.append(
            {
                "source_id": f"gsm8k-{split}-{index:06d}",
                "source_index": index,
                "question": _required_source_text(row, "question"),
                "answer": _required_source_text(row, "answer"),
            }
        )
    write_jsonl(output_path, rows)

    manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "requested_revision": revision,
        "resolved_revision": revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_rows_available": total_rows,
        "source_rows_written": len(rows),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def build_batch_input(
    source_path: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_TEACHER_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    start_index: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build an offline Batch request file with unique provenance IDs."""
    sources = list(read_jsonl(source_path))
    if start_index < 0 or start_index > len(sources):
        raise ValueError("start_index is outside the source snapshot")
    selected = sources[start_index:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        selected = selected[:limit]

    requests: list[dict[str, Any]] = []
    custom_ids: set[str] = set()
    for source in selected:
        request = batch_request(
            source,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        custom_id = request["custom_id"]
        if custom_id in custom_ids:
            raise ValueError(f"Duplicate custom_id in source snapshot: {custom_id}")
        custom_ids.add(custom_id)
        requests.append(request)
    write_jsonl(output_path, requests)

    manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "stage": "built",
        "prompt_version": PROMPT_VERSION,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_start_index": start_index,
        "request_count": len(requests),
        "variants_per_request": DEFAULT_VARIANTS_PER_SOURCE,
        "candidate_example_count": len(requests) * DEFAULT_VARIANTS_PER_SOURCE,
        "requested_model": model,
        "allow_model_fallback": False,
        "max_output_tokens": max_output_tokens,
        "batch_input_path": str(output_path),
        "batch_input_bytes": output_path.stat().st_size,
        "batch_input_sha256": file_sha256(output_path),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def estimate_batch(input_path: Path) -> dict[str, Any]:
    """Estimate request/candidate counts and coarse character-token volume."""
    requests = list(read_jsonl(input_path))
    input_characters = 0
    max_output_tokens = 0
    for request in requests:
        body = request.get("body", {})
        input_characters += len(str(body.get("instructions", "")))
        input_characters += len(str(body.get("input", "")))
        max_output_tokens += int(body.get("max_output_tokens", 0))
    return {
        "request_count": len(requests),
        "candidate_example_count": len(requests) * DEFAULT_VARIANTS_PER_SOURCE,
        "input_characters": input_characters,
        "approximate_input_tokens_at_four_characters_per_token": round(
            input_characters / 4
        ),
        "configured_maximum_output_tokens": max_output_tokens,
        "input_file_bytes": input_path.stat().st_size,
        "input_file_sha256": file_sha256(input_path),
        "note": "Token figures are planning bounds, not billing measurements.",
    }


def _openai_client() -> Any:
    """Construct the OpenAI client without loading secrets at import time."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for API stages; install the locked project environment."
        ) from exc
    return OpenAI()


def preflight_request(
    source_path: Path,
    output_path: Path,
    *,
    model: str,
    max_output_tokens: int,
    confirm_api_call: bool,
) -> dict[str, Any]:
    """Send exactly one paid request and validate its structured response."""
    if not confirm_api_call:
        raise RuntimeError("Preflight requires --confirm-api-call")
    source = next(read_jsonl(source_path), None)
    if source is None:
        raise ValueError("source snapshot is empty")
    client = _openai_client()
    response = client.responses.create(
        **response_body(
            source,
            model=model,
            max_output_tokens=max_output_tokens,
        )
    )
    returned = response.model_dump(mode="json")
    write_json(output_path, returned)
    returned_model = returned.get("model")
    if returned_model != model:
        raise RuntimeError(
            "Teacher model fallback is forbidden: "
            f"requested {model!r}, received {returned_model!r}"
        )
    response_text = getattr(response, "output_text", "")
    parsed = json.loads(response_text)
    canonical = canonical_from_variants(
        source,
        parsed,
        response_metadata={
            "request_id": returned.get("id"),
            "requested_model": model,
            "returned_model": returned_model,
            "prompt_version": PROMPT_VERSION,
            "usage": returned.get("usage"),
        },
    )
    return {
        "status": "passed",
        "response_path": str(output_path),
        "response_sha256": file_sha256(output_path),
        "canonical_rows": len(canonical),
        "requested_model": model,
        "returned_model": returned_model,
    }


def submit_batch(
    input_path: Path,
    manifest_path: Path,
    *,
    confirmation: str,
) -> dict[str, Any]:
    """Upload and submit a paid Batch only with the exact confirmation text."""
    if confirmation != PAID_CONFIRMATION:
        raise RuntimeError(
            f"Refusing paid submission; pass --confirm {PAID_CONFIRMATION}"
        )
    manifest = _read_manifest(manifest_path)
    expected_hash = manifest.get("batch_input_sha256")
    actual_hash = file_sha256(input_path)
    if expected_hash != actual_hash:
        raise RuntimeError("Batch input hash does not match its build manifest")
    if manifest.get("batch_id"):
        raise RuntimeError("Manifest already contains a batch_id")

    client = _openai_client()
    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"experiment": "reviewer-rerun-2026"},
    )
    manifest.update(
        {
            "stage": "submitted",
            "submitted_at": utc_now(),
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "batch_status": batch.status,
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def refresh_batch_status(manifest_path: Path) -> dict[str, Any]:
    """Retrieve a Batch once and persist its complete status metadata."""
    manifest = _read_manifest(manifest_path)
    batch_id = manifest.get("batch_id")
    if not batch_id:
        raise RuntimeError("Manifest does not contain a batch_id")
    batch = _openai_client().batches.retrieve(batch_id)
    batch_data = batch.model_dump(mode="json")
    manifest.update(
        {
            "stage": "status_refreshed",
            "status_checked_at": utc_now(),
            "batch_status": batch_data.get("status"),
            "request_counts": batch_data.get("request_counts"),
            "usage": batch_data.get("usage"),
            "output_file_id": batch_data.get("output_file_id"),
            "error_file_id": batch_data.get("error_file_id"),
            "batch_timestamps": {
                key: batch_data.get(key)
                for key in (
                    "created_at",
                    "in_progress_at",
                    "finalizing_at",
                    "completed_at",
                    "failed_at",
                    "expired_at",
                    "cancelled_at",
                )
            },
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def _write_download(content: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(content, "write_to_file"):
        content.write_to_file(output_path)
        return
    raw = getattr(content, "content", None)
    if isinstance(raw, bytes):
        output_path.write_bytes(raw)
        return
    text = getattr(content, "text", None)
    if isinstance(text, str):
        output_path.write_text(text, encoding="utf-8")
        return
    raise TypeError("Unsupported OpenAI file-content response")


def download_batch_files(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Download available Batch success and error JSONL files once."""
    manifest = _read_manifest(manifest_path)
    client = _openai_client()
    downloaded: dict[str, Any] = {}
    for kind, field_name in (("output", "output_file_id"), ("error", "error_file_id")):
        file_id = manifest.get(field_name)
        if not file_id:
            continue
        destination = output_dir / f"{manifest['batch_id']}.{kind}.jsonl"
        _write_download(client.files.content(file_id), destination)
        downloaded[kind] = {
            "file_id": file_id,
            "path": str(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        }
    manifest["stage"] = "downloaded"
    manifest["downloaded_at"] = utc_now()
    manifest["downloads"] = downloaded
    write_json(manifest_path, manifest)
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in manifest {path}")
    return value


def assembly_paths_from_batch_manifest(
    manifest_path: Path,
) -> tuple[Path, Path]:
    """Resolve and verify the downloaded output and submitted input paths."""
    manifest = _read_manifest(manifest_path)
    downloads = manifest.get("downloads")
    output = downloads.get("output") if isinstance(downloads, Mapping) else None
    if not isinstance(output, Mapping) or not isinstance(output.get("path"), str):
        raise RuntimeError(
            "Batch manifest has no downloaded output; refresh status and download first"
        )
    batch_output_path = Path(output["path"])
    expected_output_hash = output.get("sha256")
    if not batch_output_path.is_file():
        raise FileNotFoundError(
            f"Downloaded Batch output is missing: {batch_output_path}"
        )
    if expected_output_hash != file_sha256(batch_output_path):
        raise RuntimeError("Downloaded Batch output hash differs from its manifest")

    raw_input_path = manifest.get("batch_input_path")
    if not isinstance(raw_input_path, str):
        raise RuntimeError("Batch manifest has no batch_input_path")
    batch_input_path = Path(raw_input_path)
    expected_input_hash = manifest.get("batch_input_sha256")
    if not batch_input_path.is_file():
        raise FileNotFoundError(f"Submitted Batch input is missing: {batch_input_path}")
    if expected_input_hash != file_sha256(batch_input_path):
        raise RuntimeError("Submitted Batch input hash differs from its manifest")
    return batch_output_path, batch_input_path


def _response_output_text(body: Mapping[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks: list[str] = []
    output = body.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "".join(chunks)


def canonical_from_variants(
    source: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    response_metadata: Mapping[str, Any],
) -> list[CanonicalExample]:
    """Attach source and response provenance to three validated variants."""
    variants = parsed.get("variants")
    if not isinstance(variants, list) or len(variants) != DEFAULT_VARIANTS_PER_SOURCE:
        raise ValueError("Structured response must contain exactly three variants")
    variant_ids = [
        variant.get("variant_id")
        for variant in variants
        if isinstance(variant, Mapping)
    ]
    if sorted(variant_ids) != [1, 2, 3]:
        raise ValueError("Structured response variant IDs must be exactly 1, 2, 3")

    source_id = _required_source_text(source, "source_id")
    records: list[CanonicalExample] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("Every variant must be an object")
        variant_id = variant["variant_id"]
        value = {
            "schema_version": SCHEMA_VERSION,
            "example_id": f"{source_id}-v{variant_id:02d}",
            "source_id": source_id,
            "variant_id": variant_id,
            "source_question": _required_source_text(source, "question"),
            "source_solution": _required_source_text(source, "answer"),
            "synthetic_question": variant.get("synthetic_question"),
            "solution_steps": variant.get("solution_steps"),
            "final_answer": variant.get("final_answer"),
            "generation": dict(response_metadata),
        }
        records.append(CanonicalExample.from_mapping(value))
    return records


def assemble_batch_results(
    source_path: Path,
    batch_output_path: Path,
    canonical_output_path: Path,
    audit_output_path: Path,
    *,
    batch_input_path: Path | None = None,
    expected_model: str,
    allow_model_fallback: bool = False,
) -> dict[str, Any]:
    """Join Batch outputs by custom ID and emit canonical records plus audit."""
    all_sources = {
        _required_source_text(row, "source_id"): row for row in read_jsonl(source_path)
    }
    requested_ids: set[str] | None = None
    if batch_input_path is not None:
        requested_ids = set()
        for input_index, request in enumerate(read_jsonl(batch_input_path)):
            custom_id = request.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(
                    f"Batch input row {input_index} has an invalid custom_id"
                )
            if custom_id in requested_ids:
                raise ValueError(f"Duplicate custom_id in Batch input: {custom_id}")
            requested_ids.add(custom_id)
        if not requested_ids:
            raise ValueError("Batch input contains no requests")
        unknown_ids = sorted(requested_ids - set(all_sources))
        if unknown_ids:
            raise ValueError(
                "Batch input custom IDs are absent from the source snapshot: "
                + ", ".join(unknown_ids)
            )
        sources = {
            source_id: source
            for source_id, source in all_sources.items()
            if source_id in requested_ids
        }
    else:
        sources = all_sources
    canonical: list[CanonicalExample] = []
    audit: list[dict[str, Any]] = []
    seen_custom_ids: set[str] = set()
    successful_custom_ids: set[str] = set()

    for output_index, result in enumerate(read_jsonl(batch_output_path)):
        custom_id = result.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in sources:
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "unknown_custom_id",
                }
            )
            continue
        if custom_id in seen_custom_ids:
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "duplicate_custom_id",
                }
            )
            continue
        seen_custom_ids.add(custom_id)

        if result.get("error"):
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "batch_error",
                    "detail": result.get("error"),
                }
            )
            continue
        response = result.get("response")
        if not isinstance(response, Mapping):
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "missing_response",
                }
            )
            continue
        if response.get("status_code") != HTTP_OK:
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "http_status",
                    "status_code": response.get("status_code"),
                }
            )
            continue
        body = response.get("body")
        if not isinstance(body, Mapping) or body.get("status") != "completed":
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "response_incomplete",
                    "status": body.get("status") if isinstance(body, Mapping) else None,
                }
            )
            continue

        returned_model = body.get("model")
        if not allow_model_fallback and returned_model != expected_model:
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "unexpected_returned_model",
                    "expected_model": expected_model,
                    "returned_model": returned_model,
                }
            )
            continue

        try:
            parsed = json.loads(_response_output_text(body))
            records = canonical_from_variants(
                sources[custom_id],
                parsed,
                response_metadata={
                    "request_id": response.get("request_id") or body.get("id"),
                    "batch_result_id": result.get("id"),
                    "requested_model": expected_model,
                    "returned_model": returned_model,
                    "prompt_version": PROMPT_VERSION,
                    "usage": body.get("usage"),
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            audit.append(
                {
                    "output_index": output_index,
                    "custom_id": custom_id,
                    "reason": "invalid_structured_response",
                    "detail": str(exc),
                }
            )
            continue
        canonical.extend(records)
        successful_custom_ids.add(custom_id)

    missing_custom_ids = sorted(set(sources) - seen_custom_ids)
    retry_custom_ids = sorted(set(sources) - successful_custom_ids)
    audit.extend(
        {"custom_id": custom_id, "reason": "missing_batch_output"}
        for custom_id in missing_custom_ids
    )

    write_jsonl(canonical_output_path, (record.as_dict() for record in canonical))
    write_jsonl(audit_output_path, audit)
    manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "batch_output_path": str(batch_output_path),
        "batch_output_sha256": file_sha256(batch_output_path),
        "batch_input_path": str(batch_input_path) if batch_input_path else None,
        "batch_input_sha256": (
            file_sha256(batch_input_path) if batch_input_path else None
        ),
        "expected_model": expected_model,
        "allow_model_fallback": allow_model_fallback,
        "source_count": len(sources),
        "seen_response_count": len(seen_custom_ids),
        "successful_response_count": len(successful_custom_ids),
        "canonical_example_count": len(canonical),
        "successful_source_count": len(canonical) // DEFAULT_VARIANTS_PER_SOURCE,
        "audit_count": len(audit),
        "missing_custom_ids": missing_custom_ids,
        "retry_custom_ids": retry_custom_ids,
        "canonical_output_path": str(canonical_output_path),
        "canonical_output_sha256": file_sha256(canonical_output_path),
        "audit_output_path": str(audit_output_path),
        "audit_output_sha256": file_sha256(audit_output_path),
    }
    write_json(canonical_output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def merge_canonical_outputs(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    replace_sources_from_later: bool = False,
) -> dict[str, Any]:
    """Validate and deterministically merge canonical initial/retry outputs."""
    if not input_paths:
        raise ValueError("At least one canonical input is required")

    records_by_source_variant: dict[tuple[str, int], CanonicalExample] = {}
    seen_example_ids: dict[str, Path] = {}
    source_variant_paths: dict[tuple[str, int], Path] = {}
    source_paths: dict[str, Path] = {}
    replacement_sources: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for input_path in input_paths:
        input_records: list[CanonicalExample] = []
        input_example_ids: set[str] = set()
        input_source_variants: set[tuple[str, int]] = set()
        input_count = 0
        for raw in read_jsonl(input_path):
            record = CanonicalExample.from_mapping(raw)
            source_variant = (record.source_id, record.variant_id)
            if record.example_id in input_example_ids:
                raise ValueError(
                    f"Duplicate canonical example_id {record.example_id!r} in {input_path}"
                )
            if source_variant in input_source_variants:
                raise ValueError(
                    "Duplicate canonical source/variant "
                    f"{record.source_id!r}/{record.variant_id} in {input_path}"
                )
            input_example_ids.add(record.example_id)
            input_source_variants.add(source_variant)
            input_records.append(record)
            input_count += 1

        input_sources = {record.source_id for record in input_records}
        if replace_sources_from_later:
            for source_id in input_sources & set(source_paths):
                replacement_sources.add(source_id)
                for key in [
                    key for key in records_by_source_variant if key[0] == source_id
                ]:
                    previous = records_by_source_variant.pop(key)
                    seen_example_ids.pop(previous.example_id, None)
                    source_variant_paths.pop(key, None)
        for record in input_records:
            previous_path = seen_example_ids.get(record.example_id)
            if previous_path is not None:
                raise ValueError(
                    f"Duplicate canonical example_id {record.example_id!r} in "
                    f"{previous_path} and {input_path}"
                )
            source_variant = (record.source_id, record.variant_id)
            previous_path = source_variant_paths.get(source_variant)
            if previous_path is not None:
                raise ValueError(
                    "Duplicate canonical source/variant "
                    f"{record.source_id!r}/{record.variant_id} in "
                    f"{previous_path} and {input_path}"
                )
            records_by_source_variant[source_variant] = record
            seen_example_ids[record.example_id] = input_path
            source_variant_paths[source_variant] = input_path
            source_paths[record.source_id] = input_path
        inputs.append(
            {
                "path": str(input_path),
                "sha256": file_sha256(input_path),
                "canonical_rows": input_count,
            }
        )

    records = sorted(
        records_by_source_variant.values(),
        key=lambda record: (record.source_id, record.variant_id, record.example_id),
    )
    write_jsonl(output_path, (record.as_dict() for record in records))
    manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "inputs": inputs,
        "input_count": len(inputs),
        "replace_sources_from_later": replace_sources_from_later,
        "replacement_sources": sorted(replacement_sources),
        "canonical_example_count": len(records),
        "source_count": len({record.source_id for record in records}),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def build_rejection_retry_input(
    original_batch_input: Path,
    canonical_input: Path,
    rejection_audit: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Regenerate every source group with one or more filtered examples."""
    example_sources: dict[str, str] = {}
    for raw in read_jsonl(canonical_input):
        example_id = raw.get("example_id")
        source_id = raw.get("source_id")
        if isinstance(example_id, str) and isinstance(source_id, str):
            example_sources[example_id] = source_id

    rejected_sources: set[str] = set()
    unmapped_example_ids: list[str] = []
    for rejection in read_jsonl(rejection_audit):
        example_id = rejection.get("example_id")
        if not isinstance(example_id, str) or example_id not in example_sources:
            unmapped_example_ids.append(str(example_id))
            continue
        rejected_sources.add(example_sources[example_id])
    if unmapped_example_ids:
        raise ValueError(
            "Rejection audit contains examples absent from canonical input: "
            + ", ".join(sorted(unmapped_example_ids))
        )
    if not rejected_sources:
        raise ValueError("Rejection audit contains no source groups to retry")

    requests = [
        request
        for request in read_jsonl(original_batch_input)
        if request.get("custom_id") in rejected_sources
    ]
    found_sources = {request.get("custom_id") for request in requests}
    if found_sources != rejected_sources:
        missing = sorted(rejected_sources - found_sources)
        raise ValueError(f"Rejected source IDs not found in original input: {missing}")
    write_jsonl(output_path, requests)
    manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "stage": "filtered_source_retry_built",
        "original_batch_input": str(original_batch_input),
        "original_batch_input_sha256": file_sha256(original_batch_input),
        "canonical_input": str(canonical_input),
        "canonical_input_sha256": file_sha256(canonical_input),
        "rejection_audit": str(rejection_audit),
        "rejection_audit_sha256": file_sha256(rejection_audit),
        "retry_request_count": len(requests),
        "retry_source_ids": sorted(rejected_sources),
        "batch_input_path": str(output_path),
        "batch_input_sha256": file_sha256(output_path),
        "batch_input_bytes": output_path.stat().st_size,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def build_retry_input(
    original_batch_input: Path,
    assembly_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Copy only missing/failed source requests into a retry Batch file."""
    manifest = _read_manifest(assembly_manifest_path)
    retry_ids = set(
        manifest.get("retry_custom_ids", manifest.get("missing_custom_ids", []))
    )
    if not retry_ids:
        raise ValueError("Assembly manifest has no failed or missing custom IDs")
    requests = [
        request
        for request in read_jsonl(original_batch_input)
        if request.get("custom_id") in retry_ids
    ]
    found_ids = {request["custom_id"] for request in requests}
    if found_ids != retry_ids:
        missing = sorted(retry_ids - found_ids)
        raise ValueError(f"Retry IDs not found in original input: {missing}")
    write_jsonl(output_path, requests)
    retry_manifest = {
        "created_at": utc_now(),
        "protocol": protocol_identity(),
        "stage": "retry_built",
        "original_batch_input": str(original_batch_input),
        "original_batch_input_sha256": file_sha256(original_batch_input),
        "assembly_manifest": str(assembly_manifest_path),
        "retry_request_count": len(requests),
        "retry_custom_ids": sorted(retry_ids),
        "batch_input_path": str(output_path),
        "batch_input_sha256": file_sha256(output_path),
        "batch_input_bytes": output_path.stat().st_size,
    }
    write_json(output_path.with_suffix(".manifest.json"), retry_manifest)
    return retry_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot", help="pin and normalize GSM8K")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--dataset", default=DEFAULT_SOURCE_DATASET)
    snapshot.add_argument("--config", default=DEFAULT_SOURCE_CONFIG)
    snapshot.add_argument("--split", default=DEFAULT_SOURCE_SPLIT)
    snapshot.add_argument("--revision", default=DEFAULT_SOURCE_REVISION)
    snapshot.add_argument("--expected-rows", type=int, default=7473)
    snapshot.add_argument("--limit", type=int)

    build = commands.add_parser("build", help="build Batch JSONL offline")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    build.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    build.add_argument("--start-index", type=int, default=0)
    build.add_argument("--limit", type=int)

    estimate = commands.add_parser("estimate", help="estimate Batch volume offline")
    estimate.add_argument("--input", type=Path, required=True)

    preflight = commands.add_parser("preflight", help="send one confirmed API request")
    preflight.add_argument("--source", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    preflight.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    preflight.add_argument("--confirm-api-call", action="store_true")

    submit = commands.add_parser("submit", help="submit a confirmed paid Batch")
    submit.add_argument("--input", type=Path, required=True)
    submit.add_argument("--manifest", type=Path, required=True)
    submit.add_argument("--confirm", default="")

    status = commands.add_parser("status", help="retrieve Batch status once")
    status.add_argument("--manifest", type=Path, required=True)

    download = commands.add_parser("download", help="download Batch files")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)

    assemble = commands.add_parser("assemble", help="join Batch output by custom ID")
    assemble.add_argument("--source", type=Path, required=True)
    batch_source = assemble.add_mutually_exclusive_group(required=True)
    batch_source.add_argument("--batch-output", type=Path)
    batch_source.add_argument(
        "--batch-manifest",
        type=Path,
        help="infer and hash-check downloaded output and submitted input paths",
    )
    assemble.add_argument("--canonical-output", type=Path, required=True)
    assemble.add_argument("--audit-output", type=Path, required=True)
    assemble.add_argument(
        "--batch-input",
        type=Path,
        help="limit expected source IDs to the requests in this Batch input",
    )
    assemble.add_argument("--expected-model", default=DEFAULT_TEACHER_MODEL)
    assemble.add_argument("--allow-model-fallback", action="store_true")

    merge = commands.add_parser(
        "merge", help="merge validated canonical initial/retry outputs"
    )
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument(
        "--replace-sources-from-later",
        action="store_true",
        help="replace whole earlier source groups with records from later inputs",
    )

    retry = commands.add_parser("retry", help="build requests for missing outputs")
    retry.add_argument("--original-input", type=Path, required=True)
    retry.add_argument("--assembly-manifest", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)

    retry_rejected = commands.add_parser(
        "retry-rejected", help="build requests for source groups rejected by filtering"
    )
    retry_rejected.add_argument("--original-input", type=Path, required=True)
    retry_rejected.add_argument("--canonical-input", type=Path, required=True)
    retry_rejected.add_argument("--rejection-audit", type=Path, required=True)
    retry_rejected.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="validate canonical records")
    validate.add_argument("--input", type=Path, required=True)

    render = commands.add_parser("render", help="filter and render both arms")
    render.add_argument("--input", type=Path, required=True)
    render.add_argument("--output-root", type=Path, default=Path("data/reviewer_rerun"))
    selection = render.add_mutually_exclusive_group()
    selection.add_argument("--target-count", type=int, default=21250)
    selection.add_argument(
        "--all-accepted",
        action="store_true",
        help="render every accepted record instead of selecting an exact target",
    )
    render.add_argument("--min-solution-chars", type=int, default=120)
    render.add_argument("--max-solution-chars", type=int, default=2000)
    render.add_argument("--ngram-size", type=int, default=5)
    render.add_argument("--jaccard-threshold", type=float, default=0.85)
    render.add_argument("--validation-fraction", type=float, default=0.10)
    render.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicit generation stage."""
    args = _build_parser().parse_args(argv)
    result: dict[str, Any]
    if args.command == "snapshot":
        result = snapshot_source(
            args.output,
            dataset_name=args.dataset,
            dataset_config=args.config,
            split=args.split,
            revision=args.revision,
            expected_rows=args.expected_rows,
            limit=args.limit,
        )
    elif args.command == "build":
        result = build_batch_input(
            args.source,
            args.output,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            start_index=args.start_index,
            limit=args.limit,
        )
    elif args.command == "estimate":
        result = estimate_batch(args.input)
    elif args.command == "preflight":
        result = preflight_request(
            args.source,
            args.output,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            confirm_api_call=args.confirm_api_call,
        )
    elif args.command == "submit":
        result = submit_batch(
            args.input,
            args.manifest,
            confirmation=args.confirm,
        )
    elif args.command == "status":
        result = refresh_batch_status(args.manifest)
    elif args.command == "download":
        result = download_batch_files(args.manifest, args.output_dir)
    elif args.command == "assemble":
        batch_output_path = args.batch_output
        batch_input_path = args.batch_input
        if args.batch_manifest is not None:
            batch_output_path, inferred_input_path = assembly_paths_from_batch_manifest(
                args.batch_manifest
            )
            if batch_input_path is not None and batch_input_path != inferred_input_path:
                raise ValueError(
                    "--batch-input differs from the input recorded in --batch-manifest"
                )
            batch_input_path = inferred_input_path
        if batch_output_path is None:
            raise AssertionError("assemble requires a Batch output path")
        result = assemble_batch_results(
            args.source,
            batch_output_path,
            args.canonical_output,
            args.audit_output,
            batch_input_path=batch_input_path,
            expected_model=args.expected_model,
            allow_model_fallback=args.allow_model_fallback,
        )
    elif args.command == "merge":
        result = merge_canonical_outputs(
            args.input,
            args.output,
            replace_sources_from_later=args.replace_sources_from_later,
        )
    elif args.command == "retry":
        result = build_retry_input(
            args.original_input,
            args.assembly_manifest,
            args.output,
        )
    elif args.command == "retry-rejected":
        result = build_rejection_retry_input(
            args.original_input,
            args.canonical_input,
            args.rejection_audit,
            args.output,
        )
    elif args.command == "validate":
        records = [CanonicalExample.from_mapping(row) for row in read_jsonl(args.input)]
        result = {"status": "passed", "canonical_rows": len(records)}
    elif args.command == "render":
        result = build_paired_dataset(
            args.input,
            args.output_root,
            target_count=None if args.all_accepted else args.target_count,
            min_solution_chars=args.min_solution_chars,
            max_solution_chars=args.max_solution_chars,
            ngram_size=args.ngram_size,
            jaccard_threshold=args.jaccard_threshold,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        raise AssertionError(f"Unhandled command {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

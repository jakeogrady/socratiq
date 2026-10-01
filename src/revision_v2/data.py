"""Build and audit the revision-v2 training files.

The v2 files hold exactly the v1 canonical examples and the v1 source-group
train/validation split. Three things change, all in rendering:

1. a guiding question that ends in repeated question marks keeps one ("??" -> "?");
   the canonical data are unchanged;
2. training and validation rows are shuffled once with a recorded seed, and
   both arms use the same permutation, so row i is the same example in each;
3. nested 5,000- and 10,000-row training subsets are drawn by source problem
   (5k inside 10k inside the full 18,000 rows), with the full validation set.

Commands:
  build   write the files; with --write-manifest record their hashes (PI only),
          otherwise check them against the tracked manifest
  audit   run the pairing audit on written files
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from src.paired_dataset import CanonicalExample
from src.rerun_utils import (
    file_sha256,
    object_sha256,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)
from src.revision_v2.protocol import load_protocol, protocol_identity

MANIFEST_PATH = Path("configs/revision_v2/dataset_manifest.json")
ARMS = ("socratic", "non_socratic")
SPLITS = ("train", "valid")
TERMINAL_QUESTION_MARKS = re.compile(r"\?{2,}$")


class DataBuildError(RuntimeError):
    """Raised when the v2 data would not satisfy the protocol."""


def collapse_terminal_question_marks(question: str) -> str:
    """Collapse repeated question marks at the end of a guiding question."""
    return TERMINAL_QUESTION_MARKS.sub("?", question)


def render_answer_v2(record: CanonicalExample, *, socratic: bool) -> str:
    """Render one completion; reasoning and answer lines are byte-identical in both arms."""
    lines: list[str] = []
    for step in record.solution_steps:
        if socratic:
            lines.append(collapse_terminal_question_marks(step.guiding_question))
        lines.append(step.reasoning)
    lines.append(record.final_answer)
    return "\n".join(lines)


def render_row_v2(record: CanonicalExample, *, socratic: bool) -> dict[str, Any]:
    """Render one MLX-LM prompt/completion row with provenance identifiers."""
    return {
        "example_id": record.example_id,
        "source_id": record.source_id,
        "variant_id": record.variant_id,
        "question": record.synthetic_question,
        "answer": render_answer_v2(record, socratic=socratic),
    }


def shuffled(
    records: Sequence[CanonicalExample], seed_label: str
) -> list[CanonicalExample]:
    """Return records in one recorded, reproducible random order."""
    order = list(range(len(records)))
    random.Random(seed_label).shuffle(order)
    return [records[index] for index in order]


def nested_source_subsets(
    records: Sequence[CanonicalExample], sizes: Sequence[int], seed_label: str
) -> dict[int, list[str]]:
    """Select whole source groups so each subset has exactly the requested rows."""
    group_size = Counter(record.source_id for record in records)
    remaining = sorted(group_size)
    random.Random(seed_label).shuffle(remaining)
    selected: list[str] = []
    total = 0
    result: dict[int, list[str]] = {}
    for target in sorted(sizes):
        index = 0
        while total < target and index < len(remaining):
            source = remaining[index]
            if total + group_size[source] <= target:
                selected.append(source)
                total += group_size[source]
                remaining.pop(index)
            else:
                index += 1
        if total != target:
            msg = f"could not fill a subset of exactly {target} rows by source group"
            raise DataBuildError(msg)
        result[target] = list(selected)
    return result


def load_v1_split(
    protocol: dict[str, Any],
) -> tuple[list[CanonicalExample], list[CanonicalExample]]:
    """Return v1 train and validation records in v1 canonical order."""
    data = protocol["data"]
    canonical_path = Path(data["canonical_input"]["path"])
    if file_sha256(canonical_path) != data["canonical_input"]["sha256"]:
        msg = f"canonical input hash mismatch: {canonical_path}"
        raise DataBuildError(msg)
    records = [CanonicalExample.from_mapping(row) for row in read_jsonl(canonical_path)]
    v1_manifest = json.loads(
        Path(data["v1_dataset_manifest"]["path"]).read_text("utf-8")
    )
    validation_sources = set(v1_manifest["split"]["validation_source_ids"])
    train = [r for r in records if r.source_id not in validation_sources]
    valid = [r for r in records if r.source_id in validation_sources]
    expected = data["v1_dataset_manifest"]
    if (len(train), len(valid)) != (expected["train_rows"], expected["valid_rows"]):
        msg = f"v1 split sizes differ: {len(train)}/{len(valid)}"
        raise DataBuildError(msg)
    return train, valid


def build_rows(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return every v2 data directory's rows plus selection metadata (no I/O writes)."""
    data = protocol["data"]
    train, valid = load_v1_split(protocol)
    seed = data["row_order_seed"]
    train_order = shuffled(train, f"{seed}:train")
    valid_order = shuffled(valid, f"{seed}:valid")
    subset_spec = data["subsets"]
    subsets = nested_source_subsets(
        train, subset_spec["sizes"], f"{subset_spec['seed']}:subsets"
    )
    directories: dict[str, dict[str, list[CanonicalExample]]] = {
        "full": {"train": train_order, "valid": valid_order}
    }
    for size, sources in subsets.items():
        chosen = set(sources)
        directories[f"subset_{size // 1000}k"] = {
            "train": [r for r in train_order if r.source_id in chosen],
            "valid": valid_order,
        }
    return {
        "directories": directories,
        "subsets": subsets,
        "v1_train": train,
        "v1_valid": valid,
    }


def audit_pairing(
    socratic_rows: Sequence[dict[str, Any]],
    non_socratic_rows: Sequence[dict[str, Any]],
    canonical: dict[str, CanonicalExample],
) -> dict[str, Any]:
    """Row-by-row audit: same example per row, only guiding-question lines differ."""
    problems: list[str] = []
    collapsed = 0
    if len(socratic_rows) != len(non_socratic_rows):
        problems.append("arm row counts differ")
    for index, (soc, non) in enumerate(
        zip(socratic_rows, non_socratic_rows, strict=False)
    ):
        record = canonical.get(soc.get("example_id"))
        if record is None:
            problems.append(f"row {index}: unknown example_id")
            continue
        for key, expected in (
            ("example_id", record.example_id),
            ("source_id", record.source_id),
            ("variant_id", record.variant_id),
            ("question", record.synthetic_question),
        ):
            if soc.get(key) != expected or non.get(key) != expected:
                problems.append(f"row {index}: {key} differs")
        # Compare whole strings: a few reasoning fields contain internal newlines,
        # so a line-by-line comparison would misalign.
        questions = [
            collapse_terminal_question_marks(s.guiding_question)
            for s in record.solution_steps
        ]
        reasoning = [s.reasoning for s in record.solution_steps]
        collapsed += sum(
            s.guiding_question != q
            for s, q in zip(record.solution_steps, questions, strict=True)
        )
        if non["answer"] != "\n".join([*reasoning, record.final_answer]):
            problems.append(
                f"row {index}: non-Socratic answer is not reasoning + final answer"
            )
        interleaved = [
            part for pair in zip(questions, reasoning, strict=True) for part in pair
        ]
        if soc["answer"] != "\n".join([*interleaved, record.final_answer]):
            problems.append(
                f"row {index}: Socratic answer is not question/reasoning pairs + final answer"
            )
        if any(q.endswith("??") for q in questions):
            problems.append(f"row {index}: a guiding question still ends in '??'")
        if any(q in non["answer"] for q in questions):
            problems.append(
                f"row {index}: guiding question leaked into non-Socratic arm"
            )
    return {
        "status": "passed" if not problems else "failed",
        "rows": len(socratic_rows),
        "guiding_questions_collapsed": collapsed,
        "problems": problems[:50],
        "problem_count": len(problems),
    }


def audit_directories(
    built: dict[str, Any],
    rendered: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> dict[str, Any]:
    """Audit pairing, split integrity, ordering and subset nesting."""
    canonical = {r.example_id: r for r in [*built["v1_train"], *built["v1_valid"]]}
    report: dict[str, Any] = {"directories": {}}
    v1_train_ids = {r.example_id for r in built["v1_train"]}
    v1_valid_ids = {r.example_id for r in built["v1_valid"]}
    for name, splits in rendered.items():
        entry: dict[str, Any] = {}
        for split in SPLITS:
            pairing = audit_pairing(
                splits["socratic"][split], splits["non_socratic"][split], canonical
            )
            ids = [row["example_id"] for row in splits["socratic"][split]]
            pairing["unique_example_ids"] = len(set(ids)) == len(ids)
            entry[split] = pairing
        train_ids = {row["example_id"] for row in splits["socratic"]["train"]}
        valid_ids = {row["example_id"] for row in splits["socratic"]["valid"]}
        train_sources = {row["source_id"] for row in splits["socratic"]["train"]}
        valid_sources = {row["source_id"] for row in splits["socratic"]["valid"]}
        entry["train_subset_of_v1_train"] = train_ids <= v1_train_ids
        entry["valid_equals_v1_valid"] = valid_ids == v1_valid_ids
        entry["sources_in_both_splits"] = len(train_sources & valid_sources)
        if name == "full":
            entry["train_equals_v1_train"] = train_ids == v1_train_ids
            entry["order_changed_from_v1"] = [
                r["example_id"] for r in splits["socratic"]["train"]
            ] != [r.example_id for r in built["v1_train"]]
        else:
            full_sizes = Counter(r.source_id for r in built["v1_train"])
            subset_sizes = Counter(
                row["source_id"] for row in splits["socratic"]["train"]
            )
            entry["whole_source_groups"] = all(
                full_sizes[s] == n for s, n in subset_sizes.items()
            )
        report["directories"][name] = entry
    subsets = built["subsets"]
    sizes = sorted(subsets)
    report["subsets_nested"] = all(
        set(subsets[a]) <= set(subsets[b]) for a, b in pairwise(sizes)
    )
    report["subset_rows"] = {
        f"{s // 1000}k": len(rendered[f"subset_{s // 1000}k"]["socratic"]["train"])
        for s in sizes
    }
    passed = report["subsets_nested"] and all(
        e[split]["status"] == "passed"
        and e[split]["unique_example_ids"]
        and e["train_subset_of_v1_train"]
        and e["valid_equals_v1_valid"]
        and e["sources_in_both_splits"] == 0
        and e.get("train_equals_v1_train", True)
        and e.get("order_changed_from_v1", True)
        and e.get("whole_source_groups", True)
        for e in report["directories"].values()
        for split in SPLITS
    )
    report["status"] = "passed" if passed else "failed"
    return report


def build(
    output_root: Path, *, write_manifest: bool, manifest_path: Path = MANIFEST_PATH
) -> dict[str, Any]:
    """Write all v2 data files, audit them, and record or check their hashes."""
    protocol = load_protocol()
    built = build_rows(protocol)
    rendered: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for name, splits in built["directories"].items():
        rendered[name] = {
            arm: {
                split: [render_row_v2(r, socratic=arm == "socratic") for r in rows]
                for split, rows in splits.items()
            }
            for arm in ARMS
        }
    audit = audit_directories(built, rendered)
    if audit["status"] != "passed":
        raise DataBuildError(json.dumps(audit, indent=2)[:4000])
    files: dict[str, dict[str, Any]] = {}
    for name, arms in rendered.items():
        for arm, splits in arms.items():
            for split, rows in splits.items():
                path = output_root / name / arm / f"{split}.jsonl"
                write_jsonl(path, rows)
                files[path.as_posix()] = {
                    "rows": len(rows),
                    "sha256": file_sha256(path),
                }
    content = {
        "protocol_data_section_sha256": object_sha256(protocol["data"]),
        "row_order_seed": protocol["data"]["row_order_seed"],
        "subset_seed": protocol["data"]["subsets"]["seed"],
        "train_order_sha256": object_sha256(
            [r.example_id for r in built["directories"]["full"]["train"]]
        ),
        "valid_order_sha256": object_sha256(
            [r.example_id for r in built["directories"]["full"]["valid"]]
        ),
        "subset_source_ids": {
            f"{k // 1000}k": sorted(v) for k, v in built["subsets"].items()
        },
        "files": files,
    }
    if write_manifest:
        write_json(
            manifest_path,
            {
                "schema_version": "1.0",
                "created_at": utc_now(),
                "protocol": protocol_identity(),
                **content,
                "audit": audit,
            },
        )
        return {
            "status": "passed",
            "written_manifest": str(manifest_path),
            "audit": audit,
            "files": files,
        }
    tracked = json.loads(manifest_path.read_text("utf-8"))
    mismatches = sorted(
        path
        for path in set(tracked["files"]) | set(files)
        if tracked["files"].get(path) != files.get(path)
    )
    status = "passed" if not mismatches else "failed"
    return {
        "status": status,
        "mismatched_files": mismatches,
        "audit_status": audit["status"],
        "files": files,
    }


def verify_files(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Hash the existing data files against the tracked manifest without rewriting them."""
    tracked = json.loads(manifest_path.read_text("utf-8"))["files"]
    problems = [
        path
        for path, expected in sorted(tracked.items())
        if not Path(path).is_file() or file_sha256(Path(path)) != expected["sha256"]
    ]
    return {
        "status": "passed" if not problems else "failed",
        "files_checked": len(tracked),
        "problems": problems,
    }


def audit_written(output_root: Path) -> dict[str, Any]:
    """Re-read written files and audit them against the canonical data."""
    protocol = load_protocol()
    built = build_rows(protocol)
    rendered: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(dict)
    for name in built["directories"]:
        for arm in ARMS:
            rendered[name][arm] = {
                split: list(read_jsonl(output_root / name / arm / f"{split}.jsonl"))
                for split in SPLITS
            }
    return audit_directories(built, dict(rendered))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build")
    build_cmd.add_argument("--output-root", type=Path, default=Path("data/revision_v2"))
    build_cmd.add_argument("--write-manifest", action="store_true")
    audit_cmd = commands.add_parser("audit")
    audit_cmd.add_argument("--output-root", type=Path, default=Path("data/revision_v2"))
    audit_cmd.add_argument("--report", type=Path)
    commands.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and print a JSON summary."""
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build(args.output_root, write_manifest=args.write_manifest)
        printable = {k: v for k, v in result.items() if k not in {"files", "audit"}}
        printable["files"] = {k: v["rows"] for k, v in result["files"].items()}
    elif args.command == "verify":
        result = verify_files()
        printable = result
    else:
        result = audit_written(args.output_root)
        if args.report:
            write_json(args.report, result)
        printable = result
    print(json.dumps(printable, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

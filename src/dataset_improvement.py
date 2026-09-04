"""Strip trailing rhetorical questions from generated Socratic solutions.

The dataset generator often closes a solution with a rhetorical check addressed
to the reader ("Does the subtraction look consistent?") immediately before the
``####`` answer marker. Those sentences teach the model to end its reasoning
with a question instead of an answer, so they are removed before training.

Questions *within* the reasoning chain are the point of the Socratic format and
are preserved: only a question sentence sitting directly against the ``####``
marker is removed.

This transformation reproduces ``new_data_text/`` from ``new_data/`` for 21,240
of 21,250 examples (99.953%). The 10 exceptions are rows where the original
transformation truncated mid-word, e.g. train row 632 ends ``"= 0.75 ####"``
where the source reads ``"= 0.75W look consistent with a 25% decrease?"``.
Those are defects in the shipped data and are not reproduced here; run with
``--verify`` to list them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

INPUT_DIR = Path("new_data")
OUTPUT_DIR = Path("new_data_text")
SPLITS: tuple[str, ...] = ("train", "valid")

ANSWER_MARKER = "####"

# A trailing rhetorical question: opens after a sentence terminator followed by
# whitespace, carries no internal sentence punctuation, and runs to a question
# mark sitting immediately before the answer marker. Excluding "." from the body
# is what keeps decimals intact - "Does 260 x 0.05 match?" is left alone,
# matching the behaviour of the data the published models were trained on.
TRAILING_QUESTION = re.compile(r"(?<=[.!?])\s+[^.!?]*\?\s*(?=####)")

logger = logging.getLogger(__name__)


def clean_answer(answer: str) -> str:
    """Remove a rhetorical question sitting immediately before the answer marker."""
    text = answer.strip()
    if ANSWER_MARKER not in text:
        return text

    head, _, tail = text.rpartition(ANSWER_MARKER)
    head = TRAILING_QUESTION.sub(" ", head + ANSWER_MARKER)[: -len(ANSWER_MARKER)]
    return f"{head}{ANSWER_MARKER}{tail}".strip()


def format_example(question: str, answer: str) -> str:
    """Join a question and its cleaned answer into a single training string."""
    return f"Question: {question.strip()}\nAnswer: {clean_answer(answer)}"


def read_split(input_dir: Path, split: str) -> list[dict] | None:
    """Read one split, or return None when the file is absent."""
    path = input_dir / f"{split}.jsonl"
    if not path.exists():
        logger.warning("  %s not found - skipping", path)
        return None
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_split(input_dir: Path, output_dir: Path, split: str) -> tuple[int, int]:
    """Write the cleaned split, returning counts of examples written and modified."""
    rows = read_split(input_dir, split)
    if rows is None:
        return 0, 0

    written = 0
    modified = 0
    output_path = output_dir / f"{split}.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            if not question or not answer:
                continue
            if clean_answer(answer) != answer:
                modified += 1
            handle.write(json.dumps({"text": format_example(question, answer)}) + "\n")
            written += 1

    logger.info(
        "  %s: %d written, %d trailing questions removed", split, written, modified
    )
    return written, modified


def verify_split(input_dir: Path, reference_dir: Path, split: str) -> tuple[int, int]:
    """Compare this transformation against an existing split, returning (matched, total)."""
    rows = read_split(input_dir, split)
    reference_path = reference_dir / f"{split}.jsonl"
    if rows is None or not reference_path.exists():
        logger.warning("  cannot verify %s - missing input or reference", split)
        return 0, 0

    with reference_path.open(encoding="utf-8") as handle:
        reference = [json.loads(line)["text"] for line in handle if line.strip()]

    matched = 0
    for index, (row, expected) in enumerate(zip(rows, reference, strict=False)):
        produced = format_example(row.get("question", ""), row.get("answer", ""))
        if produced == expected:
            matched += 1
        else:
            logger.info("  %s row %d differs from %s", split, index, reference_path)

    total = min(len(rows), len(reference))
    logger.info(
        "  %s: %d/%d reproduced (%.3f%%)",
        split,
        matched,
        total,
        100 * matched / total if total else 0.0,
    )
    return matched, total


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare against the existing output instead of writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output directory if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    """Clean every split, or verify against an existing copy."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.verify:
        logger.info("Verifying %s -> %s\n", args.input, args.output)
        totals = [verify_split(args.input, args.output, split) for split in SPLITS]
        matched = sum(m for m, _ in totals)
        total = sum(t for _, t in totals)
        logger.info("\n%d/%d reproduced overall", matched, total)
        return 0

    if args.output.exists() and not args.force:
        logger.error(
            "%s already exists. This is the data the published models were "
            "trained on - pass --force to overwrite, or --verify to compare.",
            args.output,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info("Cleaning %s -> %s\n", args.input, args.output)
    removed = sum(convert_split(args.input, args.output, split)[1] for split in SPLITS)
    logger.info("\nTotal trailing questions removed: %d", removed)
    logger.info('Point a training config at:  data: "%s"', args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

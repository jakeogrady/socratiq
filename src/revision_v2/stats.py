"""Statistics fixed before the revision-v2 runs.

Primary family: Socratic minus non-Socratic, greedy, prompt P0, for three
models on four benchmarks (12 tests). Exact two-sided McNemar test, Newcombe
(1998) method-10 95% interval for the paired difference, Holm correction
across the 12. Base against fine-tuned comparisons use Wilson intervals only.

`primary-family` reads finished evaluation directories and writes one row per
test. It refuses to run unless all 24 required evaluations are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.rerun_utils import read_jsonl
from src.revision_v2.protocol import load_protocol

Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: doubled binomial tail, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def newcombe_paired(
    a: int, b: int, c: int, d: int, z: float = Z95
) -> tuple[float, float]:
    """Newcombe (1998) method 10 interval for p1 - p2 from a paired 2x2 table.

    a = both correct, b = left only, c = right only, d = both wrong.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    numerator = a * d - b * c
    phi_numerator = max(numerator - n / 2, 0) if numerator > 0 else numerator
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = phi_numerator / denominator if denominator else 0.0
    lower = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    upper = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return (p1 - p2) - lower, (p1 - p2) + upper


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_table(
    left: Sequence[bool], right: Sequence[bool]
) -> tuple[int, int, int, int]:
    """Return (both, left only, right only, neither) for aligned correctness flags."""
    if len(left) != len(right):
        msg = "paired flags must have the same length"
        raise ValueError(msg)
    a = sum(x and y for x, y in zip(left, right, strict=True))
    b = sum(x and not y for x, y in zip(left, right, strict=True))
    c = sum(y and not x for x, y in zip(left, right, strict=True))
    return a, b, c, len(left) - a - b - c


def _flags(evaluation_root: Path, condition: str, benchmark: str) -> list[bool]:
    run_dir = evaluation_root / condition / benchmark / "P0_greedy"
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    if manifest.get("status") != "completed":
        msg = f"{run_dir} is not a completed evaluation"
        raise RuntimeError(msg)
    rows = sorted(
        read_jsonl(run_dir / "predictions.jsonl"), key=lambda r: r["example_index"]
    )
    return [bool(r["correct"]) for r in rows]


def primary_family(
    evaluation_root: Path, peaks: dict[str, str]
) -> list[dict[str, Any]]:
    """Compute the 12 primary tests; peaks maps model key to its accepted run label."""
    family = load_protocol()["statistics"]["primary_family"]
    rows = []
    for model in family["models"]:
        for benchmark in family["benchmarks"]:
            left = _flags(
                evaluation_root, f"{model}_socratic_lr{peaks[model]}", benchmark
            )
            right = _flags(
                evaluation_root, f"{model}_non_socratic_lr{peaks[model]}", benchmark
            )
            a, b, c, d = paired_table(left, right)
            n = a + b + c + d
            low, high = newcombe_paired(a, b, c, d)
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "n": n,
                    "socratic_correct": a + b,
                    "non_socratic_correct": a + c,
                    "both_correct": a,
                    "socratic_only": b,
                    "non_socratic_only": c,
                    "both_wrong": d,
                    "difference_pp": 100 * (b - c) / n,
                    "newcombe95_low_pp": 100 * low,
                    "newcombe95_high_pp": 100 * high,
                    "mcnemar_exact_p": mcnemar_exact(b, c),
                }
            )
    for row, adjusted in zip(
        rows, holm([r["mcnemar_exact_p"] for r in rows]), strict=True
    ):
        row["holm_p"] = adjusted
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Write the primary-family table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["primary-family"])
    parser.add_argument(
        "--evaluation-root", type=Path, default=Path("runs/revision_v2/evaluation")
    )
    parser.add_argument(
        "--peaks",
        required=True,
        help='JSON, e.g. {"qwen3_0.6b": "8e-5", "qwen3_1.7b": "1e-4", "llama3.2_1b": "8e-5"}',
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = primary_family(args.evaluation_root, json.loads(args.peaks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tests to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

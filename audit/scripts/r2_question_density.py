"""Question density of Socratic solutions (R2 reviewer point 3).

Counts guiding questions per solution and the share of solution text taken by
question sentences, using one sentence-level rule so that questions on their
own line (revision v2) and questions embedded in a paragraph (original
submission) are measured the same way. A line-level count is reported beside it.

The original corpus is tracked on the remote branch track-synthetic-dataset
(commit 806d5a2, new_data/{train,valid}.jsonl). Extract it without a checkout:

  git show 806d5a2:new_data/train.jsonl > /tmp/original/train.jsonl
  git show 806d5a2:new_data/valid.jsonl > /tmp/original/valid.jsonl

Usage:
  python audit/scripts/r2_question_density.py
  python audit/scripts/r2_question_density.py --original /tmp/original/train.jsonl \
      /tmp/original/valid.jsonl --original-commit 806d5a2
"""

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "outputs" / "r2_question_density.json"
V2 = ROOT / "data" / "revision_v2" / "full" / "socratic"

# A question sentence is a run of text ending in one or more question marks,
# bounded by the previous sentence end or line break.
QUESTION = re.compile(r"[^.!?\n]*\?+")
MARKED = re.compile(r"^\s*####.*$", re.MULTILINE)


def solution_text(row, field):
    if field:
        value = row[field]
    elif "messages" in row:
        value = [m["content"] for m in row["messages"] if m.get("role") == "assistant"][-1]
    else:
        value = next(row[k] for k in ("answer", "completion", "solution", "text") if k in row)
    if not field and "answer" not in row and "completion" not in row and "messages" not in row:
        # Plain text rows: keep what follows the last solution header, if any.
        for header in ("<|im_start|>assistant", "Solution:", "Answer:"):
            if header in value:
                value = value.rsplit(header, 1)[1]
                break
    return MARKED.sub("", value).strip()


def measure(rows, field=None):
    counts, line_counts, shares, ends = [], [], [], []
    for row in rows:
        text = solution_text(row, field)
        questions = [q.strip() for q in QUESTION.findall(text) if q.strip()]
        body = len(re.sub(r"\s", "", text))
        qchars = sum(len(re.sub(r"\s", "", q)) for q in questions)
        counts.append(len(questions))
        line_counts.append(sum(1 for l in text.splitlines() if l.strip().endswith("?")))
        shares.append(qchars / body if body else 0.0)
        ends.append(text.rstrip().endswith("?"))
    n = len(counts)
    return {
        "solutions": n,
        "questions_per_solution_mean": round(statistics.mean(counts), 3),
        "questions_per_solution_median": statistics.median(counts),
        "questions_per_solution_min": min(counts),
        "questions_per_solution_max": max(counts),
        "share_with_no_question": round(sum(c == 0 for c in counts) / n, 4),
        "share_with_3_or_more": round(sum(c >= 3 for c in counts) / n, 4),
        "share_ending_with_question": round(sum(ends) / n, 4),
        "question_lines_per_solution_mean": round(statistics.mean(line_counts), 3),
        "question_char_share_mean": round(statistics.mean(shares), 4),
        "question_char_share_pooled": None,
    }


def pooled_share(rows, field=None):
    q = b = 0
    for row in rows:
        text = solution_text(row, field)
        b += len(re.sub(r"\s", "", text))
        q += sum(len(re.sub(r"\s", "", s)) for s in QUESTION.findall(text))
    return round(q / b, 4)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, nargs="+", help="original corpus splits (jsonl)")
    ap.add_argument("--original-field", help="field holding the solution text")
    ap.add_argument("--original-commit", help="git commit the original files came from")
    args = ap.parse_args()

    result = {
        "rule": {
            "question": "text ending in '?' after the previous '.', '!', '?' or line break",
            "solution_text": "the rendered solution with the '####' answer line removed",
            "char_share": "non-whitespace characters in question sentences / non-whitespace solution characters",
            "line_count": "lines whose stripped text ends in '?'",
        },
        "revision_v2": {},
        "original_submission": None,
    }
    for split in ("train", "valid"):
        rows = load(V2 / f"{split}.jsonl")
        stats = measure(rows)
        stats["question_char_share_pooled"] = pooled_share(rows)
        stats["source"] = str((V2 / f"{split}.jsonl").relative_to(ROOT))
        result["revision_v2"][split] = stats

    if args.original:
        result["original_submission"] = {"commit": args.original_commit}
        for path in args.original:
            rows = load(path)
            stats = measure(rows, args.original_field)
            stats["question_char_share_pooled"] = pooled_share(rows, args.original_field)
            stats["source"] = path.name
            result["original_submission"][path.stem] = stats
    else:
        result["original_submission"] = {
            "status": "not supplied",
            "note": "extract new_data/train.jsonl and new_data/valid.jsonl from commit 806d5a2 "
            "and pass them with --original (see the usage above)",
        }

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

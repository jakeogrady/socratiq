"""Teacher self-corrections left in the revision-v2 solutions (Supplementary Section S1).

A phrase search over the reasoning text, checked by hand. Every match was read.
Two matches are ordinary problem wording and are excluded. Two examples found
by a broader search ("however there are actually four closet doors") are added.
The reasoning text is the same in both arms, so the non-Socratic rendering is
searched. Other self-corrections may use wording that the search misses, so the
count is a lower bound.

Usage:
  python audit/scripts/r2_self_corrections.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "revision_v2" / "full" / "non_socratic"
OUT = ROOT / "audit" / "outputs" / "r2_self_corrections.json"

PHRASES = re.compile(
    r"(\bWait\b(?! time)|\bwait[,;:]? (?:ensure|recompute|check|that)|\bOops\b|\bApologies\b"
    r"|\badjust:|\bCorrection:|\bcorrecting\b|\brecompute\b|\bRecalculate\b"
    r"|reasoning is invalid|not allowed|must keep integer|intended final"
    r"|intended (?:answer|order|difficulty|integer)|doesn't match|conflicts with"
    r"|is incorrect|that's incorrect|need integer|not an integer|not integer"
    r"|break integer|choose interpretation|correct ones are|Actually x|\? No\.)"
)
# Read by hand: ordinary wording about the problem, with no correction.
EXCLUDED = {
    "gsm8k-train-004016-v02": "count after correcting for duplicates (problem wording)",
    "gsm8k-train-000766-v02": "without correcting overlap (problem wording)",
}
# Read by hand: a wrong step followed by a fix, without any of the phrases.
ADDED = {
    "gsm8k-train-002588-v02": "three closet doors, then 'actually four'",
    "gsm8k-train-002588-v03": "three bedroom doors, then 'actually four'",
}


def main():
    result = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "splits": {}}
    total = rows_total = 0
    for split in ("train", "valid"):
        rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]
        found = {}
        for row in rows:
            match = PHRASES.search(row["answer"])
            if (match and row["example_id"] not in EXCLUDED) or row["example_id"] in ADDED:
                text = row["answer"]
                start = match.start() if match else 0
                found[row["example_id"]] = text[max(0, start - 60) : start + 100].replace("\n", " / ")
        result["splits"][split] = {"rows": len(rows), "with_self_correction": len(found), "examples": found}
        total += len(found)
        rows_total += len(rows)
    result["excluded_after_reading"] = EXCLUDED
    result["added_after_reading"] = ADDED
    result["total"] = {"rows": rows_total, "with_self_correction": total, "share_pct": round(100 * total / rows_total, 2)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["total"]), {s: v["with_self_correction"] for s, v in result["splits"].items()})


if __name__ == "__main__":
    main()

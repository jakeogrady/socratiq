from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.rerun_utils import file_sha256, write_json, write_jsonl
from src.rescore_predictions import (
    RescoringError,
    exact_mcnemar_p,
    paired_comparisons,
    rescore_evaluations,
)


def _write_evaluation(root: Path, *, response: str = "work\n#### 8\n?") -> Path:
    run = root / "evaluation" / "qwen" / "gsm8k" / "greedy"
    predictions = run / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "experiment_id": "qwen-socratic",
                "model": "model-id",
                "model_revision": "a" * 40,
                "adapter": {"weights_sha256": "adapter-hash"},
                "benchmark": "gsm8k",
                "example_id": "gsm8k-test-000000",
                "example_index": 0,
                "reference_answer": "8",
                "samples": [{"response": response}],
                "greedy_tie_break": None,
                "predicted_answer": None,
                "valid_answer": False,
                "correct": False,
            }
        ],
    )
    write_json(
        run / "manifest.json",
        {
            "status": "completed",
            "configuration": {
                "evaluator": "evaluate_v2",
                "experiment_id": "qwen-socratic",
                "model": "model-id",
                "model_revision": "a" * 40,
                "benchmark": {"key": "gsm8k", "expected_rows": 1},
                "mode": "greedy",
                "samples": 1,
            },
            "prediction_path": "/unavailable/source/predictions.jsonl",
            "prediction_sha256": file_sha256(predictions),
        },
    )
    return run


class RescorePredictionsTests(unittest.TestCase):
    def test_exact_mcnemar_uses_discordant_pairs(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(2, 0), 0.5)
        self.assertAlmostEqual(exact_mcnemar_p(1, 9), 0.021484375)
        self.assertGreater(exact_mcnemar_p(659, 660), 0.9)

    def test_paired_comparisons_keep_decoding_modes_separate(self) -> None:
        decisions = {
            ("qwen3-0.6b-socratic", "gsm8k", "greedy"): {"one": True},
            ("qwen3-0.6b-non-socratic", "gsm8k", "greedy"): {"one": False},
            ("qwen3-0.6b-socratic", "gsm8k", "self_consistency"): {"one": False},
            ("qwen3-0.6b-non-socratic", "gsm8k", "self_consistency"): {"one": True},
        }
        rows = [
            row
            for row in paired_comparisons(decisions)
            if row["comparison_id"] == "qwen3-0.6b-socratic-vs-non-socratic"
        ]
        self.assertEqual({row["mode"] for row in rows}, {"greedy", "self_consistency"})

    def test_recovers_trailing_question_mark_without_overwriting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_evaluation(root)
            output = root / "rescored"

            result = rescore_evaluations(root / "evaluation", output)

            self.assertEqual(result["run_count"], 1)
            source = json.loads((run / "predictions.jsonl").read_text())
            self.assertFalse(source["valid_answer"])
            rescored_path = output / "runs/qwen/gsm8k/greedy/rescored.jsonl"
            rescored = json.loads(rescored_path.read_text())
            self.assertEqual(rescored["corrected_predicted_answer"], "8")
            self.assertTrue(rescored["corrected_valid_answer"])
            self.assertTrue(rescored["corrected_correct"])
            self.assertFalse(rescored["terminal_valid_answer"])
            self.assertTrue(rescored["format_only_recovery"])
            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["corrected_correct_answers"], "1")
            self.assertEqual(summary["terminal_correct_answers"], "0")

    def test_rejects_tampered_source_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_evaluation(root)
            with (run / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(RescoringError, "hash mismatch"):
                rescore_evaluations(root / "evaluation", root / "rescored")

    def test_refuses_to_overwrite_an_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_evaluation(root)
            output = root / "rescored"
            output.mkdir()
            with self.assertRaisesRegex(RescoringError, "refusing to overwrite"):
                rescore_evaluations(root / "evaluation", output)


if __name__ == "__main__":
    unittest.main()

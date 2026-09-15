from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.rerun_utils import write_json, write_jsonl
from src.summarize_results import (
    ReportingError,
    generate_reports,
    summarize_prediction_rows,
    wilson_interval,
)


def prediction(index: int, correct: bool) -> dict:
    return {
        "experiment_id": "qwen-base",
        "model": "model-id",
        "benchmark": "svamp",
        "example_id": f"svamp-test-{index:06d}",
        "example_index": index,
        "valid_answer": True,
        "correct": correct,
        "samples": [{"generation_seconds": 1.5}],
        "greedy_tie_break": None,
    }


class WilsonTests(unittest.TestCase):
    def test_svamp_interval_uses_actual_denominator(self) -> None:
        lower, upper = wilson_interval(127, 300)
        self.assertAlmostEqual(lower, 0.369, places=3)
        self.assertAlmostEqual(upper, 0.480, places=3)

    def test_prediction_summary_rejects_incorrect_expected_count(self) -> None:
        rows = [prediction(index, index < 2) for index in range(3)]
        with self.assertRaisesRegex(ReportingError, "expected 300"):
            summarize_prediction_rows(rows, expected_rows=300, require_full=True)

    def test_prediction_summary_uses_observed_rows(self) -> None:
        rows = [prediction(index, index < 2) for index in range(3)]
        summary = summarize_prediction_rows(
            rows,
            expected_rows=300,
            require_full=False,
        )
        self.assertEqual(summary["observed_rows"], 3)
        self.assertEqual(summary["correct_answers"], 2)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)

    def test_prediction_summary_rejects_mixed_model_revisions(self) -> None:
        rows = [prediction(0, True), prediction(1, False)]
        rows[0]["model_revision"] = "a" * 40
        rows[1]["model_revision"] = "b" * 40
        with self.assertRaisesRegex(ReportingError, "homogeneous"):
            summarize_prediction_rows(rows, expected_rows=2, require_full=True)


class ReportGenerationTests(unittest.TestCase):
    def test_reports_are_generated_from_manifests_and_raw_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_run = root / "evaluation" / "run"
            train_run = root / "training" / "run"
            predictions = eval_run / "predictions.jsonl"
            write_jsonl(predictions, [prediction(0, True), prediction(1, False)])
            write_json(
                eval_run / "manifest.json",
                {
                    "configuration": {
                        "benchmark": {"expected_rows": 2},
                        "mode": "greedy",
                        "samples": 1,
                        "adapter": None,
                    },
                    "prediction_path": str(predictions),
                    "completed_at": "2026-09-14T00:00:00+00:00",
                },
            )
            write_json(
                train_run / "manifest.json",
                {
                    "experiment_id": "qwen-tuned",
                    "status": "completed",
                    "configuration": {"model": "model-id"},
                    "environment": {
                        "hardware": {"model": "Mac", "chip": "M4", "memory_bytes": "1"},
                        "packages": {"mlx": "1", "mlx-lm": "2"},
                        "python": "3.13",
                    },
                    "model": {"resolved_revision": "abc"},
                    "elapsed_seconds": 10,
                    "peak_process_bytes": 100,
                    "training_metrics": {
                        "peak_mlx_memory_gb": 5.0,
                        "trainable_parameters": {
                            "trainable_count": 10,
                            "total_count": 100,
                        },
                    },
                    "adapter": {"path": "adapter", "bytes": 12, "sha256": "hash"},
                },
            )
            output = root / "results"
            manifest = generate_reports(
                root / "evaluation",
                root / "training",
                output,
            )
            self.assertEqual(manifest["evaluation_runs"], 1)
            self.assertEqual(manifest["training_runs"], 1)
            self.assertTrue((output / "summaries" / "summary.csv").exists())
            self.assertIn(
                "qwen-base",
                (output / "reproducibility" / "reproducibility.md").read_text(),
            )
            self.assertIn(
                "SVAMP: zero-shot; all 300 rows",
                (output / "reproducibility" / "reproducibility.md").read_text(),
            )
            self.assertIn(
                "mlx-community/Qwen3-0.6B-bf16",
                (output / "reproducibility" / "reproducibility.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()

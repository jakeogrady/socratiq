from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = REPOSITORY_ROOT / "scripts/run_training_matrix.sh"
EVALUATION_SCRIPT = REPOSITORY_ROOT / "scripts/run_evaluation_matrix.sh"
CLEAN_EVALUATION_SCRIPT = REPOSITORY_ROOT / "scripts/run_clean_evaluation_matrix.sh"
GSMHARD_EVALUATION_SCRIPT = REPOSITORY_ROOT / "scripts/run_gsmhard_evaluation_matrix.sh"


def run_plan(script: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603 - repository-owned fixed script
        [str(script), *arguments, "--plan"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout


class MatrixScriptTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        for script in (
            TRAINING_SCRIPT,
            EVALUATION_SCRIPT,
            CLEAN_EVALUATION_SCRIPT,
            GSMHARD_EVALUATION_SCRIPT,
        ):
            with self.subTest(script=script.name):
                completed = subprocess.run(  # noqa: S603
                    ["/bin/sh", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_mandatory_training_plan_has_three_isolated_runs(self) -> None:
        output = run_plan(TRAINING_SCRIPT, "--full", "--mandatory")
        self.assertEqual(output.count("PLAN:"), 3)
        self.assertIn("qwen3-0.6b-socratic", output)
        self.assertIn("qwen3-0.6b-non-socratic", output)
        self.assertIn("llama3.2-1b-socratic", output)
        self.assertNotIn("qwen3-1.7b-socratic", output)

    def test_optional_training_plan_adds_the_qwen_17_pair(self) -> None:
        output = run_plan(TRAINING_SCRIPT, "--include-qwen-1.7b")
        self.assertEqual(output.count("PLAN:"), 5)
        self.assertIn("qwen3-1.7b-socratic", output)
        self.assertIn("qwen3-1.7b-non-socratic", output)

    def test_mandatory_greedy_plan_has_fifteen_full_runs(self) -> None:
        output = run_plan(EVALUATION_SCRIPT, "--full", "--greedy", "--mandatory")
        self.assertEqual(output.count("PLAN:"), 15)
        self.assertIn("--mode greedy --samples 1 --temperature 0.0", output)
        self.assertIn("--top-p 1.0 --top-k 0 --max-tokens 512 --seed 42", output)
        self.assertNotIn("--limit", output)

    def test_smoke_and_sc5_plans_are_separate_from_official_outputs(self) -> None:
        greedy = run_plan(EVALUATION_SCRIPT, "--smoke", "--greedy")
        sc5 = run_plan(EVALUATION_SCRIPT, "--smoke", "--sc5")
        self.assertEqual(greedy.count("--limit 20"), 15)
        self.assertIn("runs/reviewer_rerun/smoke/evaluation", greedy)
        self.assertIn("--mode self_consistency --samples 5", sc5)
        self.assertIn("--temperature 0.7 --top-p 0.95 --top-k 20", sc5)

    def test_optional_evaluation_plan_adds_nine_runs(self) -> None:
        output = run_plan(EVALUATION_SCRIPT, "--include-qwen-1.7b")
        self.assertEqual(output.count("PLAN:"), 24)
        self.assertIn("qwen3-1.7b-base", output)
        self.assertIn("qwen3-1.7b-socratic", output)
        self.assertIn("qwen3-1.7b-non-socratic", output)

    def test_clean_matrix_uses_isolated_roots_and_final_adapters(self) -> None:
        full = run_plan(
            CLEAN_EVALUATION_SCRIPT,
            "--full",
            "--greedy",
            "--include-qwen-1.7b",
        )
        smoke = run_plan(
            CLEAN_EVALUATION_SCRIPT,
            "--smoke",
            "--sc5",
            "--include-qwen-1.7b",
        )
        self.assertEqual(full.count("PLAN:"), 24)
        self.assertEqual(smoke.count("PLAN:"), 24)
        self.assertIn("runs/reviewer_rerun/evaluation_clean_v1/", full)
        self.assertNotIn("runs/reviewer_rerun/evaluation/", full)
        self.assertIn("runs/reviewer_rerun/evaluation_clean_v1_smoke/", smoke)
        self.assertIn(
            "runs/reviewer_rerun/training/qwen3_0.6b_socratic/adapter",
            smoke,
        )
        self.assertNotIn("smoke_adapter", smoke)
        self.assertIn("--limit 20", smoke)

    def test_gsmhard_matrix_is_complete_and_isolated(self) -> None:
        full = run_plan(GSMHARD_EVALUATION_SCRIPT, "--full", "--greedy")
        smoke = run_plan(GSMHARD_EVALUATION_SCRIPT, "--smoke", "--sc5")

        self.assertEqual(full.count("PLAN:"), 8)
        self.assertEqual(smoke.count("PLAN:"), 8)
        self.assertEqual(full.count("--benchmark gsm_hard"), 8)
        self.assertEqual(full.count("--protocol-extension"), 8)
        self.assertIn("runs/reviewer_rerun/evaluation_gsmhard_v1/", full)
        self.assertNotIn("runs/reviewer_rerun/evaluation_clean_v1/", full)
        self.assertIn("runs/reviewer_rerun/evaluation_gsmhard_v1_smoke/", smoke)
        self.assertEqual(smoke.count("--limit 20"), 8)
        self.assertEqual(smoke.count("--mode self_consistency --samples 5"), 8)
        self.assertIn(
            "runs/reviewer_rerun/training/qwen3_1.7b_non_socratic/adapter",
            smoke,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluate_v2 import (
    BENCHMARKS,
    MODEL_REVISIONS,
    EvaluationError,
    answers_equal,
    build_prompt,
    derive_seed,
    extract_marked_number,
    extract_terminal_marked_number,
    majority_vote,
    materialize_evaluation_model,
    normalize_numeric,
    normalize_reference_answer,
    render_model_prompt,
    requested_model_revision,
)


class AnswerExtractionTests(unittest.TestCase):
    def test_uses_last_marked_answer(self) -> None:
        self.assertEqual(extract_marked_number("#### 7\nCorrection: #### 8.00"), "8")

    def test_has_no_unmarked_fallback(self) -> None:
        self.assertIsNone(extract_marked_number("The calculation uses 5 and gives 9."))

    def test_accepts_text_after_the_last_marked_number(self) -> None:
        self.assertEqual(extract_marked_number("#### 8 apples"), "8")

    def test_accepts_trailing_question_mark_from_socratic_output(self) -> None:
        self.assertEqual(extract_marked_number("work\n#### 8\n?"), "8")

    def test_terminal_diagnostic_rejects_trailing_question_mark(self) -> None:
        self.assertIsNone(extract_terminal_marked_number("work\n#### 8\n?"))
        self.assertEqual(extract_terminal_marked_number("work\n#### 8"), "8")

    def test_normalizes_commas_signs_and_decimals(self) -> None:
        self.assertEqual(extract_marked_number("#### +1,200.500"), "1200.5")
        self.assertEqual(extract_marked_number("#### -0.00"), "0")
        self.assertEqual(normalize_numeric("8.000"), "8")

    def test_normalizes_integers_at_or_above_decimal_context_precision(self) -> None:
        value = "10000000000000000000000000000"
        self.assertEqual(normalize_numeric(value), value)
        self.assertEqual(extract_marked_number(f"#### {value}"), value)

    def test_reference_accepts_marked_or_bare_number(self) -> None:
        self.assertEqual(normalize_reference_answer("work #### 8"), "8")
        self.assertEqual(normalize_reference_answer(8.0), "8")
        self.assertIsNone(normalize_reference_answer("answer is 8"))

    def test_decimal_equivalence(self) -> None:
        self.assertTrue(answers_equal("8", "8.00"))
        self.assertFalse(answers_equal("8", "9"))


class VotingAndSeedTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        self.assertEqual(majority_vote(["4", "5", "4"]), ("4", False))

    def test_tie_prefers_declared_greedy_answer(self) -> None:
        self.assertEqual(
            majority_vote(["10", "2", "10", "2", "3"], greedy_answer="10"),
            ("10", True),
        )

    def test_tie_without_greedy_is_numeric_and_order_independent(self) -> None:
        self.assertEqual(majority_vote(["10", "2"]), ("2", True))
        self.assertEqual(majority_vote(["2", "10"]), ("2", True))

    def test_seed_derivation_is_stable_and_sample_specific(self) -> None:
        first = derive_seed(42, "exp", "gsm8k", 3, 0)
        self.assertEqual(first, derive_seed(42, "exp", "gsm8k", 3, 0))
        self.assertNotEqual(first, derive_seed(42, "exp", "gsm8k", 3, 1))


class PromptTests(unittest.TestCase):
    def test_prompt_contains_only_supplied_train_shots_and_target(self) -> None:
        shots = [{"question": "TRAIN SHOT", "answer": "work #### 1"}]
        prompt = build_prompt(
            "TARGET TEST QUESTION",
            shots,
            question_column="question",
            answer_column="answer",
        )
        self.assertIn("TRAIN SHOT", prompt)
        self.assertIn("TARGET TEST QUESTION", prompt)
        self.assertTrue(prompt.endswith("Question: TARGET TEST QUESTION\nAnswer:"))
        self.assertIn("#### <number>", prompt)

    def test_benchmark_registry_has_full_counts_and_train_only_gsm_shots(self) -> None:
        self.assertEqual(BENCHMARKS["gsm8k"].expected_rows, 1319)
        self.assertEqual(BENCHMARKS["multiarith"].expected_rows, 180)
        self.assertEqual(BENCHMARKS["svamp"].expected_rows, 300)
        self.assertTrue(
            all(
                spec.revision and len(spec.revision) == 40
                for spec in BENCHMARKS.values()
            )
        )
        self.assertEqual(BENCHMARKS["gsm8k"].few_shot_split, "train")
        self.assertNotEqual(
            BENCHMARKS["gsm8k"].few_shot_split,
            BENCHMARKS["gsm8k"].target_split,
        )

    def test_model_registry_uses_immutable_revisions(self) -> None:
        self.assertEqual(len(MODEL_REVISIONS), 3)
        self.assertTrue(
            all(len(revision) == 40 for revision in MODEL_REVISIONS.values())
        )
        for model, revision in MODEL_REVISIONS.items():
            self.assertEqual(requested_model_revision(model), revision)

    def test_unregistered_or_mutable_remote_revision_is_rejected(self) -> None:
        with self.assertRaises(EvaluationError):
            requested_model_revision("example/unregistered-model")
        with self.assertRaises(EvaluationError):
            requested_model_revision("example/unregistered-model", "main")

    def test_local_model_is_content_hashed_and_rejects_remote_revision(self) -> None:
        with TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "weights.safetensors").write_bytes(b"frozen weights")
            materialized, provenance = materialize_evaluation_model(str(model_path))
            self.assertEqual(materialized, model_path.resolve())
            self.assertIsNone(provenance["requested_revision"])
            self.assertIsNone(provenance["resolved_revision"])
            self.assertEqual(len(provenance["content_sha256"]), 64)
            with self.assertRaises(EvaluationError):
                requested_model_revision(str(model_path), "0" * 40)

    def test_chat_template_records_explicit_thinking_setting(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json_like(messages, kwargs)

        rendered = render_model_prompt(FakeTokenizer(), "prompt", enable_thinking=False)
        self.assertIn("'enable_thinking': False", rendered)

    def test_unsupported_requested_thinking_fails(self) -> None:
        class OldTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "rendered"

        with self.assertRaises(EvaluationError):
            render_model_prompt(OldTokenizer(), "prompt", enable_thinking=True)


def json_like(messages, kwargs) -> str:
    return f"{messages!r} {kwargs!r}"


if __name__ == "__main__":
    unittest.main()

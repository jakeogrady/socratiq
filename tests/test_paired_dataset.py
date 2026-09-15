from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.paired_dataset import (
    CanonicalExample,
    DatasetValidationError,
    build_paired_dataset,
    filter_and_deduplicate,
    generate_ngrams,
    jaccard_similarity,
    render_answer,
    render_record,
    split_by_source,
    validate_rendered_pair,
)
from src.rerun_utils import read_jsonl, write_jsonl


def canonical_mapping(
    example_id: str,
    source_id: str,
    variant_id: int,
    *,
    question: str | None = None,
) -> dict:
    return {
        "schema_version": "1.1",
        "example_id": example_id,
        "source_id": source_id,
        "variant_id": variant_id,
        "source_question": "A source question?",
        "source_solution": "The source worked solution is 9. #### 9",
        "synthetic_question": question
        or f"{example_id} has a distinct arithmetic word problem; what is its total?",
        "solution_steps": [
            {
                "guiding_question": "What quantity should be calculated first?",
                "reasoning": "The first quantity is 12 - 3 = 9, which remains non-negative.",
            },
            {
                "guiding_question": "How does that determine the requested total?",
                "reasoning": (
                    "The requested total is 9 because the preceding subtraction "
                    "directly gives the number of items that remain."
                ),
            },
        ],
        "final_answer": "#### 9",
        "generation": {"requested_model": "teacher", "returned_model": "teacher"},
    }


class CanonicalValidationTests(unittest.TestCase):
    def test_guiding_question_must_be_separate_question(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"][0]["guiding_question"] = "Calculate the first value."
        with self.assertRaisesRegex(DatasetValidationError, "must end"):
            CanonicalExample.from_mapping(row)

    def test_final_answer_must_be_one_positive_integer(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["final_answer"] = "The answer is 9"
        with self.assertRaisesRegex(DatasetValidationError, "positive integer"):
            CanonicalExample.from_mapping(row)

    def test_synthetic_problem_must_contain_a_direct_question(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["synthetic_question"] = "This arithmetic story never asks for a result."
        with self.assertRaisesRegex(DatasetValidationError, "direct question"):
            CanonicalExample.from_mapping(row)

    def test_solution_step_count_must_be_between_two_and_six(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"] = row["solution_steps"][:1]
        with self.assertRaisesRegex(DatasetValidationError, "between 2 and 6"):
            CanonicalExample.from_mapping(row)

        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"].extend(
            {
                "guiding_question": f"What distinct quantity is calculated at step {i}?",
                "reasoning": (
                    f"Step {i} records a distinct intermediate quantity while keeping "
                    "the complete declarative reasoning safely above sixty characters."
                ),
            }
            for i in range(3, 8)
        )
        with self.assertRaisesRegex(DatasetValidationError, "between 2 and 6"):
            CanonicalExample.from_mapping(row)

    def test_reasoning_length_must_be_between_sixty_and_three_hundred(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"][0]["reasoning"] = "Too short."
        with self.assertRaisesRegex(DatasetValidationError, "between 60 and 300"):
            CanonicalExample.from_mapping(row)

        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"][0]["reasoning"] = "x" * 301
        with self.assertRaisesRegex(DatasetValidationError, "between 60 and 300"):
            CanonicalExample.from_mapping(row)

    def test_duplicate_solution_step_is_rejected(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"].append(dict(row["solution_steps"][0]))
        with self.assertRaisesRegex(DatasetValidationError, "duplicated"):
            CanonicalExample.from_mapping(row)

    def test_incorrect_explicit_calculation_is_rejected(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"][0]["reasoning"] = (
            "The first quantity is 12 - 3 = 8, which is deliberately incorrect "
            "for this validation test."
        )
        with self.assertRaisesRegex(DatasetValidationError, "incorrect equality"):
            CanonicalExample.from_mapping(row)

    def test_compound_explicit_calculations_are_validated_whole(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"] = [
            {
                "guiding_question": "What is the combined total?",
                "reasoning": (
                    "Adding all three quantities gives the combined total "
                    "14 + 19 + 11 = 44 for the first calculation."
                ),
            },
            {
                "guiding_question": "What requested total follows from that sum?",
                "reasoning": (
                    "The requested combined total is therefore 44, using the sum "
                    "of all three stated quantities."
                ),
            },
        ]
        row["final_answer"] = "#### 44"
        CanonicalExample.from_mapping(row)

        row["solution_steps"][0]["reasoning"] = (
            "Adding all three quantities gives the combined total "
            "14 + 19 + 11 = 43, which is deliberately incorrect here."
        )
        with self.assertRaisesRegex(DatasetValidationError, "incorrect equality"):
            CanonicalExample.from_mapping(row)

    def test_fraction_currency_parentheses_and_chains_are_supported(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"] = [
            {
                "guiding_question": "What is the percentage amount?",
                "reasoning": (
                    "Converting eighty percent and multiplying gives the amount "
                    "80/100 * 20 = 16 in exact units."
                ),
            },
            {
                "guiding_question": "What is the exact fractional amount?",
                "reasoning": (
                    "Taking one third of the stated total gives "
                    "1/3 * 240 = 240 / 3 = 80 in exact units."
                ),
            },
            {
                "guiding_question": "What is the overtime rate?",
                "reasoning": (
                    "Adding the fifty-percent premium produces the hourly rate "
                    "20 + (20 * 0.5) = 20 + 10 = $30.00."
                ),
            },
            {
                "guiding_question": "What is the final pay?",
                "reasoning": (
                    "Combining the regular and overtime amounts gives the final pay "
                    "$480.00 + $270.00 = $750.00."
                ),
            },
        ]
        row["final_answer"] = "#### 750"
        CanonicalExample.from_mapping(row)

    def test_multistep_subtraction_is_not_truncated_to_its_tail(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"] = [
            {
                "guiding_question": "What amount remains?",
                "reasoning": (
                    "Subtracting both used portions from the starting amount gives "
                    "90 - 40 - 12 = 38 remaining units."
                ),
            },
            {
                "guiding_question": "What final amount does the problem request?",
                "reasoning": (
                    "The requested remaining amount is therefore 38 units after both "
                    "stated portions have been removed."
                ),
            },
        ]
        row["final_answer"] = "#### 38"
        CanonicalExample.from_mapping(row)

    def test_last_explicit_result_must_match_final_answer(self) -> None:
        row = canonical_mapping("x-v01", "x", 1)
        row["solution_steps"][0]["reasoning"] = (
            "Subtracting the removed amount gives the first quantity "
            "12 - 3 = 9, which is the last explicit result."
        )
        row["final_answer"] = "#### 10"
        with self.assertRaisesRegex(DatasetValidationError, "does not match"):
            CanonicalExample.from_mapping(row)


class DeduplicationTests(unittest.TestCase):
    def test_ngram_result_is_reusable(self) -> None:
        ngrams = generate_ngrams("one two three four five six", 5)
        self.assertEqual(len(ngrams), 2)
        self.assertEqual(len(ngrams), 2)

    def test_exact_jaccard(self) -> None:
        left = generate_ngrams("one two three four five six seven eight nine ten", 5)
        right = generate_ngrams(
            "one two three four five six seven eight nine ten eleven", 5
        )
        self.assertAlmostEqual(jaccard_similarity(left, right), 6 / 7)

    def test_filter_rejects_exact_and_near_duplicates(self) -> None:
        base_question = "one two three four five six seven eight nine ten?"
        rows = [
            canonical_mapping("a-v01", "a", 1, question=base_question),
            canonical_mapping("b-v01", "b", 1, question=base_question.upper()),
            canonical_mapping(
                "c-v01", "c", 1, question=f"{base_question[:-1]} eleven?"
            ),
            canonical_mapping(
                "d-v01",
                "d",
                1,
                question="apples baskets sales totals remain unrelated words here?",
            ),
        ]
        accepted, rejected = filter_and_deduplicate(
            rows,
            min_solution_chars=0,
            max_solution_chars=1000,
            ngram_size=5,
            jaccard_threshold=0.85,
        )
        self.assertEqual([row.example_id for row in accepted], ["a-v01", "d-v01"])
        self.assertEqual(
            [row["reason"] for row in rejected],
            ["exact_duplicate_question", "near_duplicate_question"],
        )
        self.assertAlmostEqual(rejected[1]["similarity"], 6 / 7)


class PairingTests(unittest.TestCase):
    def test_render_removes_only_guiding_questions(self) -> None:
        record = CanonicalExample.from_mapping(canonical_mapping("a-v01", "a", 1))
        socratic = render_answer(record, socratic=True)
        non_socratic = render_answer(record, socratic=False)
        for step in record.solution_steps:
            self.assertIn(step.reasoning, socratic)
            self.assertIn(step.reasoning, non_socratic)
            self.assertIn(step.guiding_question, socratic)
            self.assertNotIn(step.guiding_question, non_socratic)
        self.assertTrue(socratic.endswith("#### 9"))
        self.assertTrue(non_socratic.endswith("#### 9"))

    def test_group_split_has_no_source_leakage_and_is_deterministic(self) -> None:
        records = [
            CanonicalExample.from_mapping(
                canonical_mapping(f"s{source}-v{variant:02d}", f"s{source}", variant)
            )
            for source in range(8)
            for variant in range(1, 4)
        ]
        train1, valid1, manifest1 = split_by_source(records, seed=42)
        train2, valid2, manifest2 = split_by_source(records, seed=42)
        self.assertEqual(
            [row.example_id for row in train1], [row.example_id for row in train2]
        )
        self.assertEqual(
            [row.example_id for row in valid1], [row.example_id for row in valid2]
        )
        self.assertEqual(manifest1, manifest2)
        self.assertFalse(
            {row.source_id for row in train1} & {row.source_id for row in valid1}
        )

    def test_pair_validator_fails_on_leaked_question(self) -> None:
        record = CanonicalExample.from_mapping(canonical_mapping("a-v01", "a", 1))
        socratic = render_record(record, socratic=True)
        non_socratic = render_record(record, socratic=False)
        non_socratic["answer"] = socratic["answer"]
        with self.assertRaisesRegex(DatasetValidationError, "non-Socratic rendering"):
            validate_rendered_pair([record], [socratic], [non_socratic])

    def test_end_to_end_build_writes_matched_splits_and_hashes(self) -> None:
        rows = [
            canonical_mapping(
                f"s{source}-v{variant:02d}",
                f"s{source}",
                variant,
                question=(
                    f"source {source} variant {variant} unique objects values operations "
                    f"remain distinct token {source}-{variant}?"
                ),
            )
            for source in range(4)
            for variant in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            write_jsonl(input_path, rows)
            manifest = build_paired_dataset(
                input_path,
                root / "output",
                target_count=10,
                min_solution_chars=0,
                max_solution_chars=1000,
                jaccard_threshold=1.0,
                validation_fraction=0.2,
                seed=42,
            )
            self.assertEqual(manifest["accepted_rows"], 10)
            self.assertEqual(manifest["pairing_report"]["status"], "passed")
            soc_train = list(read_jsonl(root / "output" / "socratic" / "train.jsonl"))
            non_train = list(
                read_jsonl(root / "output" / "non_socratic" / "train.jsonl")
            )
            self.assertEqual(
                [row["example_id"] for row in soc_train],
                [row["example_id"] for row in non_train],
            )
            self.assertTrue(manifest["outputs"]["canonical"]["sha256"])


if __name__ == "__main__":
    unittest.main()

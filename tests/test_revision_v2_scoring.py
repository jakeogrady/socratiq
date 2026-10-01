from __future__ import annotations

import random
import unittest

from src import evaluate_v2 as v1
from src.revision_v2 import scoring


class PrimaryRuleTests(unittest.TestCase):
    def test_last_marked_answer_anywhere(self) -> None:
        self.assertEqual(scoring.last_marked("#### 18\nmore text\n#### 126"), "126")

    def test_trailing_text_and_question_mark_do_not_invalidate(self) -> None:
        self.assertEqual(scoring.last_marked("#### 8\n?"), "8")
        self.assertEqual(scoring.last_marked("so #### 45. Done?"), "45")

    def test_commas_decimals_and_signs_normalise(self) -> None:
        self.assertEqual(scoring.last_marked("#### 1,234"), "1234")
        self.assertEqual(scoring.last_marked("#### 8.0"), "8")
        self.assertEqual(scoring.last_marked("#### -0"), "0")
        self.assertEqual(scoring.last_marked("#### 0.50"), "0.5")
        self.assertEqual(scoring.last_marked("#### -12.5"), "-12.5")

    def test_huge_integers_do_not_crash(self) -> None:
        self.assertEqual(scoring.last_marked("#### " + "9" * 40), "9" * 40)

    def test_no_marker_means_no_answer(self) -> None:
        self.assertIsNone(scoring.last_marked("The answer is 42."))
        self.assertIsNone(scoring.last_marked(""))

    def test_marker_without_number_is_invalid(self) -> None:
        self.assertIsNone(scoring.last_marked("#### forty-two"))


class SensitivityRuleTests(unittest.TestCase):
    def test_first_marked_takes_the_first_answer(self) -> None:
        self.assertEqual(scoring.first_marked("#### 18\n#### 126"), "18")
        self.assertIsNone(scoring.first_marked("no marker 5"))

    def test_fallback_uses_marked_answer_when_present(self) -> None:
        self.assertEqual(
            scoring.last_marked_fallback_last_number("x 7 #### 3 then 9"), "3"
        )

    def test_fallback_uses_last_number_when_no_marker(self) -> None:
        self.assertEqual(
            scoring.last_marked_fallback_last_number("15 + 27 = 42 apples"), "42"
        )
        self.assertEqual(
            scoring.last_marked_fallback_last_number("It costs $1,500.00."), "1500"
        )
        self.assertEqual(scoring.last_marked_fallback_last_number("x = -5"), "-5")
        self.assertIsNone(scoring.last_marked_fallback_last_number("no digits here"))

    def test_fallback_does_not_read_a_hyphen_as_a_sign(self) -> None:
        self.assertEqual(scoring.last_marked_fallback_last_number("pages 3-4"), "4")

    def test_all_three_rules_are_registered(self) -> None:
        self.assertEqual(
            set(scoring.RULES),
            {"last_marked", "first_marked", "last_marked_fallback_last_number"},
        )
        self.assertEqual(scoring.PRIMARY_RULE, "last_marked")
        extracted = scoring.extract_all("12 then #### 5\n#### 6")
        self.assertEqual(
            extracted,
            {
                "last_marked": "6",
                "first_marked": "5",
                "last_marked_fallback_last_number": "6",
            },
        )


class ComparisonAndVoteTests(unittest.TestCase):
    def test_exact_decimal_comparison(self) -> None:
        self.assertTrue(scoring.answers_equal("8", "8.0"))
        self.assertFalse(scoring.answers_equal("8.01", "8"))
        self.assertFalse(scoring.answers_equal(None, "8"))

    def test_vote_prefers_majority_then_greedy_then_numeric(self) -> None:
        self.assertEqual(scoring.majority_vote(["3", "3", "5"]), ("3", False))
        self.assertEqual(
            scoring.majority_vote(["3", "5"], greedy_answer="5"), ("5", True)
        )
        self.assertEqual(scoring.majority_vote(["5", "3"]), ("3", True))
        self.assertEqual(scoring.majority_vote([None, None]), (None, False))


class DiagnosticTests(unittest.TestCase):
    def test_bare_answer(self) -> None:
        self.assertTrue(scoring.is_bare_answer("#### 42"))
        self.assertTrue(scoring.is_bare_answer("#### 42\n#### 42\n"))
        self.assertFalse(scoring.is_bare_answer("2 + 2 = 4\n#### 4"))
        self.assertFalse(scoring.is_bare_answer(""))

    def test_repetition_loop(self) -> None:
        self.assertTrue(scoring.is_repetition_loop("work\n" + "#### 42\n" * 12))
        self.assertFalse(scoring.is_repetition_loop("#### 10\n#### 10"))
        self.assertFalse(
            scoring.is_repetition_loop("\n".join(f"line {i}" for i in range(30)))
        )

    def test_question_lines_and_limit(self) -> None:
        self.assertTrue(scoring.has_question_line("How many?\n3 + 4 = 7\n#### 7"))
        self.assertFalse(scoring.has_question_line("3 + 4 = 7\n#### 7"))
        self.assertTrue(scoring.diagnose("x", "length")["limit_hit"])
        self.assertFalse(scoring.diagnose("x", "stop")["limit_hit"])


class AgreementWithV1Tests(unittest.TestCase):
    """The primary rule and vote must reproduce the v1 implementation exactly."""

    def test_primary_rule_matches_v1_on_generated_text(self) -> None:
        rng = random.Random(7)  # noqa: S311 - reproducible fuzzing, not security
        pieces = [
            "#### ",
            "####",
            "12",
            "1,234",
            "-3",
            "8.0",
            "0.50",
            " ",
            "\n",
            "?",
            "text",
            "=",
            "+",
            "7",
            "99999999999999999999999999999",
        ]
        for _ in range(5000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
            self.assertEqual(
                scoring.last_marked(text), v1.extract_marked_number(text), text
            )

    def test_vote_matches_v1(self) -> None:
        rng = random.Random(11)  # noqa: S311 - reproducible fuzzing, not security
        for _ in range(2000):
            answers = [rng.choice(["1", "2", "3", "10", None]) for _ in range(5)]
            greedy = rng.choice(["1", "2", "3", None])
            self.assertEqual(
                scoring.majority_vote(answers, greedy_answer=greedy),
                v1.majority_vote(answers, greedy_answer=greedy),
            )


if __name__ == "__main__":
    unittest.main()

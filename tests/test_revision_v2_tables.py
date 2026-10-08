"""Unit tests for the revision-v2 evidence tables (formatting and condition parsing)."""

import unittest
from fractions import Fraction

from src.revision_v2 import tables


class FormattingTests(unittest.TestCase):
    def test_fixed_rounds_half_up_from_exact_counts(self) -> None:
        # 577/1319 = 43.745..., 2/8 = 25.0, 1/8 = 12.5 exactly.
        self.assertEqual(tables.fixed(tables.percent(577, 1319)), "43.7")
        self.assertEqual(tables.fixed(Fraction(125, 1000), 2), "0.13")
        self.assertEqual(tables.fixed(Fraction(-1, 1000)), "0.0")

    def test_signed_marks_direction(self) -> None:
        self.assertEqual(tables.signed(Fraction(7, 1319) * 100), "+0.5")
        self.assertEqual(tables.signed(-5.0), "-5.0")
        self.assertEqual(tables.signed(0.0), "0.0")

    def test_p_text(self) -> None:
        self.assertEqual(tables.p_text(1.0), "1.00")
        self.assertEqual(tables.p_text(0.7386), "0.74")
        self.assertEqual(tables.p_text(0.0931), "0.093")
        self.assertEqual(tables.p_text(0.00119), "0.0012")
        self.assertEqual(tables.p_text(0.00001), "<0.0001")

    def test_tex_interval_uses_math_minus(self) -> None:
        self.assertEqual(tables.tex_interval("-2.1", "+3.2"), "[$-2.1$, $+3.2$]")
        self.assertEqual(tables.tex_interval("7.5", "10.6"), "[7.5, 10.6]")


class ConditionTests(unittest.TestCase):
    def test_parse_condition(self) -> None:
        self.assertEqual(
            tables.parse_condition("qwen3_0.6b_non_socratic_10k_lr8e-5"),
            {"model": "qwen3_0.6b", "arm": "non_socratic", "data": "10k"},
        )
        self.assertEqual(
            tables.parse_condition("qwen3_1.7b_socratic_lr1e-4"),
            {"model": "qwen3_1.7b", "arm": "socratic", "data": "full"},
        )
        self.assertEqual(
            tables.parse_condition("llama3.2_1b_base"),
            {"model": "llama3.2_1b", "arm": "base", "data": "none"},
        )

    def test_parse_condition_rejects_unknown(self) -> None:
        with self.assertRaises(tables.EvidenceError):
            tables.parse_condition("gemma_socratic_lr8e-5")
        with self.assertRaises(tables.EvidenceError):
            tables.parse_condition("qwen3_0.6b_mixed_lr8e-5")


if __name__ == "__main__":
    unittest.main()

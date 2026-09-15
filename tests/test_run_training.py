from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.run_training import (
    TrainingConfigurationError,
    collect_environment,
    compare_paired_configs,
    load_yaml,
    parse_training_log,
    run_training_job,
    validate_training_config,
)

CONFIG_DIR = Path("configs/reviewer_rerun")


class TrainingConfigurationTests(unittest.TestCase):
    def test_all_reviewer_configs_validate(self) -> None:
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            if path.name == "protocol.yaml":
                continue
            with self.subTest(path=path):
                validation = validate_training_config(load_yaml(path))
                self.assertEqual(validation["status"], "passed")

    def test_qwen_pairs_differ_only_in_paths(self) -> None:
        pairs = (
            ("qwen3_0.6b_socratic.yaml", "qwen3_0.6b_non_socratic.yaml"),
            ("qwen3_1.7b_socratic.yaml", "qwen3_1.7b_non_socratic.yaml"),
        )
        for left_name, right_name in pairs:
            with self.subTest(pair=(left_name, right_name)):
                result = compare_paired_configs(
                    load_yaml(CONFIG_DIR / left_name),
                    load_yaml(CONFIG_DIR / right_name),
                )
                self.assertEqual(result["status"], "passed")

    def test_invalid_historical_mlp_target_is_rejected(self) -> None:
        config = load_yaml(CONFIG_DIR / "qwen3_0.6b_socratic.yaml")
        config["lora_parameters"]["keys"][4] = "self_attn.gate_proj"
        with self.assertRaisesRegex(TrainingConfigurationError, "seven intended"):
            validate_training_config(config)

    def test_scientific_notation_must_parse_as_numeric_yaml(self) -> None:
        config = load_yaml(CONFIG_DIR / "qwen3_0.6b_socratic.yaml")
        config["learning_rate"] = "8e-5"
        with self.assertRaisesRegex(TrainingConfigurationError, "numeric YAML"):
            validate_training_config(config)

    def test_model_revision_must_be_an_immutable_sha(self) -> None:
        config = load_yaml(CONFIG_DIR / "qwen3_0.6b_socratic.yaml")
        config["model_revision"] = "main"
        with self.assertRaisesRegex(TrainingConfigurationError, "model_revision"):
            validate_training_config(config)

    def test_model_precision_is_explicit(self) -> None:
        config = load_yaml(CONFIG_DIR / "qwen3_0.6b_socratic.yaml")
        config.pop("model_precision")
        with self.assertRaisesRegex(TrainingConfigurationError, "model_precision"):
            validate_training_config(config)

    def test_paired_hyperparameter_drift_is_rejected(self) -> None:
        left = load_yaml(CONFIG_DIR / "qwen3_0.6b_socratic.yaml")
        right = copy.deepcopy(left)
        right["data"] = "other-data"
        right["adapter_path"] = "other-adapter"
        right["learning_rate"] = 1e-3
        with self.assertRaisesRegex(TrainingConfigurationError, "learning_rate"):
            compare_paired_configs(left, right)

    def test_dry_run_requires_no_model_or_dataset_download(self) -> None:
        result = run_training_job(
            CONFIG_DIR / "qwen3_0.6b_socratic.yaml",
            experiment_id="dry-run-test",
            run_dir=None,
            dry_run=True,
            smoke_iters=None,
            minimum_free_gib=40,
            skip_model_preflight=False,
            skip_revision_resolution=False,
        )
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(
            result["model"]["resolved_revision"],
            "42096995f6402fde107068cf530136fe64b604f8",
        )
        self.assertTrue(result["resolved_executable"].endswith("mlx_lm.lora"))
        self.assertFalse(result["dataset"]["files"]["train"]["exists"])


class TrainingReportingTests(unittest.TestCase):
    def test_parse_mlx_log(self) -> None:
        text = """Trainable parameters: 0.912% (11.272M/1235.814M)
Iter 1: Val loss 2.499, Val took 25.639s
Iter 200: Train loss 1.0, Peak mem 5.631 GB
Iter 201: Val loss 1.900, Val took 25.000s
"""
        metrics = parse_training_log(text)
        self.assertEqual(metrics["trainable_parameters"]["trainable_count"], 11_272_000)
        self.assertEqual(metrics["trainable_parameters"]["total_count"], 1_235_814_000)
        self.assertEqual(metrics["peak_mlx_memory_gb"], 5.631)
        self.assertEqual(metrics["best_validation"], {"iteration": 201, "loss": 1.9})

    def test_environment_report_does_not_capture_environment_variables(self) -> None:
        report = collect_environment()
        self.assertIn("disk", report)
        self.assertIn("git", report)
        self.assertNotIn("environ", report)
        self.assertNotIn("OPENAI_API_KEY", str(report))


if __name__ == "__main__":
    unittest.main()

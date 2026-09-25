from __future__ import annotations

import unittest
from pathlib import Path

from src.evaluate_v2 import BENCHMARKS
from src.openai_conversion_v2 import _build_parser
from src.rerun_utils import file_sha256, protocol_identity
from src.run_training import load_yaml


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_required_counts_and_no_fallback(self) -> None:
        protocol = load_yaml(Path("configs/reviewer_rerun/protocol.yaml"))
        self.assertEqual(protocol["protocol_version"], "1.2")
        self.assertEqual(protocol["source_dataset"]["expected_train_rows"], 7473)
        self.assertEqual(protocol["synthetic_data"]["expected_candidate_rows"], 22419)
        self.assertEqual(protocol["synthetic_data"]["target_accepted_rows"], 20000)
        self.assertFalse(protocol["synthetic_data"]["allow_model_fallback"])
        self.assertEqual(
            protocol["synthetic_data"]["prompt_version"], "matched-pairs-v3"
        )
        self.assertEqual(protocol["synthetic_data"]["canonical_schema_version"], "1.1")
        self.assertEqual(protocol["synthetic_data"]["solution_steps"]["minimum"], 2)
        self.assertEqual(protocol["synthetic_data"]["solution_steps"]["maximum"], 6)
        self.assertTrue(
            protocol["synthetic_data"]["solution_steps"][
                "guiding_question_requires_terminal_question_mark"
            ]
        )
        self.assertEqual(
            protocol["evaluation"]["benchmarks"]["gsm8k"]["expected_rows"], 1319
        )
        self.assertEqual(
            protocol["evaluation"]["benchmarks"]["multiarith"]["expected_rows"], 180
        )
        self.assertEqual(
            protocol["evaluation"]["benchmarks"]["svamp"]["expected_rows"], 300
        )
        self.assertEqual(protocol["evaluation"]["self_consistency"]["samples"], 5)
        revisions = [
            protocol["source_dataset"]["revision"],
            *(
                benchmark["revision"]
                for benchmark in protocol["evaluation"]["benchmarks"].values()
            ),
            *(model["revision"] for model in protocol["models"].values()),
        ]
        self.assertTrue(all(len(revision) == 40 for revision in revisions))

    def test_protocol_has_a_stable_file_identity(self) -> None:
        identity = protocol_identity()
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertTrue(identity["path"].endswith("protocol.yaml"))

    def test_render_default_matches_protocol_target(self) -> None:
        protocol = load_yaml(Path("configs/reviewer_rerun/protocol.yaml"))
        args = _build_parser().parse_args(["render", "--input", "canonical.jsonl"])
        self.assertEqual(
            args.target_count,
            protocol["synthetic_data"]["target_accepted_rows"],
        )

    def test_training_configs_match_protocol_model_pins(self) -> None:
        protocol = load_yaml(Path("configs/reviewer_rerun/protocol.yaml"))
        config_names = {
            "qwen3_0.6b_socratic.yaml": "qwen3_0.6b",
            "qwen3_0.6b_non_socratic.yaml": "qwen3_0.6b",
            "qwen3_1.7b_socratic.yaml": "qwen3_1.7b",
            "qwen3_1.7b_non_socratic.yaml": "qwen3_1.7b",
            "llama3.2_1b_socratic.yaml": "llama3.2_1b",
        }
        for config_name, model_key in config_names.items():
            with self.subTest(config=config_name):
                config = load_yaml(Path("configs/reviewer_rerun") / config_name)
                model = protocol["models"][model_key]
                self.assertEqual(config["model"], model["identifier"])
                self.assertEqual(config["model_revision"], model["revision"])
                self.assertEqual(config["model_precision"], model["precision"])

    def test_gsmhard_extension_matches_registry_and_preserves_base_protocol(
        self,
    ) -> None:
        extension_path = Path(
            "configs/reviewer_rerun/extensions/gsmhard_extension_v1.yaml"
        )
        extension = load_yaml(extension_path)
        benchmark = extension["benchmark"]
        registered = BENCHMARKS["gsm_hard"]

        self.assertEqual(extension["extension_version"], "1.0")
        self.assertEqual(benchmark["dataset"], registered.dataset)
        self.assertEqual(benchmark["revision"], registered.revision)
        self.assertEqual(benchmark["expected_rows"], registered.expected_rows)
        self.assertEqual(
            extension["few_shot_source"]["revision"],
            registered.few_shot_revision,
        )
        base_protocol = Path(extension["base_protocol"]["path"])
        self.assertEqual(
            extension["base_protocol"]["sha256"], file_sha256(base_protocol)
        )
        adapter_checksums = extension["evaluation"]["adapter_checksum_manifest"]
        self.assertEqual(
            adapter_checksums["sha256"],
            file_sha256(Path(adapter_checksums["path"])),
        )
        self.assertEqual(extension["matrix"]["expected_full_runs"], 16)
        self.assertEqual(extension["matrix"]["expected_full_decisions"], 21104)


if __name__ == "__main__":
    unittest.main()

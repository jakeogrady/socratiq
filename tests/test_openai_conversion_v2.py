from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.openai_conversion_v2 import (
    PAID_CONFIRMATION,
    assemble_batch_results,
    batch_request,
    build_batch_input,
    build_retry_input,
    estimate_batch,
    preflight_request,
    submit_batch,
)
from src.rerun_utils import read_jsonl, write_jsonl


def source_row(source_id: str = "gsm8k-train-000000") -> dict:
    return {
        "source_id": source_id,
        "source_index": 0,
        "question": "Ana has 12 apples and gives away 3. How many remain?",
        "answer": "Ana subtracts 3 from 12, obtaining 9. #### 9",
    }


def structured_variants() -> dict:
    return {
        "variants": [
            {
                "variant_id": index,
                "synthetic_question": f"A distinct valid problem variation number {index}?",
                "solution_steps": [
                    {
                        "guiding_question": "What value should be found first?",
                        "reasoning": "Subtracting 3 from 12 gives 9 and completes the calculation.",
                    }
                ],
                "final_answer": "#### 9",
            }
            for index in (1, 2, 3)
        ]
    }


class RequestConstructionTests(unittest.TestCase):
    def test_request_contains_full_worked_solution_and_schema(self) -> None:
        request = batch_request(
            source_row(), model="teacher-snapshot", max_output_tokens=500
        )
        body = request["body"]
        self.assertEqual(request["custom_id"], "gsm8k-train-000000")
        self.assertIn("Ana subtracts 3 from 12", body["input"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["store"])

    def test_build_and_estimate_are_offline_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            input_path = root / "batch.jsonl"
            write_jsonl(source_path, [source_row(), source_row("gsm8k-train-000001")])
            manifest = build_batch_input(
                source_path,
                input_path,
                model="teacher-snapshot",
                max_output_tokens=500,
            )
            estimate = estimate_batch(input_path)
            self.assertEqual(manifest["request_count"], 2)
            self.assertEqual(manifest["candidate_example_count"], 6)
            self.assertEqual(estimate["request_count"], 2)
            self.assertEqual(estimate["configured_maximum_output_tokens"], 1000)
            self.assertTrue(input_path.with_suffix(".manifest.json").exists())

    def test_paid_stages_refuse_without_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "confirm-api-call"):
                preflight_request(
                    root / "source.jsonl",
                    root / "out.json",
                    model="teacher",
                    max_output_tokens=100,
                    confirm_api_call=False,
                )
            with self.assertRaisesRegex(RuntimeError, PAID_CONFIRMATION):
                submit_batch(
                    root / "batch.jsonl",
                    root / "manifest.json",
                    confirmation="no",
                )


class AssemblyTests(unittest.TestCase):
    def test_assemble_joins_by_custom_id_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            output_path = root / "batch-output.jsonl"
            canonical_path = root / "canonical.jsonl"
            audit_path = root / "audit.jsonl"
            source = source_row()
            write_jsonl(source_path, [source])
            write_jsonl(
                output_path,
                [
                    {
                        "id": "batch-result-1",
                        "custom_id": source["source_id"],
                        "response": {
                            "status_code": 200,
                            "request_id": "request-1",
                            "body": {
                                "id": "response-1",
                                "status": "completed",
                                "model": "teacher-snapshot",
                                "usage": {"input_tokens": 10, "output_tokens": 20},
                                "output": [
                                    {
                                        "type": "message",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": json.dumps(
                                                    structured_variants()
                                                ),
                                            }
                                        ],
                                    }
                                ],
                            },
                        },
                        "error": None,
                    }
                ],
            )
            manifest = assemble_batch_results(
                source_path,
                output_path,
                canonical_path,
                audit_path,
                expected_model="teacher-snapshot",
            )
            rows = list(read_jsonl(canonical_path))
            self.assertEqual(len(rows), 3)
            self.assertEqual(manifest["successful_source_count"], 1)
            self.assertEqual(rows[0]["source_id"], source["source_id"])
            self.assertEqual(
                rows[0]["generation"]["returned_model"], "teacher-snapshot"
            )
            self.assertEqual(list(read_jsonl(audit_path)), [])

    def test_model_mismatch_is_audited_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            output_path = root / "output.jsonl"
            canonical_path = root / "canonical.jsonl"
            audit_path = root / "audit.jsonl"
            source = source_row()
            write_jsonl(source_path, [source])
            write_jsonl(
                output_path,
                [
                    {
                        "custom_id": source["source_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "status": "completed",
                                "model": "different-model",
                                "output_text": json.dumps(structured_variants()),
                            },
                        },
                    }
                ],
            )
            manifest = assemble_batch_results(
                source_path,
                output_path,
                canonical_path,
                audit_path,
                expected_model="teacher-snapshot",
            )
            self.assertEqual(manifest["canonical_example_count"], 0)
            self.assertEqual(manifest["retry_custom_ids"], ["gsm8k-train-000000"])
            self.assertEqual(
                list(read_jsonl(audit_path))[0]["reason"],
                "unexpected_returned_model",
            )

    def test_retry_file_contains_only_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "batch.jsonl"
            assembly = root / "assembly.json"
            retry = root / "retry.jsonl"
            requests = [
                batch_request(
                    source_row(f"source-{index}"),
                    model="teacher",
                    max_output_tokens=100,
                )
                for index in range(3)
            ]
            write_jsonl(original, requests)
            assembly.write_text(
                json.dumps({"missing_custom_ids": ["source-1"]}), encoding="utf-8"
            )
            manifest = build_retry_input(original, assembly, retry)
            self.assertEqual(manifest["retry_request_count"], 1)
            self.assertEqual(list(read_jsonl(retry))[0]["custom_id"], "source-1")

    def test_retry_prefers_all_failed_and_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "batch.jsonl"
            assembly = root / "assembly.json"
            retry = root / "retry.jsonl"
            requests = [
                batch_request(
                    source_row(f"source-{index}"),
                    model="teacher",
                    max_output_tokens=100,
                )
                for index in range(3)
            ]
            write_jsonl(original, requests)
            assembly.write_text(
                json.dumps(
                    {
                        "missing_custom_ids": ["source-2"],
                        "retry_custom_ids": ["source-1", "source-2"],
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_retry_input(original, assembly, retry)
            self.assertEqual(manifest["retry_request_count"], 2)
            self.assertEqual(
                [row["custom_id"] for row in read_jsonl(retry)],
                ["source-1", "source-2"],
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.openai_conversion_v2 import (
    PAID_CONFIRMATION,
    PROMPT_VERSION,
    VARIANTS_SCHEMA,
    assemble_batch_results,
    assembly_paths_from_batch_manifest,
    batch_request,
    build_batch_input,
    build_rejection_retry_input,
    build_retry_input,
    canonical_from_variants,
    estimate_batch,
    merge_canonical_outputs,
    preflight_request,
    submit_batch,
)
from src.rerun_utils import file_sha256, read_jsonl, write_jsonl


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
                        "reasoning": (
                            "Subtracting the given amount from the starting quantity gives "
                            "12 - 3 = 9 items remaining."
                        ),
                    },
                    {
                        "guiding_question": "What remaining quantity does the problem request?",
                        "reasoning": (
                            "The requested remaining quantity is therefore 9 items because "
                            "the subtraction accounts for every item given away."
                        ),
                    },
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
        self.assertEqual(PROMPT_VERSION, "matched-pairs-v3")
        variant = VARIANTS_SCHEMA["properties"]["variants"]["items"]
        self.assertEqual(variant["properties"]["solution_steps"]["minItems"], 2)
        self.assertEqual(variant["properties"]["solution_steps"]["maxItems"], 6)
        reasoning = variant["properties"]["solution_steps"]["items"]["properties"][
            "reasoning"
        ]
        self.assertEqual(reasoning["minLength"], 60)
        self.assertEqual(reasoning["maxLength"], 300)
        guiding_question = variant["properties"]["solution_steps"]["items"][
            "properties"
        ]["guiding_question"]
        self.assertEqual(guiding_question["pattern"], r"\?$")
        self.assertEqual(variant["properties"]["synthetic_question"]["pattern"], r"\?")

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

    def test_retry_assembly_is_scoped_to_its_batch_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            batch_input_path = root / "retry.jsonl"
            batch_output_path = root / "retry-output.jsonl"
            canonical_path = root / "retry-canonical.jsonl"
            audit_path = root / "retry-audit.jsonl"
            requested_source = source_row("source-1")
            write_jsonl(source_path, [source_row("source-0"), requested_source])
            write_jsonl(
                batch_input_path,
                [
                    batch_request(
                        requested_source, model="teacher", max_output_tokens=100
                    )
                ],
            )
            write_jsonl(
                batch_output_path,
                [
                    {
                        "custom_id": "source-1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": "response-1",
                                "status": "completed",
                                "model": "teacher",
                                "output_text": json.dumps(structured_variants()),
                            },
                        },
                    }
                ],
            )

            manifest = assemble_batch_results(
                source_path,
                batch_output_path,
                canonical_path,
                audit_path,
                batch_input_path=batch_input_path,
                expected_model="teacher",
            )

            self.assertEqual(manifest["source_count"], 1)
            self.assertEqual(manifest["successful_source_count"], 1)
            self.assertEqual(manifest["missing_custom_ids"], [])
            self.assertEqual(manifest["retry_custom_ids"], [])
            self.assertEqual(manifest["batch_input_path"], str(batch_input_path))

    def test_merge_validates_uniqueness_and_orders_canonical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            output_path = root / "merged.jsonl"
            first_records = canonical_from_variants(
                source_row("source-1"), structured_variants(), response_metadata={}
            )
            second_records = canonical_from_variants(
                source_row("source-0"), structured_variants(), response_metadata={}
            )
            write_jsonl(first_path, (record.as_dict() for record in first_records))
            write_jsonl(second_path, (record.as_dict() for record in second_records))

            manifest = merge_canonical_outputs([first_path, second_path], output_path)
            merged = list(read_jsonl(output_path))

            self.assertEqual(manifest["canonical_example_count"], 6)
            self.assertEqual(manifest["source_count"], 2)
            self.assertEqual(merged[0]["source_id"], "source-0")
            self.assertEqual(merged[-1]["source_id"], "source-1")
            with self.assertRaisesRegex(ValueError, "Duplicate canonical example_id"):
                merge_canonical_outputs([first_path, first_path], root / "bad.jsonl")

    def test_batch_manifest_resolves_and_hash_checks_assembly_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch_input_path = root / "batch.jsonl"
            batch_output_path = root / "batch-output.jsonl"
            manifest_path = root / "batch.manifest.json"
            write_jsonl(batch_input_path, [{"custom_id": "source-0"}])
            write_jsonl(batch_output_path, [{"custom_id": "source-0"}])
            manifest_path.write_text(
                json.dumps(
                    {
                        "batch_input_path": str(batch_input_path),
                        "batch_input_sha256": file_sha256(batch_input_path),
                        "downloads": {
                            "output": {
                                "path": str(batch_output_path),
                                "sha256": file_sha256(batch_output_path),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            output, submitted_input = assembly_paths_from_batch_manifest(manifest_path)
            self.assertEqual(output, batch_output_path)
            self.assertEqual(submitted_input, batch_input_path)

            batch_output_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "output hash differs"):
                assembly_paths_from_batch_manifest(manifest_path)

    def test_filtered_retry_replaces_whole_source_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_input = root / "batch.jsonl"
            initial_path = root / "initial.jsonl"
            retry_input = root / "filtered-retry.jsonl"
            retry_path = root / "retry.jsonl"
            rejection_path = root / "rejections.jsonl"
            merged_path = root / "merged.jsonl"
            source = source_row("source-0")
            write_jsonl(
                original_input,
                [batch_request(source, model="teacher", max_output_tokens=100)],
            )
            initial = canonical_from_variants(
                source, structured_variants(), response_metadata={"attempt": 1}
            )
            replacement = canonical_from_variants(
                source, structured_variants(), response_metadata={"attempt": 2}
            )
            write_jsonl(initial_path, (record.as_dict() for record in initial))
            write_jsonl(retry_path, (record.as_dict() for record in replacement))
            write_jsonl(
                rejection_path,
                [{"example_id": "source-0-v01", "reason": "solution_length"}],
            )

            retry_manifest = build_rejection_retry_input(
                original_input,
                initial_path,
                rejection_path,
                retry_input,
            )
            merge_manifest = merge_canonical_outputs(
                [initial_path, retry_path],
                merged_path,
                replace_sources_from_later=True,
            )

            self.assertEqual(retry_manifest["retry_request_count"], 1)
            self.assertEqual(retry_manifest["retry_source_ids"], ["source-0"])
            self.assertEqual(merge_manifest["replacement_sources"], ["source-0"])
            self.assertEqual(merge_manifest["canonical_example_count"], 3)
            self.assertTrue(
                all(
                    row["generation"]["attempt"] == 2 for row in read_jsonl(merged_path)
                )
            )


if __name__ == "__main__":
    unittest.main()

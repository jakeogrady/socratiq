from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.paired_dataset import CanonicalExample
from src.rerun_utils import file_sha256
from src.revision_v2 import configs, data, evaluate, llama_base, queue, stats, train
from src.revision_v2.protocol import ProtocolError, load_protocol, require_v2_output


def canonical(
    example_id: str, source_id: str, question: str = "How many apples??"
) -> CanonicalExample:
    return CanonicalExample.from_mapping(
        {
            "schema_version": "1.1",
            "example_id": example_id,
            "source_id": source_id,
            "variant_id": 1,
            "source_question": "q",
            "source_solution": "s",
            "synthetic_question": "Ann has 2 apples and gets 3 more. How many apples does she have?",
            "solution_steps": [
                {
                    "guiding_question": question,
                    "reasoning": "Ann starts with 2 apples and gets 3 more, so 2 + 3 = 5 apples.",
                },
                {
                    "guiding_question": "What is the total?",
                    "reasoning": "The total number of apples Ann has is therefore 5 apples in all.",
                },
            ],
            "final_answer": "#### 5",
        }
    )


class DataTests(unittest.TestCase):
    def test_collapse_only_terminal_repeats(self) -> None:
        self.assertEqual(
            data.collapse_terminal_question_marks("How many??"), "How many?"
        )
        self.assertEqual(
            data.collapse_terminal_question_marks("How many???"), "How many?"
        )
        self.assertEqual(
            data.collapse_terminal_question_marks("Why?? How?"), "Why?? How?"
        )

    def test_render_keeps_reasoning_identical_in_both_arms(self) -> None:
        record = canonical("e1", "s1")
        socratic = data.render_answer_v2(record, socratic=True)
        plain = data.render_answer_v2(record, socratic=False)
        self.assertTrue(socratic.startswith("How many apples?\n"))
        self.assertNotIn("??", socratic)
        self.assertEqual(
            plain, "\n".join([s.reasoning for s in record.solution_steps] + ["#### 5"])
        )

    def test_nested_subsets_are_exact_and_nested(self) -> None:
        records = [
            canonical(f"e{s}-{v}", f"s{s}") for s in range(40) for v in range(1 + s % 3)
        ]
        subsets = data.nested_source_subsets(records, [10, 30], "7:subsets")
        sizes = {
            s: sum(r.source_id in set(sources) for r in records)
            for s, sources in subsets.items()
        }
        self.assertEqual(sizes, {10: 10, 30: 30})
        self.assertTrue(set(subsets[10]) <= set(subsets[30]))

    def test_shuffle_is_reproducible_and_changes_order(self) -> None:
        records = [canonical(f"e{i}", f"s{i}") for i in range(50)]
        first = [r.example_id for r in data.shuffled(records, "2026:train")]
        self.assertEqual(
            first, [r.example_id for r in data.shuffled(records, "2026:train")]
        )
        self.assertNotEqual(first, [r.example_id for r in records])

    def test_pairing_audit_detects_a_leak(self) -> None:
        record = canonical("e1", "s1")
        soc = [data.render_row_v2(record, socratic=True)]
        non = [data.render_row_v2(record, socratic=False)]
        self.assertEqual(
            data.audit_pairing(soc, non, {"e1": record})["status"], "passed"
        )
        leaked = [dict(non[0], answer=soc[0]["answer"])]
        self.assertEqual(
            data.audit_pairing(soc, leaked, {"e1": record})["status"], "failed"
        )

    def test_written_files_match_tracked_manifest(self) -> None:
        if not Path("data/revision_v2/full/socratic/train.jsonl").is_file():
            self.skipTest("v2 data not built on this machine")
        self.assertEqual(data.verify_files()["status"], "passed")


class ConfigAndScheduleTests(unittest.TestCase):
    def test_generated_configs_match_protocol(self) -> None:
        self.assertEqual(configs.generate(check=True)["status"], "passed")

    def test_pairs_differ_only_in_identity_data_and_adapter(self) -> None:
        runs = configs.training_runs()
        by_pair: dict[str, list[dict]] = {}
        for run in runs:
            by_pair.setdefault(run["pair_id"], []).append(run)
        self.assertEqual(len(by_pair), 9)
        for pair in by_pair.values():
            left, right = (train.load_config(Path(r["config_path"])) for r in pair)
            differing = {
                k for k in left.keys() | right.keys() if left.get(k) != right.get(k)
            }
            self.assertEqual(differing, {"run_id", "data", "adapter_path"})

    def test_schedule_replay_meets_gate_one(self) -> None:
        for run in configs.training_runs():
            replay = configs.replay_schedule(
                train.load_config(Path(run["config_path"]))
            )
            with self.subTest(run=run["run_id"]):
                self.assertEqual(replay["updates"], 188)
                self.assertEqual(replay["discarded_microbatches"], 0)
                self.assertAlmostEqual(replay["final_lr"], 1e-6, delta=1e-12)
                if run["model_key"] == "qwen3_1.7b":
                    self.assertEqual(replay["peak_updates"], [1])
                else:
                    self.assertEqual(replay["peak_updates"][0], 16)
                    self.assertAlmostEqual(
                        replay["lr_by_update"][15],
                        run["peak"],
                        delta=run["peak"] * 1e-6,
                    )
                    for k in range(1, 17):
                        self.assertAlmostEqual(
                            replay["lr_by_update"][k - 1],
                            run["peak"] * k / 16,
                            delta=run["peak"] * 1e-6,
                        )

    def test_output_paths_refuse_v1_runs(self) -> None:
        with self.assertRaises(ProtocolError):
            require_v2_output(Path("runs/reviewer_rerun/evaluation/x"))
        self.assertEqual(
            require_v2_output(Path("runs/revision_v2/x")), Path("runs/revision_v2/x")
        )


def write_safetensors(
    path: Path, tensors: dict[str, bytes], metadata: dict | None = None
) -> None:
    header: dict = {"__metadata__": metadata or {"format": "pt"}}
    offset = 0
    for name, payload in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [len(payload) // 2],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header).encode()
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors.values())
    )


class LlamaBaseTests(unittest.TestCase):
    def test_patch_fixes_the_date(self) -> None:
        config = {
            "chat_template": "A\n" + llama_base.ORIGINAL_DATE_BLOCK + "\nB",
            "eos_token": "x",
        }
        patched = llama_base.patch_chat_template(config)
        self.assertNotIn("strftime_now", patched["chat_template"])
        self.assertIn('set date_string = "26 Jul 2024"', patched["chat_template"])
        self.assertEqual(patched["eos_token"], "x")

    def test_patch_refuses_unknown_template(self) -> None:
        with self.assertRaises(llama_base.LlamaBaseError):
            llama_base.patch_chat_template({"chat_template": "no date logic"})

    def test_byte_comparison_ignores_header_metadata_but_not_bits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left, same, different = Path(tmp, "a"), Path(tmp, "b"), Path(tmp, "c")
            write_safetensors(left, {"w": b"\x01\x02\x03\x04"})
            write_safetensors(same, {"w": b"\x01\x02\x03\x04"}, {"format": "mlx"})
            write_safetensors(different, {"w": b"\x01\x02\x03\x05"})
            self.assertTrue(
                llama_base.compare_safetensors(left, same)["all_tensors_bitwise_equal"]
            )
            self.assertFalse(
                llama_base.compare_safetensors(left, different)[
                    "all_tensors_bitwise_equal"
                ]
            )

    def test_placed_checkpoint_verifies(self) -> None:
        path = Path(load_protocol()["models"]["llama3.2_1b"]["local_path"])
        if not path.is_dir():
            self.skipTest("Llama checkpoint not placed on this machine")
        self.assertEqual(llama_base.verify_checkpoint(path)["status"], "passed")


class TrainingChecksTests(unittest.TestCase):
    def test_logged_learning_rates_are_checked(self) -> None:
        replay = configs.replay_schedule(
            train.load_config(
                Path("configs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5.yaml")
            )
        )
        lines = [
            f"Iter {it}: Train loss 1.000, Learning Rate {lr:.3e}, It/sec 1.0"
            for it, lr in replay["logged_lr_by_iteration"].items()
        ]
        good = train.check_logged_learning_rates("\n".join(lines), replay)
        self.assertEqual(good["status"], "passed")
        self.assertEqual(good["logged_at_update_16"], "8.000e-05")
        self.assertEqual(good["logged_at_update_188"], "1.000e-06")
        bad = train.check_logged_learning_rates(
            "\n".join(lines).replace("8.000e-05", "4.000e-05"), replay
        )
        self.assertEqual(bad["status"], "failed")

    def test_pilot_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {
                "status": "completed",
                "run_id": "p",
                "validation_losses": [
                    {"iteration": 1, "loss": 2.4},
                    {"iteration": 6016, "loss": 0.93},
                ],
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(train.pilot_check(run_dir)["decision"], "accept")
            manifest["validation_losses"][-1]["loss"] = 0.96
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(train.pilot_check(run_dir)["decision"], "reject")
            manifest["validation_losses"][-1]["loss"] = float("nan")
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(train.pilot_check(run_dir)["decision"], "reject")

    def test_edited_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp, "x.yaml")
            copy.write_text(
                Path(
                    "configs/revision_v2/training/qwen3_0.6b_socratic_lr8e-5.yaml"
                ).read_text()
            )
            with self.assertRaises(train.TrainingError):
                train.validate_config(copy)


class EvaluationGuardTests(unittest.TestCase):
    def test_protocol_combinations(self) -> None:
        self.assertEqual(
            evaluate.evaluation_plan("gsm8k", "P0", "greedy")["spec"].few_shot_indices,
            (),
        )
        self.assertEqual(
            evaluate.evaluation_plan("gsm8k", "P4", "greedy")["spec"].few_shot_indices,
            (0, 1, 2, 3),
        )
        self.assertEqual(
            evaluate.evaluation_plan("gsm_hard", "P0", "greedy")["max_tokens"], 512
        )
        for bad in (
            ("svamp", "P4", "greedy"),
            ("gsm_hard", "P0", "sc5"),
            ("gsm8k", "P4", "sc5"),
        ):
            with self.subTest(bad=bad), self.assertRaises(evaluate.EvaluationError):
                evaluate.evaluation_plan(*bad)

    def test_adapter_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapters.safetensors").write_bytes(b"weights")
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.adapter_identity(adapter, "0" * 64)
            digest = file_sha256(adapter / "adapters.safetensors")
            self.assertEqual(
                evaluate.adapter_identity(adapter, digest)["weights_sha256"], digest
            )

    def test_decisions_cover_every_rule(self) -> None:
        samples = [
            evaluate.score_sample({"response": text, "finish_reason": "stop"})
            for text in ("#### 5", "#### 5\n#### 6", "7")
        ]
        decisions = evaluate.decide(samples, None, "5")
        self.assertEqual(decisions["last_marked"]["predicted_answer"], "5")
        self.assertEqual(decisions["first_marked"]["predicted_answer"], "5")
        self.assertTrue(decisions["last_marked_fallback_last_number"]["vote_tied"])


class StatisticsTests(unittest.TestCase):
    def test_mcnemar_matches_v1_audit(self) -> None:
        self.assertAlmostEqual(
            stats.mcnemar_exact(125, 210), 3.97180680e-06, delta=1e-12
        )
        self.assertAlmostEqual(stats.mcnemar_exact(133, 84), 0.00107280, delta=1e-8)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)

    def test_newcombe_matches_v1_audit(self) -> None:
        a, b, c = 322, 125, 210
        low, high = stats.newcombe_paired(a, b, c, 1319 - a - b - c)
        self.assertAlmostEqual(100 * low, -9.14, places=2)
        self.assertAlmostEqual(100 * high, -3.74, places=2)

    def test_holm(self) -> None:
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class QueueTests(unittest.TestCase):
    def test_every_core_item_is_assigned_once(self) -> None:
        items = queue.expand("isik", "8e-5") + queue.expand("chee", "8e-5")
        ids = [i["id"] for i in items if not i["id"].startswith("2_determinism")]
        self.assertEqual(len(ids), len(set(ids)))
        trained = {i["run_id"] for i in items if i["kind"] == "train"}
        expected = {
            r["run_id"]
            for r in configs.training_runs()
            if r["peak_variant"] in {"default", "fixed"}
        }
        self.assertEqual(trained, expected)
        evaluated = {i["condition_id"] for i in items if i["kind"] == "eval"}
        self.assertEqual(
            evaluated,
            expected | {"qwen3_0.6b_base", "qwen3_1.7b_base", "llama3.2_1b_base"},
        )

    def test_matched_pairs_stay_on_one_machine(self) -> None:
        for machine in ("isik", "chee"):
            items = queue.expand(machine, "4e-5")
            runs = {i["run_id"] for i in items if i["kind"] == "train"}
            for run in runs:
                partner = (
                    run.replace("_non_socratic", "_socratic")
                    if "_non_socratic" in run
                    else run.replace("_socratic", "_non_socratic")
                )
                self.assertIn(partner, runs)

    def test_stage_order(self) -> None:
        for machine in ("isik", "chee"):
            stages = [i["stage"] for i in queue.expand(machine, "8e-5")]
            self.assertEqual(stages, sorted(stages))

    def test_report_lists_every_item(self) -> None:
        text = queue.report("chee", "8e-5")
        self.assertEqual(len(text.splitlines()), len(queue.expand("chee", "8e-5")) + 2)
        self.assertIn("machine=chee peak=8e-5", text)

    def test_halved_peak_changes_only_affected_models(self) -> None:
        runs = {
            i["run_id"] for i in queue.expand("chee", "4e-5") if i["kind"] == "train"
        }
        self.assertIn("qwen3_1.7b_socratic_lr1e-4", runs)
        self.assertIn("qwen3_0.6b_socratic_5k_lr4e-5", runs)


class ScriptSyntaxTests(unittest.TestCase):
    def test_phase2_shell_scripts_parse(self) -> None:
        for script in (
            "phase2_setup.sh",
            "phase2_queue_isik.sh",
            "phase2_queue_chee.sh",
        ):
            with self.subTest(script=script):
                completed = subprocess.run(  # noqa: S603 - repository-owned fixed script
                    ["/bin/sh", "-n", f"scripts/{script}"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

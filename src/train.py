import argparse
import json
import logging
import random
import re
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import mlx.optimizers as optim
from mlx.nn import Module
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.utils import save_model
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ValLossRecorder(TrainingCallback):
    """Record validation loss values."""

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file

        if self.output_file.exists():
            self.output_file.unlink()

    def on_val_loss_report(self, val_info: dict) -> None:
        """Call whenever validation loss is reported."""
        val_loss = val_info.get("val_loss") or val_info.get("loss")
        step = val_info.get("step")

        if val_loss is None or step is None:
            return

        record = {
            "step": step,
            "val_loss": float(val_loss),
        }

        with self.output_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info("[Step %s] Validation Loss: %s", step, val_loss)


def create_run_directory(args: Namespace) -> tuple[str, Path]:
    """Create unique run directory and return (run_name, run_dir)."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split("/")[-1]

    run_name = f"{model_short}_lr{args.lr}_rank{args.rank}_seed{args.seed}_{timestamp}"

    run_dir = Path("../data") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting run: %s", run_name)
    logger.info("Saving to: %s", run_dir)

    return run_name, run_dir


def load_model_and_tokenizer(model_name: str) -> tuple[Module, PreTrainedTokenizer]:
    """Load model and configure tokenizer."""
    model, tokenizer = load(model_name)

    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>assistant\n' }}{% endif %}"
    )

    return model, tokenizer


def load_datasets(tokenizer: PreTrainedTokenizer) -> tuple[CacheDataset, CacheDataset]:
    """Load train and validation datasets."""
    config = DatasetConfig(
        prompt_feature="prompt",
        completion_feature="answer",
    )

    dataset_path = Path("../data")

    train_ds, val_ds, _ = load_local_dataset(dataset_path, tokenizer, config)

    return CacheDataset(train_ds), CacheDataset(val_ds)


def apply_lora(model: Module, rank: int) -> None:
    """Freeze model and apply LoRA adapters."""
    model.freeze()

    linear_to_lora_layers(
        model,
        num_layers=16,
        config={
            "rank": rank,
            "dropout": 0.05,
            "scale": 20.0,
        },
    )


def build_training_args(run_dir: Path) -> SFTTrainingArgs:
    """Create SFT training arguments."""
    return SFTTrainingArgs(
        batch_size=1,
        iters=200,
        grad_checkpoint=True,
        adapter_file=str(run_dir / "adapters.safetensors"),
    )


def run_training(
    model: dict,
    train_dataset: CacheDataset,
    val_dataset: CacheDataset,
    training_args: SFTTrainingArgs,
    lr: float,
    run_dir: Path,
) -> object:
    """Run SFT training and return metrics."""
    val_log_path = run_dir / "validation_log.jsonl"
    callback = ValLossRecorder(val_log_path)

    return train_sft(
        model=model,
        optimizer=optim.Adam(learning_rate=lr),
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_callback=callback,
    )


def save_run_artifacts(
    run_name: str,
    run_dir: Path,
    metrics: dict,
    args: Namespace,
) -> None:
    """Save metrics locally and append to global results file."""
    with (run_dir / "final_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    record = {
        "run_name": run_name,
        "model": args.model_name,
        "lr": args.lr,
        "rank": args.rank,
        "seed": args.seed,
        "metrics": metrics,
    }

    results_path = Path(args.results_file)
    with results_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Run saved successfully.")


class DatasetConfig(BaseModel):
    """Configuration for dataset features."""

    prompt_feature: str = "prompt"
    completion_feature: str = "answer"


def extract_question_from_batch_input(prompt_text: str) -> str:
    """Extract question from batch input."""
    match = re.search(
        r"Problem:\n(.*?)\n\nSolution:",
        prompt_text,
        re.DOTALL,
    )

    if not match:
        msg = "Could not extract question"
        raise ValueError(msg)

    return match.group(1).strip()


def extract_socratic_answer(batch_item: dict) -> str | None:
    """Extract socratic answer."""
    try:
        body = batch_item["response"]["body"]

        if body.get("status") != "completed":
            return None

        if body.get("incomplete_details") is not None:
            return None

        for block in body["output"]:
            if block["type"] == "message":
                return block["content"][0]["text"]

    except Exception:
        return None

    return None


def write_jsonl(path: Path, data: list) -> None:
    """Write .Jsonl."""
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f)
            f.write("\n")


def build_train_valid_split(
    batch_input_path: Path,
    results_path: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    """Split .jsonl into train.jsonl and valid.jsonl."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_map = read_batch_jsonl(batch_input_path)

    samples = construct_final_socratic_file(results_path, question_map)

    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)

    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:]

    write_jsonl(output_dir / "train.jsonl", train_samples)
    write_jsonl(output_dir / "valid.jsonl", valid_samples)


def read_batch_jsonl(batch_input_path: Path) -> dict:
    """Read batch .jsonl."""
    question_map = {}

    with batch_input_path.open() as f:
        for line in f:
            item = json.loads(line)

            custom_id = item["custom_id"]

            # Extract question from prompt
            user_prompt = None
            for msg in item["body"]["input"]:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                    break

            if not user_prompt:
                continue

            try:
                question = extract_question_from_batch_input(user_prompt)
                question_map[custom_id] = question
            except Exception:
                logger.exception("Could not extract question from batch input")
                continue

    return question_map


def construct_final_socratic_file(results_path: Path, question_map: dict) -> list:
    """Construct final socratic file."""
    samples = []

    with results_path.open() as f:
        for line in f:
            try:
                item = json.loads(line)

                custom_id = item["custom_id"]

                if custom_id not in question_map:
                    continue

                answer = extract_socratic_answer(item)
                if not answer:
                    continue

                samples.append(
                    {
                        "prompt": question_map[custom_id],
                        "answer": answer,
                    }
                )

            except Exception:
                logger.exception("Found an error")
                continue

    return samples


def build_parser() -> argparse.ArgumentParser:
    """Build parser for argument collation."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="File to append final results (optional)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser


def finetune(args: Namespace) -> None:
    """Finetune model."""
    run_name, run_dir = create_run_directory(args)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    train_dataset, val_dataset = load_datasets(tokenizer)

    apply_lora(model, args.rank)

    training_args = build_training_args(run_dir)

    metrics = run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        lr=args.lr,
        run_dir=run_dir,
    )

    logger.info("Training complete: %s", metrics)

    save_run_artifacts(run_name, run_dir, metrics, args)

    merged_model_dir = run_dir / "merged_model"
    merged_model_dir.mkdir(exist_ok=True, parents=True)

    # Save model weights
    save_model(
        str(merged_model_dir),
        model,
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(merged_model_dir))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    finetune(args)

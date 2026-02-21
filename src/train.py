import json
import logging
import random
import re
from pathlib import Path

import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def finetune() -> None:
    """Fine-tune model on Socratic JSONL dataset using MLX."""
    # Load model + tokenizer
    model_path = "mlx-community/Llama-3.2-3B-8bit"
    model, tokenizer = load(model_path)

    # Set a simple chat template for Socratic-style Q&A
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>assistant\n' }}{% endif %}"
    )

    config = DatasetConfig(prompt_feature="prompt", completion_feature="answer")

    # Path to folder containing train.jsonl, valid.jsonl, test.jsonl
    dataset_path = Path("../data")
    train_ds, val_ds, _ = load_local_dataset(dataset_path, tokenizer, config)

    # Freeze base model before LoRA
    model.freeze()

    # Apply LoRA to linear layers
    linear_to_lora_layers(
        model,
        num_layers=16,
        config={"rank": 8, "dropout": 0.05, "scale": 20.0},
    )

    # Training arguments
    training_args = TrainingArgs(
        batch_size=1,
        iters=200,
        grad_checkpoint=True,
    )

    # Wrap MLX datasets in CacheDataset
    train_dataset = CacheDataset(train_ds)
    val_dataset = CacheDataset(val_ds)

    # Start training
    metrics = train(
        model=model,
        optimizer=optim.Adam(learning_rate=1e-5),
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    build_train_valid_split(
        Path("../batch_input.jsonl"),
        Path("../socratic_results_final.jsonl"),
        Path("../data"),
    )

    finetune()

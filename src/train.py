import argparse
import json
import logging
import random
import re
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import mlx.optimizers as optim
from huggingface_hub import HfApi
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.utils import save_config, save_model
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------
# Validation Loss Logger
# ------------------------------------------------


class ValLossRecorder(TrainingCallback):
    def __init__(self, output_file: Path):
        self.output_file = output_file

        if self.output_file.exists():
            self.output_file.unlink()

    def on_val_loss_report(self, val_info: dict):
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

        logger.info("[Step %s] Val Loss: %s", step, val_loss)


# ------------------------------------------------
# Dataset Utilities
# ------------------------------------------------


class DatasetConfig(BaseModel):
    prompt_feature: str = "prompt"
    completion_feature: str = "answer"


def extract_question_from_batch_input(prompt_text: str) -> str:
    match = re.search(
        r"Problem:\n(.*?)\n\nSolution:",
        prompt_text,
        re.DOTALL,
    )

    if not match:
        raise ValueError("Could not extract question")

    return match.group(1).strip()


def extract_socratic_answer(batch_item: dict) -> str | None:
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


def read_batch_jsonl(batch_input_path: Path) -> dict:
    question_map = {}

    with batch_input_path.open() as f:
        for line in f:
            item = json.loads(line)

            custom_id = item["custom_id"]

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
                logger.exception("Batch parse error")
                continue

    return question_map


def construct_final_socratic_file(results_path: Path, question_map: dict) -> list:
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
                continue

    return samples


def build_train_valid_split(
    batch_input_path: Path,
    results_path: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
):
    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    question_map = read_batch_jsonl(batch_input_path)

    samples = construct_final_socratic_file(results_path, question_map)

    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)

    write_jsonl(output_dir / "train.jsonl", samples[:split_idx])
    write_jsonl(output_dir / "valid.jsonl", samples[split_idx:])


def write_jsonl(path: Path, data: list):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f)
            f.write("\n")


# ------------------------------------------------
# Model Loading
# ------------------------------------------------


def load_model_and_tokenizer(model_name: str):
    model, tokenizer, config = load(model_name, return_config=True)

    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn><eos>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>assistant\\n' }}{% endif %}"
    )

    return model, tokenizer, config


# ------------------------------------------------
# Training
# ------------------------------------------------


def apply_lora(model, rank: int):
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


def load_datasets(tokenizer):
    config = DatasetConfig()

    dataset_path = PROJECT_ROOT / "data"

    train_ds, val_ds, _ = load_local_dataset(
        dataset_path,
        tokenizer,
        config,
    )

    return CacheDataset(train_ds), CacheDataset(val_ds)


def build_training_args(run_dir: Path):
    return SFTTrainingArgs(
        batch_size=1,
        iters=5,
        grad_checkpoint=True,
        adapter_file=str(run_dir / "adapters.safetensors"),
    )


def run_training(model, train_dataset, val_dataset, args, lr, run_dir):
    callback = ValLossRecorder(run_dir / "validation_log.jsonl")

    return train_sft(
        model=model,
        optimizer=optim.Adam(learning_rate=lr),
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_callback=callback,
    )


# ------------------------------------------------
# Saving + Upload
# ------------------------------------------------


def save_finetuned_model(model, tokenizer, config, save_dir: Path):
    logger.info("Saving model to %s", save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, save_dir / "config.json")

    save_model(str(save_dir), model)

    tokenizer.save_pretrained(str(save_dir))


def upload_to_hf(merged_model_dir: Path, repo_id: str):
    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(merged_model_dir),
        repo_type="model",
    )


# ------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------


def create_run_directory(args):
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    model_short = args.model_name.split("/")[-1]

    run_name = f"{model_short}_lr{args.lr}_rank{args.rank}_seed{args.seed}_{timestamp}"

    run_dir = PROJECT_ROOT / "data" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_name, run_dir


def finetune(args: Namespace):
    run_name, run_dir = create_run_directory(args)

    model, tokenizer, config = load_model_and_tokenizer(args.model_name)

    train_dataset, val_dataset = load_datasets(tokenizer)

    apply_lora(model, args.rank)

    training_args = build_training_args(run_dir)

    run_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        lr=args.lr,
        run_dir=run_dir,
    )

    merged_model_dir = run_dir / "merged_model"

    save_finetuned_model(
        model,
        tokenizer,
        config,
        merged_model_dir,
    )

    upload_to_hf(
        merged_model_dir,
        repo_id=f"Jakeog123/{run_name}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    finetune(args)

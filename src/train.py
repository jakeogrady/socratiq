import logging
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
    finetune()

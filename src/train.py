import logging

import mlx.optimizers as optim
import numpy as np
from datasets import Dataset, load_dataset
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
from mlx_lm.tuner.datasets import CacheDataset, create_dataset
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetConfig(BaseModel):
    """Configuration for dataset features."""

    prompt_feature: str = "prompt"
    completion_feature: str = "answer"


def load_hf_dataset(
    ds_path: str,
    tokenizer: PreTrainedTokenizer,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a HuggingFace dataset."""
    try:
        dataset = load_dataset(ds_path)

        names = ["train", "valid", "test"]
        ds_config = DatasetConfig()
        train, val, test = [
            (
                create_dataset(dataset[name], tokenizer, config=ds_config)
                if name in dataset
                else []
            )
            for name in names
        ]
    except Exception:
        logger.exception("Error loading dataset %s", ds_path)
        raise
    return train, val, test


def finetune() -> None:
    """Fine-tune model on a custom dataset."""
    rng = np.random.default_rng(42)
    rng.normal()
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-8bit")

    training_args = TrainingArgs(
        batch_size=1,
        iters=200,
        grad_checkpoint=True,
    )

    train_ds, val_ds, _ = load_hf_dataset("mlx-community/gsm8k", tokenizer)

    model.freeze()

    linear_to_lora_layers(
        model,
        num_layers=16,
        config={"rank": 8, "dropout": 0.05, "scale": 20.0},
    )

    metrics = train(
        model=model,
        optimizer=optim.Adam(learning_rate=1e-5),
        args=training_args,
        train_dataset=CacheDataset(train_ds),
        val_dataset=CacheDataset(val_ds),
    )

    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    finetune()

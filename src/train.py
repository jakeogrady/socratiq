import logging

import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, train

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def finetune() -> None:
    """Fine-tune model on a custom dataset."""
    model, _ = load("mlx-community/Mistral-7B-Instruct-v0.3-8bit")

    training_args = TrainingArgs(
        batch_size=1,
        iters=200,
        grad_checkpoint=True,
    )

    metrics = train(
        model=model,
        optimizer=optim.Adam(learning_rate=1e-5),
        args=training_args,
        train_dataset="data/train.jsonl",
        val_dataset="data/valid.jsonl",
    )

    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    finetune()

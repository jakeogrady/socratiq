import logging
import re
import time
from typing import TypeVar

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, ConfigDict

from src.constants import (
    ANSWER_REGEX,
    DATASET_FORMAT,
    OPENAI_GSM8K,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


class GSM8KDataset(BaseModel):
    """GSM8K Dataset wrapper class."""

    train: Dataset
    test: Dataset

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: dict) -> None:
        super().__init__(**data)

    def get_test_case_answer(self, index: int) -> str | None:
        """Extract the correct answer from the dataset for a given test case index."""
        answer_match = re.search(ANSWER_REGEX, self.test[index]["text"])
        return answer_match.group(1).strip() if answer_match else None


def load_gsm8k(split: str = "main") -> DatasetDict:
    """Load the GSM8K dataset from Hugging Face."""
    return load_dataset(OPENAI_GSM8K, split)


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    """Preprocess the GSM8K dataset to the desired format."""
    return dataset.map(
        lambda example: {
            "text": DATASET_FORMAT.format(
                question=example["question"],
                answer=example["answer"],
            ),
        },
        remove_columns=["question", "answer"],
    )


def load_and_process_gsm8k() -> GSM8KDataset:
    """Load and preprocess the GSM8K dataset."""
    logger.info("Loading dataset...")
    start = time.time()

    ds = load_gsm8k()
    logger.info("Dataset loaded in %ss", time.time() - start)

    return GSM8KDataset(**preprocess_dataset(ds))


if __name__ == "__main__":
    ds = load_gsm8k()
    preprocess_dataset(ds)

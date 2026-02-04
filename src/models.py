import logging
import re
import time
from collections.abc import Generator
from typing import TypeVar

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, ConfigDict, Field

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
    valid: Dataset | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: dict) -> None:
        super().__init__(**data)

        if self.valid is None:
            self.generate_validation()

    def train_length(self) -> None:
        """Log the length of the training dataset."""
        logger.info("Length of Training Dataset: %s", len(self.train))

    def test_length(self) -> None:
        """Log the length of the test dataset."""
        logger.info("Length of Test Dataset: %s", len(self.test))

    def generate_validation(self) -> None:
        """Generate a validation set from the training set."""
        split = self.train.train_test_split(test_size=0.1, seed=42)
        self.train = split["train"]
        self.valid = split["test"]

    def get_test_case_answer(self, index: int) -> str | None:
        """Extract the correct answer from the dataset for a given test case index."""
        answer_match = re.search(ANSWER_REGEX, self.test[index]["text"])
        return answer_match.group(1).strip() if answer_match else None

    def yield_train_cases(self, length: T | None) -> Generator:
        """Yield training cases one by one."""
        length = length if length is not None else len(self.train)

        for i in range(length):
            training_task = self.train[i]["answer"]

            yield training_task

    def convert_to_jsonl(self) -> None:
        """Convert datasets to JSONL format and save to files."""
        for name in []:
            dataset = getattr(self, name)
            dataset = preprocess_dataset(dataset)
            if dataset is not None:
                logger.info("Converting %s dataset to JSONL format...", name)
                dataset.to_json(f"data/{name}.jsonl", orient_records=True)


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

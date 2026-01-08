from typing import Any

from datasets import load_dataset, DatasetDict, Dataset
from pydantic import BaseModel, ConfigDict

from constants import OPENAI_GSM8K, DATASET_FORMAT


class GSM8KDataset(BaseModel):
    train: Dataset
    test: Dataset
    validation: DatasetDict | None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: Any):
        super().__init__(**data)

    def train_length(self):
        print(f"Length of Training Dataset: {len(self.train)}")

    def test_length(self):
        print(f"Length of Test Dataset: {len(self.test)}")

    def generate_validation(self):
        self.validation = self.train.train_test_split(test_size=0.1, seed=42)
        print(self.validation)


def load_gsm8k(split: str = "main") -> DatasetDict:
    return load_dataset(OPENAI_GSM8K, split)


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    return dataset.map(
        lambda example: {
            "text": DATASET_FORMAT.format(
                question=example["question"], answer=example["answer"]
            )
        },
        remove_columns=["question", "answer"],
    )


def load_and_process_gsm8k():
    ds = load_gsm8k()
    return preprocess_dataset(ds)


if __name__ == "__main__":
    ds = load_and_process_gsm8k()

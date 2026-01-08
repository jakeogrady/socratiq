from datasets import load_dataset, DatasetDict, Dataset
from pydantic import BaseModel, ConfigDict

from constants import OPENAI_GSM8K, DATASET_FORMAT


class GSM8KDataset(BaseModel):
    train: Dataset
    test: Dataset

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def train_length(self):
        print(f"Length of Training Dataset: {len(self.train)}")

    def test_length(self):
        print(f"Length of Test Dataset: {len(self.test)}")


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

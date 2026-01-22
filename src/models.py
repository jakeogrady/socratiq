import logging
import re
import time
from collections.abc import Generator
from typing import TypeVar

import torch
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from constants import (
    ANSWER_REGEX,
    DATASET_FORMAT_PHI_2,
    MODEL_NAME,
    OPENAI_GSM8K,
    QUESTION_REGEX,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


class Model(BaseModel):
    """Model wrapper class."""

    name: str
    model: PreTrainedModel
    device: torch.device | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: dict) -> None:
        super().__init__(**data)

        self.device = torch.device("cpu")
        self.model.to(self.device)
        torch.set_num_threads(4)

        logger.info("Using device: %s", self.device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self.model.gradient_checkpointing_enable()

    @staticmethod
    def create_model(name: str = MODEL_NAME) -> "Model":
        """Create a Model instance."""
        logger.info("Loading model %s in FP16 on CPU...", name)
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map=None,
        )
        logger.info("Model loaded in %ss", time.time() - start)
        wrapper = Model(name=name, model=model)

        if not hasattr(wrapper, "device"):
            wrapper.device = torch.device("cpu")
        return wrapper

    def generate_prompt(
        self, test_set: Dataset, few_shot_num: int = 4, target_question_index: int = 1
    ) -> str:
        """Generate a few-shot prompt for the model."""
        few_shot_texts = test_set[:few_shot_num]["text"]
        few_shot_block = "\n\n".join(few_shot_texts)

        match = re.search(
            QUESTION_REGEX,
            test_set[few_shot_num + target_question_index]["text"],
            re.DOTALL,
        )
        target_question = match.group(1).strip()

        logger.info("Target Question: %s", target_question)

        return few_shot_block + "\n\nQuestion: " + target_question + "\nAnswer:"

    def generate_response(
        self,
        tokenizer: AutoTokenizer,
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> str:
        """Generate a response from the model given a text prompt."""
        inputs = tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1500,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        match = re.search(r"(####\s*-?\d+)", generated_text)
        if match:
            return generated_text[: match.end()].strip()
        return generated_text.strip()


class Tokenizer(BaseModel):
    """Tokenizer wrapper class."""

    name: str
    model: PreTrainedTokenizer

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def print_chat_template(self) -> None:
        """Print the chat template of the tokenizer, if available."""
        logger.info(self.model.chat_template)

    @staticmethod
    def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
        """Load the tokenizer for the specified model."""
        logger.info("Loading tokenizer %ss ...", model_name)
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        logger.info("Tokenizer loaded in %ss ...", time.time() - start)
        return tokenizer


class GSM8KDataset(BaseModel):
    """GSM8K Dataset wrapper class."""

    train: Dataset
    test: Dataset
    validation: DatasetDict | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, /, **data: dict) -> None:
        super().__init__(**data)

    def train_length(self) -> None:
        """Log the length of the training dataset."""
        logger.info("Length of Training Dataset: %s", len(self.train))

    def test_length(self) -> None:
        """Log the length of the test dataset."""
        logger.info("Length of Test Dataset: %s", len(self.test))

    def generate_validation(self) -> None:
        """Generate a validation set from the training set."""
        self.validation = self.train.train_test_split(test_size=0.1, seed=42)

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


def load_gsm8k(split: str = "main") -> DatasetDict:
    """Load the GSM8K dataset from Hugging Face."""
    return load_dataset(OPENAI_GSM8K, split)


def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
    """Preprocess the GSM8K dataset to the desired format."""
    return dataset.map(
        lambda example: {
            "text": DATASET_FORMAT_PHI_2.format(
                question=example["question"],
                answer=example["answer"],
            ),
        },
        remove_columns=["question", "answer"],
    )


def load_and_process_gsm8k() -> GSM8KDataset:
    """Load and preprocess the GSM8K dataset."""
    ds = load_gsm8k()
    return GSM8KDataset(**preprocess_dataset(ds))


if __name__ == "__main__":
    ds = load_and_process_gsm8k()
    logger.info(ds.generate_prompt())

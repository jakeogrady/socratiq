import logging
import re
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from mlx.nn import Module
from mlx_lm import generate, load
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.constants import DATASET_CONVERSION_PROMPT, MISTRAL_7B_Q4
from src.models import GSM8KDataset, load_gsm8k

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_dataset(output_path: str, conversion_fn: Callable) -> None:
    """Convert a dataset using a provided conversion function."""
    # Read the input dataset
    dataset = load_gsm8k()

    # Convert the dataset using the provided conversion function
    converted_data = conversion_fn(dataset)

    # Write the converted dataset to the output file
    with Path.open(output_path, "w") as outfile:
        outfile.write(converted_data)


def model_request_conversion(
    model: Module, tokenizer: TokenizerWrapper, task: str
) -> str:
    """Convert a task into a model request format."""
    prompt = DATASET_CONVERSION_PROMPT.format(task=task)

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=150,
    )


def generate_new_dataset(dataset: GSM8KDataset) -> Dataset:
    """Generate a new dataset by converting each task using the model."""
    converted_tasks = []
    model, tokenizer = load(MISTRAL_7B_Q4)

    for i, task in enumerate(dataset.yield_train_cases(2)):
        converted_task = model_request_conversion(model, tokenizer, task)
        converted_task = process_converted_answer(converted_task)
        converted_tasks.append(
            {"question": dataset.test["question"][i], "answer": converted_task}
        )
        logger.info("\n\nOriginal Task:\n%s", task)
        logger.info("\nConverted Task:\n%s", converted_task)

    return Dataset.from_list(converted_tasks)


def process_converted_answer(answer: str) -> str:
    """Process the converted task to ensure proper formatting."""
    clean_answer = answer.replace("\\/", "/")

    lines = clean_answer.split("\n")
    lines = [line.strip() for line in lines]

    clean_answer = "\n".join(lines)

    clean_answer = clean_answer.replace("\u00d7", "x")

    pattern = r"Socratic worked version:\s*(.+)$"

    match = re.search(pattern, clean_answer, re.DOTALL)

    if match:
        return match.group(1).strip()

    return clean_answer


if __name__ == "__main__":
    dataset = GSM8KDataset(**load_gsm8k())

    ds = generate_new_dataset(dataset)
    ds.to_json("data/converted_gsm8k.jsonl")

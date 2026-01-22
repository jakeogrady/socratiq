import logging
import re
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.models import GSM8KDataset, load_gsm8k
from src.train import Model, Tokenizer

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


def model_request_conversion(model: Model, tokenizer: AutoTokenizer, task: str) -> str:
    """Convert a task into a model request format."""
    prompt = f"""
        Convert the following GSM8K math solution into a Socratic-style worked solution.

        Task:
        {task}

        Instructions:
        - Rewrite the solution as a sequence of Socratic questions.
        - Each question must immediately include its answer.
        - Preserve all intermediate computations exactly as in the original solution.
        - Do not add new reasoning steps.
        - Do not include explanations, narration, or meta commentary.
        - Each line should be a single question followed by its answer.
        - Each line should be a question, such as What is X? followed by the answer.
        - Maintain the final answer format exactly as in the original solution.
        - The answer should only have the workings and final answer, nothing else.
        - There should be no reference to the original task, only include the
            converted socratic version.

        Output format:
        A numbered list where each item is:
        Question? Answer.

        Example:

        Original solution:
        Janet sells 16 - 3 - 4 = 9 duck eggs a day.
        She makes 9 * 2 = 18 dollars every day.
        #### 18

        Socratic worked version:
        1. How many duck eggs does Janet sell each day after subtracting 3 and 4 from 16? 16 - 3 - 4 = 9.
        2. How much money does Janet make if she sells 9 eggs at 2 dollars each? 9 x 2 = 18.
        #### 18
         """
    return model.generate_response(tokenizer, prompt, temperature=0.2)


def generate_new_dataset(
    model: Model, tokenizer: AutoTokenizer, dataset: GSM8KDataset
) -> Dataset:
    """Generate a new dataset by converting each task using the model."""
    converted_tasks = []

    for i, task in enumerate(dataset.yield_train_cases(10)):
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
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model = Model.create_model(model_name)
    tokenizer = Tokenizer.load_tokenizer(model_name)
    dataset = GSM8KDataset(**load_gsm8k())

    ds = generate_new_dataset(model, tokenizer, dataset)
    ds.to_json("data/converted_gsm8k.jsonl")

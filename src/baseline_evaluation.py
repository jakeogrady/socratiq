import argparse
import csv
import logging
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CSV_COLUMNS: list[str] = [
    "test_index",
    "correct_answer",
    "generated_answer",
    "is_correct",
    "raw_response",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument(
        "--test_cases", type=int, default=100, help="Number of test cases"
    )
    parser.add_argument(
        "--few_shot_num", type=int, default=4, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start evaluation from this index"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="evaluation_results.csv",
        help="CSV to append results",
    )
    parser.add_argument("--print_answer", action="store_true", default=False)
    parser.add_argument("--print_prompt", action="store_true")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Self-consistency samples per question",
    )
    parser.add_argument("--self_consistency", action="store_true", default=True)
    parser.add_argument(
        "--dataset_name", type=str, default="gsm8k", help="HF dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="HF dataset config"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column with questions"
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="answer",
        help="Column with correct answers",
    )

    return parser


def load_hf_dataset(
    dataset_name: str,
    split: str = "test",
    dataset_config: str | None = None,
) -> Dataset:
    """Load any Hugging Face dataset split as a Dataset object."""
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset


def generate_prompt(
    train_set: Dataset,
    test_set: Dataset,
    few_shot_num: int = 4,
    target_question_index: int = 0,
    text_column: str = "text",
    answer_column: str = "answer",
) -> str:
    """Generate a few-shot prompt for any HF dataset including answers in few-shots."""
    few_shot_texts: list[str] = []
    for shot_index in range(few_shot_num):
        example: dict = train_set[shot_index]
        question_text: str = example[text_column].strip()
        answer_text: str = str(example[answer_column]).strip()
        few_shot_texts.append(f"Question: {question_text}\nAnswer: {answer_text}")

    few_shot_block: str = "\n\n".join(few_shot_texts)
    instruction_block: str = (
        "You are a helpful math tutor. Solve the following problems step by step.\n\n"
    )
    target_question: str = test_set[target_question_index][text_column].strip()

    return (
        f"{instruction_block}{few_shot_block}\n\nQuestion: {target_question}\nAnswer:"
    )


def self_consistency_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    num_samples: int = 5,
) -> list[str]:
    """Generate multiple responses for self-consistency sampling."""
    responses: list[str] = []

    for _ in range(num_samples):
        sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20, min_p=0)
        response: str = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler
        )
        responses.append(response)

    return responses


def extract_final_number(
    text: str,
    answer_regex: str = r"[-+]?\d*\.?\d+",
) -> str | None:
    """Extract the final numeric answer from a model response, preferring the #### marker."""
    hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    if hash_match:
        return hash_match.group(1)
    matches: list[str] = re.findall(answer_regex, text)
    if matches:
        return matches[-1]
    return None


def majority_vote(answers: list[str]) -> str | None:
    """Return the most common answer in a list of candidate answers."""
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def validate_answer(generated_answer: str, correct_answer: str) -> bool:
    """Check if the generated answer matches the correct answer as integers."""
    try:
        return int(generated_answer) == int(correct_answer)
    except Exception:
        return False


def get_answer(dataset: Dataset, index: int, answer_column: str = "answer") -> str:
    """Retrieve the correct answer string for a given dataset row index."""
    return dataset[index][answer_column]


if __name__ == "__main__":
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    logger.info("Model Name: %s", args.model_name)

    if args.adapter_path:
        logger.info("Loading Adapters Directory...")
        model, tokenizer = load(args.model_name, adapter_path=args.adapter_path)
    else:
        logger.info("Loading base model (no adapters)")
        model, tokenizer = load(args.model_name)

    dataset: Dataset = load_hf_dataset(
        args.dataset_name,
        split=args.dataset_split,
        dataset_config=args.dataset_config,
    )

    safe_model_name: str = args.model_name.replace("/", "-")
    adapter_tag: str = Path(args.adapter_path).stem if args.adapter_path else ""
    dataset_tag: str = args.dataset_name.replace("/", "-")
    adapter_suffix: str = f"_{adapter_tag}" if adapter_tag else ""
    eval_filename: str = f"eval_results/{safe_model_name}_{dataset_tag}{adapter_suffix}_{args.num_samples}_{args.results_file}"
    logger.info("Model Filename %s", eval_filename)

    start: int
    try:
        existing_df: pd.DataFrame = pd.read_csv(eval_filename)
        start = max(args.start_index, len(existing_df))
        logger.info("Resuming from index %d", start)
    except Exception as resume_error:
        logger.warning("CSV corrupted — restarting from scratch %s", resume_error)
        start = args.start_index

    end: int = min(start + args.test_cases, len(dataset))
    logger.info("Evaluating test cases from %d to %d", start, end - 1)

    answer_correct: int = 0

    for question_index in range(start, end):
        text_prompt: str = generate_prompt(
            dataset,
            dataset,
            few_shot_num=args.few_shot_num,
            target_question_index=question_index,
            text_column=args.text_column,
            answer_column=args.answer_column,
        )

        logger.info("Evaluating question index %d", question_index)

        _generation_start: float = time.time()

        responses: list[str]
        if args.self_consistency:
            responses = self_consistency_generate(
                model, tokenizer, text_prompt, num_samples=args.num_samples
            )
        else:
            responses = [generate(model, tokenizer, prompt=text_prompt, max_tokens=256)]

        candidate_answers: list[str] = [
            extract_final_number(response)
            for response in responses
            if extract_final_number(response) is not None
        ]
        predicted_answer: str | None = majority_vote(candidate_answers)
        raw_answer: str = get_answer(
            dataset, question_index, answer_column=args.answer_column
        )
        correct_answer: str | None = extract_final_number(str(raw_answer))

        is_correct: bool = predicted_answer is not None and validate_answer(
            predicted_answer, correct_answer
        )
        if is_correct:
            answer_correct += 1

        result_file_exists: bool = Path(eval_filename).exists()

        with Path(eval_filename).open("a", newline="", encoding="utf-8") as csv_file:
            writer: csv.DictWriter = csv.DictWriter(
                csv_file, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL
            )

            if not result_file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "test_index": question_index,
                    "correct_answer": correct_answer,
                    "generated_answer": predicted_answer or "No Answer",
                    "is_correct": is_correct,
                    "raw_response": str(responses),
                }
            )

    results_df: pd.DataFrame = pd.read_csv(eval_filename)
    accuracy: float = results_df["is_correct"].mean() * 100
    logger.info("Total Correct Answers: %d/%d", answer_correct, args.test_cases)
    logger.info("Accuracy: %.2f%%", accuracy)

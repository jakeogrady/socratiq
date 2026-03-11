import argparse
import csv
import logging
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from src.constants import (
    ANSWER_REGEX,
    FEW_SHOT_NUM,
    QUESTION_REGEX,
    TEST_CASES,
)
from src.models import load_and_process_gsm8k
from src.summarize import summarize_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CSV_COLUMNS = [
    "test_index",
    "correct_answer",
    "generated_answer",
    "is_correct",
    "raw_response",
]


def generate_prompt(
    train_set: Dataset,
    test_set: Dataset,
    few_shot_num: int = 4,
    target_question_index: int = 0,
) -> str:
    """Generate a few-shot prompt for GSM8K evaluation."""
    few_shot_texts = train_set[:few_shot_num]["text"]
    few_shot_block = "\n\n".join(few_shot_texts)

    instruction_block = (
        "You are a helpful math tutor. Solve the following problems step by step.\n\n"
    )

    match = re.search(
        QUESTION_REGEX,
        test_set[target_question_index]["text"],
        re.DOTALL,
    )

    if not match:
        msg = "Could not extract question at index {target_question_index}"
        raise ValueError(msg)

    target_question = match.group(1).strip()

    return (
        instruction_block
        + few_shot_block
        + "\n\nQuestion: "
        + target_question
        + "\nAnswer:"
    )


def validate_answer(generated_answer: str, correct_answer: str) -> bool:
    """Validate if the generated answer matches the correct answer."""
    try:
        logger.info(
            "Validating Generated Answer: %s against Correct Answer: %s",
            generated_answer,
            correct_answer,
        )
        return int(generated_answer) == int(correct_answer)
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--test_cases",
        type=int,
        default=TEST_CASES,
        help="Number of test cases to evaluate",
    )
    parser.add_argument(
        "--few_shot_num",
        type=int,
        default=FEW_SHOT_NUM,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start evaluation from this test case index (0-based)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="evaluation_results.csv",
        help="File to append results to (CSV)",
    )
    parser.add_argument(
        "--print_answer",
        action="store_true",
        default=False,
        help="Whether to print the generated answers",
    )
    parser.add_argument(
        "--print_prompt", action="store_true", help="Whether to print the prompts"
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA qwen3_adapters directory",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of self-consistency samples per question",
    )

    parser.add_argument(
        "--self-consistency", action="store_true", default=True, help="Self Consistency"
    )

    return parser


def self_consistency_generate(model, tokenizer, prompt, num_samples=5):
    responses = []

    for _ in range(num_samples):
        sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20, min_p=0)

        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler
        )
        responses.append(response)

    return responses


def extract_final_number(text):
    matches = re.findall(ANSWER_REGEX, text)
    if matches:
        return matches[-1]
    return None


def majority_vote(answers):
    if not answers:
        return None

    return Counter(answers).most_common(1)[0][0]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Model Name: %s", args.model_name)

    if args.adapter_path:
        logger.info("Loading Adapters Directory...")
        model, tokenizer = load(args.model_name, adapter_path=args.adapter_path)

        logger.info("Loaded Fine-tuned model")
    else:
        logger.info("Loading base model (no adapters)")
        model, tokenizer = load(args.model_name)

    answer_correct = 0

    dataset = load_and_process_gsm8k()

    start = args.start_index
    eval_filename = f"{args.model_name.replace('/', '-')}_{args.results_file}"
    logger.info("Model Filename %s", eval_filename)

    try:
        df = pd.read_csv(eval_filename)
        start = max(start, len(df))
        logger.info("Resuming from index %d", start)
    except Exception as e:
        logger.warning("CSV corrupted — restarting from scratch %s", e)
        start = args.start_index

    end = min(start + args.test_cases, len(dataset.test))

    logger.info("Evaluating test cases from %d to %d", start, end - 1)

    for i in range(start, end):
        text_prompt = generate_prompt(
            dataset.train,
            dataset.test,
            few_shot_num=4,
            target_question_index=i,
        )

        chat = [{"role": "user", "content": text_prompt}]

        text_prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.info("Evaluating question index %d", i)

        generation_start = time.time()

        if args.self_consistency:
            responses = self_consistency_generate(
                model, tokenizer, text_prompt, num_samples=args.num_samples
            )
        else:
            responses = [
                generate(
                    model,
                    tokenizer,
                    prompt=text_prompt,
                    max_tokens=256,
                )
            ]

        logger.info("Response generated in %ss", time.time() - generation_start)

        answers = []

        for r in responses:
            num = extract_final_number(r)
            if num is not None:
                answers.append(num)

        predicted_answer = majority_vote(answers)
        correct_answer = dataset.get_test_case_answer(i)

        is_correct = False

        if predicted_answer is not None:
            if validate_answer(predicted_answer, correct_answer):
                is_correct = True
                answer_correct += 1

        file_exists = Path(eval_filename).exists()

        with Path(eval_filename).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)

            if not file_exists:
                writer.writeheader()
                file_exists = True

            writer.writerow(
                {
                    "test_index": i,
                    "correct_answer": correct_answer,
                    "generated_answer": predicted_answer or "No Answer",
                    "is_correct": is_correct,
                    "raw_response": str(responses),
                }
            )

    logger.info("Total Correct Answers: %d out of %d", answer_correct, args.test_cases)
    df = pd.read_csv(eval_filename)
    accuracy = df["is_correct"].mean() * 100

    if len(df) == len(dataset.test):
        summarize_results(eval_filename)

    logger.info("Accuracy: %.2f%%", accuracy)

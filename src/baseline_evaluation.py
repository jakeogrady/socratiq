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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CSV_COLUMNS = [
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
    dataset_name: str, split: str = "test", dataset_config: str = None
) -> Dataset:
    """Load any Hugging Face dataset split as a Dataset object."""
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset


def generate_prompt(
    train_set,
    test_set,
    few_shot_num: int = 4,
    target_question_index: int = 0,
    text_column: str = "text",
    answer_column: str = "answer",
) -> str:
    """Generate a few-shot prompt for any HF dataset including answers in few-shots."""
    few_shot_texts = []
    for i in range(few_shot_num):
        example = train_set[i]
        q = example[text_column].strip()
        a = str(example[answer_column]).strip()
        few_shot_texts.append(f"Question: {q}\nAnswer: {a}")

    few_shot_block = "\n\n".join(few_shot_texts)

    instruction_block = (
        "You are a helpful math tutor. Solve the following problems step by step.\n\n"
    )

    target_question = test_set[target_question_index][text_column].strip()

    prompt = (
        f"{instruction_block}{few_shot_block}\n\nQuestion: {target_question}\nAnswer:"
    )

    return prompt


def self_consistency_generate(model, tokenizer, prompt, num_samples=5):
    """Generate multiple responses for self-consistency sampling."""
    responses = []

    for _ in range(num_samples):
        sampler = make_sampler(temp=0.7, top_p=0.95, top_k=20, min_p=0)

        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, sampler=sampler
        )
        responses.append(response)

    return responses


def extract_final_number(text, answer_regex=r"[-+]?\d*\.?\d+"):
    """Extract the last number from a generated answer string."""
    matches = re.findall(answer_regex, text)
    if matches:
        return matches[-1]
    return None


def majority_vote(answers):
    """Return the most common answer in a list of answers."""
    if not answers:
        return None

    return Counter(answers).most_common(1)[0][0]


def validate_answer(generated_answer: str, correct_answer: str) -> bool:
    """Check if the generated answer matches the correct answer."""
    try:
        return int(generated_answer) == int(correct_answer)
    except Exception:
        return False


def get_answer(dataset, index: int, answer_column: str = "answer"):
    """Retrieve the correct answer for a dataset row."""
    return dataset[index][answer_column]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Model Name: %s", args.model_name)

    if args.adapter_path:
        logger.info("Loading Adapters Directory...")
        model, tokenizer = load(args.model_name, adapter_path=args.adapter_path)
    else:
        logger.info("Loading base model (no adapters)")
        model, tokenizer = load(args.model_name)

    # Load dataset
    dataset = load_hf_dataset(
        args.dataset_name,
        split=args.dataset_split,
        dataset_config=args.dataset_config,
    )

    # Setup CSV
    safe_model_name = args.model_name.replace("/", "-")
    eval_filename = (
        f"eval_results/{safe_model_name}_{args.dataset_name.replace('/', '-')}"
        f"{'_adapter' if args.adapter_path else ''}_{args.num_samples}_{args.results_file}"
    )
    logger.info("Model Filename %s", eval_filename)

    try:
        df = pd.read_csv(eval_filename)
        start = max(args.start_index, len(df))
        logger.info("Resuming from index %d", start)
    except Exception as e:
        logger.warning("CSV corrupted — restarting from scratch %s", e)
        start = args.start_index

    end = min(start + args.test_cases, len(dataset))
    logger.info("Evaluating test cases from %d to %d", start, end - 1)

    answer_correct = 0

    for i in range(start, end):
        text_prompt = generate_prompt(
            dataset,
            dataset,
            few_shot_num=args.few_shot_num,
            target_question_index=i,
            text_column=args.text_column,
            answer_column=args.answer_column,
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
            responses = [generate(model, tokenizer, prompt=text_prompt, max_tokens=256)]

        answers = [
            extract_final_number(r)
            for r in responses
            if extract_final_number(r) is not None
        ]
        predicted_answer = majority_vote(answers)
        correct_answer = get_answer(dataset, i, answer_column=args.answer_column)

        is_correct = predicted_answer is not None and validate_answer(
            predicted_answer, correct_answer
        )
        if is_correct:
            answer_correct += 1

        file_exists = Path(eval_filename).exists()

        with Path(eval_filename).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "test_index": i,
                    "correct_answer": correct_answer,
                    "generated_answer": predicted_answer or "No Answer",
                    "is_correct": is_correct,
                    "raw_response": str(responses),
                }
            )

    df = pd.read_csv(eval_filename)
    accuracy = df["is_correct"].mean() * 100
    logger.info("Total Correct Answers: %d/%d", answer_correct, args.test_cases)
    logger.info("Accuracy: %.2f%%", accuracy)
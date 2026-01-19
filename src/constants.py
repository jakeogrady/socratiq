"""Constants used across the project."""

OPENAI_GSM8K = "openai/gsm8k"
DATASET_FORMAT = "### Question\n{question}\n\n### Answer\n{answer}"
DATASET_FORMAT_PHI_2 = "Question: {question}\nAnswer: {answer}\n\n"
PHI_2 = "microsoft/phi-2"
QUESTION_PARSE_REGEX = r"Question:\s*(.*?)\nAnswer:"
ANSWER_REGEX = r"####\s*(-?\d+)"
FEW_SHOT_NUM = 4
TEST_CASES = 10
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

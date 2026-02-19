"""Constants used across the project."""

OPENAI_GSM8K = "openai/gsm8k"
DATASET_FORMAT = "Question:\n{question}\nAnswer:\n{answer}"
QUESTION_REGEX = r"Question:\s*(.*?)\nAnswer:"
ANSWER_REGEX = r"####\s*(-?\d+)"
FEW_SHOT_NUM = 4
TEST_CASES = 10
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_2_3B = "mlx-community/Llama-3.2-3B-8bit"
LLAMA_3_2_3B_INSTRUCT = "mlx-community/Llama-3.2-3B-Instruct"
LLAMA_3_2_1B_INSTRUCT = "mlx-community/Llama-3.2-1B-Instruct-MLXTuned"
MISTRAL_7B_Q4 = "mlx-community/Mistral-7B-Instruct-v0.3-8bit"
DATASET_CONVERSION_PROMPT = """
You are converting GSM8K-style math solutions into Socratic worked solutions.

Original solution (including final answer):
{task}


Instructions:

1. Rewrite the solution as a numbered list of Socratic questions.
2. Each numbered item must be exactly one sentence: a question immediately followed by its answer.
3. Each question must correspond to **one computation or statement** in the original solution.
4. Preserve all numbers and operations exactly — do not combine, split, or modify steps.
5. Do not add explanations or commentary.
6. At the very end, write the final answer on its own line **starting with #### followed by a space and the number**, exactly like this:

#### 42

Example:

Original solution:
He has 3 pencils and buys 2 more. He gives 1 to a friend. #### 4

Socratic worked version:
1. How many pencils does he have before giving any away? 3.
2. How many pencils does he have after buying 2 more? 3 + 2 = 5.
3. How many pencils does he have after giving 1 to a friend? 5 - 1 = 4.
#### 4

Remember:
- The numbered steps must correspond to the original solution.
- The final answer must always appear in the correct format as #### <number>.
- Do not include any text before or after the worked solution.

Important: At the end of your output, write the final numeric answer **on its own line** starting with #### followed by the number. Do not skip this line.
"""

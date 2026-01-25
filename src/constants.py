"""Constants used across the project."""

OPENAI_GSM8K = "openai/gsm8k"
DATASET_FORMAT = "Question:\n{question}\nAnswer:\n{answer}"
QUESTION_REGEX = r"Question:\s*(.*?)\nAnswer:"
ANSWER_REGEX = r"####\s*(-?\d+)"
FEW_SHOT_NUM = 4
TEST_CASES = 10
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MISTRAL_7B_Q4 = "mlx-community/Mistral-7B-Instruct-v0.3-8bit"
DATASET_CONVERSION_PROMPT = """
    You are converting GSM8K-style math solutions into Socratic worked solutions.

    Task:
    {task}

    Rules (follow strictly):
    1. Rewrite the solution as a numbered list of Socratic questions.
    2. Each numbered item must be exactly one sentence:
       - A question immediately followed by its answer.
    3. Every question must directly correspond to one computation or statement
       in the original solution.
    4. Preserve all mathematical expressions exactly as they appear:
       - Same numbers
       - Same operations
       - Same order
       - Same results
    5. Do not introduce any new steps.
    6. Do not combine or split steps.
    7. Do not simplify expressions unless they were simplified in the original.
    8. Do not add explanations, commentary, or rephrasing beyond turning
       statements into questions.
    9. Use plain math symbols only ( + − * ÷ = ).
    10. Do not use << >> notation unless it appears in the original.
    11. Do not restate the original problem.
    12. Do not include any text before or after the worked solution.
    13. The final answer must appear exactly once, on its own line, formatted as:
        #### <number>

    Output format (strict):
    1. Question? Answer.
    2. Question? Answer.
    ...
    #### <number>

    Examples:

    Original solution:
    Janet sells 16 - 3 - 4 = 9 duck eggs a day.
    She makes 9 * 2 = 18 dollars every day.
    #### 18

    Socratic worked version:
    1. How many duck eggs does Janet sell each day after subtracting 3 and 4 from 16? 16 - 3 - 4 = 9.
    2. How much money does Janet make if she sells 9 eggs at 2 dollars each? 9 * 2 = 18.
    #### 18

    Original solution:
    He writes each friend 3*2=6 pages a week.
    So he writes 6*2=12 pages every week.
    That means he writes 12*52=624 pages a year.
    #### 624

    Socratic worked version:
    1. How many pages does he write for each friend if he writes 3 pages twice a week? 3 * 2 = 6.
    2. How many pages does he write each week if he writes 6 pages for each friend and has 2 friends? 6 * 2 = 12.
    3. How many pages does he write in a year if he writes 12 pages every week for 52 weeks? 12 * 52 = 624.
    #### 624
 """

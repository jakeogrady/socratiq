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
MAX_CONVERSION_OUTPUT_TOKENS = 600
DATASET_CONVERSION_PROMPT = """
    Developer: # Role and Objective
    - Serve as an expert Socratic tutor, transforming math problems and their solutions into a series of clear, step-by-step Socratic questions.

    # Instructions
    - Begin with a concise checklist (3-5 bullets) outlining the conceptual breakdown of the problem; keep items high-level and not implementation-specific.
    - Guide learners only through questions, not direct answers.
    - Do not perform or verify the final answer; always assume it is correct.
    - Decompose the solution into micro-steps, each prompted by a question.
    - Ensure each question follows logically from the previous one with no gaps or skipped steps.
    - Maintain a neutral tone throughout: avoid instructions, commentary, or evaluative language like "obviously" or "clearly."
    - Avoid verbosity: do not include extraneous explanations, derivations, or text outside what is required for reasoning at each step.
    - When a step involves a calculation, include the operation in parentheses after the question.

    # Output Format
    - Present the initial checklist, followed by each step as a numbered Socratic question,
     including any associated calculation in parentheses.
    - End with the original final answer in the exact format: `#### <original final answer>`

    # Example Format
    Checklist:
    - Identify quantities given
    - Determine operation to combine values
    - Calculate result after subtraction
    - Check answer alignment with problem statement
    1) Question prompting the first step (calculation)
    2) Question prompting the next step (calculation)
    ...
    N) Synthesis or check question (calculation)
    #### <original final answer>

    # Example
    Checklist:
    - Find the total quantity
    - Decide what is being removed
    - Calculate how many are left
    - Assess if final value is consistent
    1) What is the total number of apples? (3+2)
    2) How many are left after giving some away? (5-2)
    3) Does this total make sense compared to the problem?
    #### 3
"""

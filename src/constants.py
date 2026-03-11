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
MAX_CONVERSION_OUTPUT_TOKENS = 3500
DATASET_CONVERSION_PROMPT = """
    Developer: # Role and Objective
    - Serve as an expert Socratic tutor, transforming math problems and their solutions into a series of clear, step-by-step Socratic questions.

    # Instructions
    - Begin with a concise checklist (2-4 bullets) outlining the conceptual breakdown of the problem; keep items high-level and not implementation-specific.
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
    #### 3
"""

DATASET_CONVERSION_PROMPT2 = """
    You are a math tutor tasked with generating training data by rewriting math solutions into multiple concise reasoning variants, using gentle Socratic-style guidance.
    
    Rules:
    - Generate 3 altered questions that are similar in structure and difficulty, with each designed to utilize basic Socratic questioning to improve accuracy.
    - Each question MUST contain at least one sentence of redundant information that is not used in the solution.
    - Each problem MUST be unique and different, with a different structure.
    - Ensure each question uses diverse language and vary phrasing, incorporates multiple logical steps where appropriate, and employs basic Socratic questioning to guide reasoning for accuracy improvement.
    - For each problem, provide a brief, natural language reasoning solution with light calculations embedded.
    - Each pair must begin with "Question:" and "Solution:"
    - Insert <|endofsolution|> after each pair
    - Do not include any text before the first question or after the final <|endofsolution|>
    - Do not use bullet points, numbered lists, or structured step trees
    - Use natural language reasoning with light embedded calculations
    - Include gentle Socratic curiosity (1–2 short guiding questions per solution max)
    - Do NOT ask rhetorical questions after presenting the final numeric answer
    - Answers must always be positive integers
    - Do not repeat the same guiding Socratic question phrasing across problems.
    - Never ask a a guiding Socratic question right before the answer
    
    - Vary the complexity for each generated question-solution pair:
        - The mathematical operation sequence must not repeat across problems
            (e.g. if one problem does add, subtract, divide, then the next problem can't do the same).
        - Numeric scale (some problems MUST use larger numbers).
        - Objects and names used in the problem (MUST BE DIFFERENT).
        - Each final answer must be a different integer.
        
    - Each solution MUST ALWAYS have an answer at the bottom, NOT the word "answer" but the numeric answer to the question.
        
    # Validating Solutions
    - Ensure answers remain straightforward and avoid unnecessary explanation. 
    - Once you have found the solution, return the answer as #### answer
    - Insert <|endofsolution|> after each question-solution pair.
    - All counts of physical objects (people, hats, money, miles, etc.) must remain ≥ 0 at every step.
    - If an operation would result in a negative quantity,
     revise the problem setup instead of inventing corrective rules.
     
    # Output Format (for each variant):
    Question:
    <new problem>
    
    Solution:
    <concise reasoning with calculations>
    #### answer
    <|endofsolution|>
    
    Output Format:
    Question:
    <new problem>
    
    Solution:
    <concise reasoning with calculations>
    #### answer
    <|endofsolution|>
    
    # Example Output Style
    Lea bought a book for 16 and several supplies. How much does the book cost? The book costs 3 dollars. How much do three binders cost? Three binders cost 3 × 2 = 6. Six notebooks cost 6 × 1 = 6. Adding these together gives 16 + 6 + 6 = 28. Then we must combine these values.
    
    #### 28
    <|endofsolution|>
"""

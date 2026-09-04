# Socratiq

Fine-tuning small language models to solve grade-school math problems using Socratic reasoning. Built on the GSM8K benchmark, this project converts standard math solutions into step-by-step Socratic question-answer pairs, fine-tunes Qwen3 models via LoRA, and evaluates accuracy with self-consistency sampling.

---

## Results

| Model | Base Accuracy | Fine-tuned Accuracy | Gain |
|---|---|---|---|
| Qwen3-0.6B | 36.47% | 49.05% | +12.58pp |
| Qwen3-1.7B | 53.53% | 66.49% | +12.96pp |

Evaluated on GSM8K, 4-shot, 1 sample.

---

## How It Works

**1. Dataset Generation** — GSM8K solutions are sent to GPT via the OpenAI Batch API and rewritten into Socratic question-solution pairs (`openai_conversion.py`). Each output is cleaned, deduplicated, and split into train/validation sets.

**2. Data Cleaning** — Rhetorical questions that sit immediately before the `####` answer marker are removed, and each example is flattened into a single `text` field for training (`dataset_improvement.py`). Questions inside the reasoning chain are the point of the Socratic format and are preserved. This produces `new_data_text/`, which the Qwen3-1.7B runs 5–8 were trained on.

**3. Fine-tuning** — LoRA adapters are trained on the converted dataset using `mlx-lm` on Apple Silicon. Training logs are parsed and visualised to find the best checkpoint (`val_loss.py`).

**4. Evaluation** — Models are evaluated on GSM8K with optional self-consistency sampling (majority vote over multiple samples) and results written to CSV (`baseline_evaluation.py`).

**5. Figures** — All paper figures are generated from `generate_figures.py` and `lora_figure.py`, saved as both PDF and PNG under `outputs/`.

---

## Project Structure

```
socratiq/
├── src/
│   ├── __init__.py
│   ├── constants.py            # Prompts, regex patterns, model paths
│   ├── models.py               # GSM8K dataset loader and preprocessor
│   ├── baseline_evaluation.py  # Model evaluation with self-consistency
│   ├── dataset_improvement.py  # Data cleaning pipeline
│   ├── generate_figures.py     # Result figures (bar charts, line plots)
│   ├── lora_figure.py          # LoRA architecture diagram
│   ├── openai_conversion.py    # GPT batch API dataset conversion
│   ├── val_loss.py             # Training log parser and ASCII loss curve
│   └── outputs/                # Saved figures (.pdf + .png)
├── *.yaml                      # 20 LoRA training configs
├── new_data/                   # Converted dataset, {question, answer}
├── new_data_text/              # Cleaned dataset, single {text} field
├── Makefile
└── pyproject.toml
```

---

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets **Python 3.13+**. Fine-tuning with `mlx-lm` requires **Apple Silicon** (M1 or later).

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
git clone https://github.com/jakeogrady/socratiq.git
cd socratiq

# Install all dependencies (including dev group)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Environment Variables

Copy the template and fill in your own credentials:

```bash
cp .env.example .env
```

| Variable | Required for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `make conversion` | Socratic dataset generation through the Batch API. Training and evaluation never read it. |
| `HF_TOKEN` | Gated model downloads | Only needed if a config points at a gated repository such as `meta-llama/*`. The `mlx-community` mirrors used throughout the Makefile are public. |

`.env` is git-ignored and must never be committed.

---

## Usage

All common tasks are driven by `make`. Commands use `caffeinate` to prevent macOS sleep during long runs.

### 1. Generate the Socratic dataset

Converts the GSM8K train split into Socratic QA pairs via the OpenAI Batch API. Submits in chunks of 1,000, waits for each to complete, then merges and splits into `new_data/train.jsonl` and `new_data/valid.jsonl`.

```bash
make conversion
```

### 2. Clean the dataset

Removes rhetorical questions sitting immediately before the `####` marker and flattens each example into a single `text` field. Output is written to `new_data_text/`.

```bash
uv run python src/dataset_improvement.py
```

`new_data_text/` already exists and is the data the published models were trained on, so the script refuses to overwrite it without `--force`. To check that the transformation still reproduces it:

```bash
uv run python src/dataset_improvement.py --verify
```

This reports 21,240 of 21,250 examples reproduced (99.953%). The 10 exceptions are rows where the original transformation truncated mid-word — for example `train` row 632 ends `"= 0.75 ####"` where the source reads `"= 0.75W look consistent with a 25% decrease?"`. They are defects in the shipped data and are not reproduced.

### 3. Fine-tune with LoRA

Trains LoRA adapters using `mlx-lm`. Logs are timestamped and saved to `logs/`.

```bash
make train CONFIG=qwen-1.7-lora-config-run-6.yaml
```

### 4. Inspect training loss

Parses a training log and prints a loss table with directional change indicators and the best checkpoint, plus an ASCII loss curve.

```bash
make val-loss LOG_FILE=logs/train_20250101_120000_qwen3_0.6b.yaml.log
```

### 5. Evaluate a model

**Single benchmark run:**

```bash
# GSM8K — base model
make baseline-eval-gsm8k MODEL_NAME=mlx-community/Qwen3-0.6B NUM_SAMPLES=4

# GSM8K — with LoRA adapter
make baseline-eval-gsm8k MODEL_NAME=mlx-community/Qwen3-0.6B ADAPTER_PATH=adapters/run2 NUM_SAMPLES=4

# SVAMP or MultiArith
make baseline-eval-svamp MODEL_NAME=mlx-community/Qwen3-0.6B NUM_SAMPLES=4
make baseline-eval-multiarith MODEL_NAME=mlx-community/Qwen3-0.6B NUM_SAMPLES=4
```

**Self-consistency sweep** (runs n=1, 2, 4 automatically):

```bash
make loop-eval-gsm8k MODEL_NAME=mlx-community/Qwen3-0.6B ADAPTER_PATH=adapters/run2
make loop-eval-svamp MODEL_NAME=mlx-community/Qwen3-0.6B
make loop-eval-multiarith MODEL_NAME=mlx-community/Qwen3-0.6B
```

Results are appended to a CSV under `eval_results/` and can be resumed mid-run by re-running the same command.

### 6. Generate figures

```bash
uv run python src/generate_figures.py  # Result figures → outputs/fig1_*.pdf/png
uv run python src/lora_figure.py       # LoRA architecture diagram → outputs/fig_lora_architecture.*
```

---

## Development

```bash
# Format and lint (ruff)
make lint

# Run all pre-commit hooks
make pre-commit-all
```

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, targeting Python 3.13 with a strict ruleset.

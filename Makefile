lint:
	uv run ruff format
	uv run ruff check --fix


pre-commit-all:
	uv run pre-commit run --all-files


baseline-eval:
	uv run python3 src/baseline_evaluation.py

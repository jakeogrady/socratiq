lint:
	uv run ruff format
	uv run ruff check --fix


pre-commit-all:
	uv run pre-commit run --all-files

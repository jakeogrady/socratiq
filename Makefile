lint:
	uv run ruff format
	uv run ruff check --fix


pre-commit-all:
	uv run pre-commit run --all-files

# Run with make baseline-eval START_INDEX=100
baseline-eval:
	caffeinate -s uv run python3 src/baseline_evaluation.py \
	--test_cases ${TEST_CASES} \
	--start_index ${START_INDEX} \
	--model_name ${MODEL_NAME}

conversion:
	caffeinate -s uv run python3 src/openai_conversion.py

train:
	caffeinate -s uv run python3 src/train.py \
	--model_name ${MODEL_NAME} \
	--lr ${LR} \
	--rank ${RANK} \
	--results_file ../data/summary.jsonl

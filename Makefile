lint:
	uv run ruff format
	uv run ruff check --fix


pre-commit-all:
	uv run pre-commit run --all-files

# Run with make baseline-eval START_INDEX=100
ADAPTER_PATH?=
NUM_SAMPLES?=
baseline-eval:
	caffeinate -s uv run python3 src/baseline_evaluation.py \
	--test_cases ${TEST_CASES} \
	--start_index ${START_INDEX} \
	--model_name ${MODEL_NAME} \
	$(if ${ADAPTER_PATH},--adapter_path ${ADAPTER_PATH}) \
	$(if ${NUM_SAMPLES},--num_samples ${NUM_SAMPLES}) \

conversion:
	caffeinate -s uv run python3 src/openai_conversion.py

CONFIG?=
train:
	caffeinate -s uv run mlx_lm.lora --config ${CONFIG}

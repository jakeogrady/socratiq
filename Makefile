lint:
	uv run ruff format
	uv run ruff check --fix


pre-commit-all:
	uv run pre-commit run --all-files

ADAPTER_PATH?=
NUM_SAMPLES?=
FEW_SHOT_NUM?=
TEXT_COLUMN?=
ANSWER_COLUMN?=
CONFIG?=

baseline-eval-gsm8k:
	caffeinate -s uv run python3 src/baseline_evaluation.py \
	--test_cases 1320 \
	--start_index 0 \
	--model_name ${MODEL_NAME} \
	$(if ${FEW_SHOT_NUM}, --few_shot_num ${FEW_SHOT_NUM}) \
	$(if ${ADAPTER_PATH},--adapter_path ${ADAPTER_PATH}) \
	$(if ${NUM_SAMPLES},--num_samples ${NUM_SAMPLES}) \
	--text_column "question" \
	--answer_column "answer" \
	--dataset_config "main" \
	--dataset_name="openai/gsm8k"

baseline-eval-svamp:
	caffeinate -s uv run python3 src/baseline_evaluation.py \
	--test_cases 300 \
	--start_index 0 \
	--model_name ${MODEL_NAME} \
	$(if ${FEW_SHOT_NUM}, --few_shot_num ${FEW_SHOT_NUM}) \
	$(if ${ADAPTER_PATH},--adapter_path ${ADAPTER_PATH}) \
	$(if ${NUM_SAMPLES},--num_samples ${NUM_SAMPLES}) \
	--text_column="question_concat" \
	--answer_column="Answer" \
	--dataset_name="ChilleD/SVAMP"

baseline-eval-multiarith:
	caffeinate -s uv run python3 src/baseline_evaluation.py \
	--test_cases 180 \
	--start_index 0 \
	--model_name ${MODEL_NAME} \
	$(if ${FEW_SHOT_NUM}, --few_shot_num ${FEW_SHOT_NUM}) \
	$(if ${ADAPTER_PATH},--adapter_path ${ADAPTER_PATH}) \
	$(if ${NUM_SAMPLES},--num_samples ${NUM_SAMPLES}) \
	--text_column="question" \
	--answer_column="final_ans" \
	--dataset_name="ChilleD/MultiArith"

loop-eval-gsm8k:
	for n in 1 2 4; do \
		$(MAKE) baseline-eval-gsm8k \
		MODEL_NAME=${MODEL_NAME} \
		ADAPTER_PATH=${ADAPTER_PATH} \
		FEW_SHOT_NUM=4 \
		NUM_SAMPLES=$$n; \
	done

loop-eval-svamp:
	for n in 1 2 4; do \
		$(MAKE) baseline-eval-svamp \
		MODEL_NAME=${MODEL_NAME} \
		ADAPTER_PATH=${ADAPTER_PATH} \
		FEW_SHOT_NUM=0 \
		NUM_SAMPLES=$$n; \
	done

loop-eval-multiarith:
	for n in 1 2 4; do \
		$(MAKE) baseline-eval-multiarith \
		MODEL_NAME=${MODEL_NAME} \
		ADAPTER_PATH=${ADAPTER_PATH} \
		FEW_SHOT_NUM=0 \
		NUM_SAMPLES=$$n; \
	done

conversion:
	caffeinate -s uv run python3 src/openai_conversion.py

CONFIG ?=
train:
	mkdir -p logs; \
	LOGFILE="logs/train_$$(date +%Y%m%d_%H%M%S)_$(CONFIG).log"; \
	caffeinate -s uv run mlx_lm.lora --config $(CONFIG) 2>&1 | tee "$$LOGFILE"

val-loss:
	uv run python3 src/val_loss.py $(LOG_FILE)

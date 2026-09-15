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


# Reviewer-rerun workflow. These targets use the small Python 3.13 preparation
# environment until the full locked ML stack can be installed safely.
RERUN_PYTHON?=.venv/bin/python
RERUN_RUFF?=.venv/bin/ruff
RERUN_CONFIG_DIR?=configs/reviewer_rerun

rerun-bootstrap:
	./scripts/bootstrap_m4.sh

rerun-test:
	$(RERUN_PYTHON) -m unittest discover -s tests -v

rerun-lint:
	$(RERUN_RUFF) format --check \
		src/rerun_utils.py \
		src/paired_dataset.py \
		src/openai_conversion_v2.py \
		src/evaluate_v2.py \
		src/run_training.py \
		src/summarize_results.py \
		tests
	$(RERUN_RUFF) check \
		src/rerun_utils.py \
		src/paired_dataset.py \
		src/openai_conversion_v2.py \
		src/evaluate_v2.py \
		src/run_training.py \
		src/summarize_results.py \
		tests

rerun-validate-configs:
	$(RERUN_PYTHON) -m src.run_training validate \
		$(RERUN_CONFIG_DIR)/qwen3_0.6b_socratic.yaml \
		$(RERUN_CONFIG_DIR)/qwen3_0.6b_non_socratic.yaml \
		--pair
	$(RERUN_PYTHON) -m src.run_training validate \
		$(RERUN_CONFIG_DIR)/qwen3_1.7b_socratic.yaml \
		$(RERUN_CONFIG_DIR)/qwen3_1.7b_non_socratic.yaml \
		--pair
	$(RERUN_PYTHON) -m src.run_training validate \
		$(RERUN_CONFIG_DIR)/llama3.2_1b_socratic.yaml

rerun-environment:
	$(RERUN_PYTHON) -m src.run_training environment \
		--output results/reviewer_rerun/reproducibility/environment_preflight.json

rerun-dry-run-training:
	$(RERUN_PYTHON) -m src.run_training run \
		--config $(RERUN_CONFIG_DIR)/qwen3_0.6b_socratic.yaml \
		--experiment-id qwen3-0.6b-socratic \
		--dry-run
	$(RERUN_PYTHON) -m src.run_training run \
		--config $(RERUN_CONFIG_DIR)/qwen3_0.6b_non_socratic.yaml \
		--experiment-id qwen3-0.6b-non-socratic \
		--dry-run
	$(RERUN_PYTHON) -m src.run_training run \
		--config $(RERUN_CONFIG_DIR)/llama3.2_1b_socratic.yaml \
		--experiment-id llama3.2-1b-socratic \
		--dry-run

rerun-check: rerun-test rerun-lint rerun-validate-configs

# Usage: make rerun-batch-estimate BATCH_INPUT=path/to/batch.jsonl
rerun-batch-estimate:
	@test -n "$(BATCH_INPUT)" || (echo "BATCH_INPUT is required" && exit 2)
	$(RERUN_PYTHON) -m src.openai_conversion_v2 estimate --input $(BATCH_INPUT)

rerun-plan-matrices:
	./scripts/run_training_matrix.sh --full --mandatory --plan
	./scripts/run_evaluation_matrix.sh --full --greedy --mandatory --plan

rerun-reports:
	$(RERUN_PYTHON) -m src.summarize_results --require-mandatory-matrix

rerun-reports-partial:
	$(RERUN_PYTHON) -m src.summarize_results --allow-partial

rerun-smoke-reports:
	$(RERUN_PYTHON) -m src.summarize_results \
		--evaluation-root runs/reviewer_rerun/smoke/evaluation \
		--training-root runs/reviewer_rerun/smoke/training \
		--output-root results/reviewer_rerun/smoke \
		--allow-partial

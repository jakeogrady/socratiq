#!/bin/sh

# Plan or execute the corrected reviewer evaluation matrix. Planning is the
# default. Re-running an executed matrix safely resumes per-example outputs
# when the existing configuration hash matches.

set -eu

ACTION="plan"
STAGE="full"
MODE="greedy"
INCLUDE_QWEN_17="0"
FULL_OUTPUT_ROOT=${SOCRATIQ_EVALUATION_OUTPUT_ROOT:-runs/reviewer_rerun/evaluation}
SMOKE_OUTPUT_ROOT=${SOCRATIQ_EVALUATION_SMOKE_OUTPUT_ROOT:-runs/reviewer_rerun/smoke/evaluation}
SMOKE_USE_FULL_ADAPTERS=${SOCRATIQ_SMOKE_USE_FULL_ADAPTERS:-0}

case "$SMOKE_USE_FULL_ADAPTERS" in
    0|1) ;;
    *)
        printf '%s\n' "ERROR: SOCRATIQ_SMOKE_USE_FULL_ADAPTERS must be 0 or 1." >&2
        exit 2
        ;;
esac

usage() {
    printf '%s\n' \
        "Usage: ./scripts/run_evaluation_matrix.sh [--full|--smoke]" \
        "       [--greedy|--sc5] [--mandatory|--include-qwen-1.7b]" \
        "       [--plan|--execute]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --plan) ACTION="plan" ;;
        --execute) ACTION="execute" ;;
        --full) STAGE="full" ;;
        --smoke) STAGE="smoke" ;;
        --greedy) MODE="greedy" ;;
        --sc5) MODE="sc5" ;;
        --mandatory) INCLUDE_QWEN_17="0" ;;
        --include-qwen-1.7b) INCLUDE_QWEN_17="1" ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '%s\n' "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    printf '%s\n' "ERROR: run ./scripts/bootstrap_m4.sh first." >&2
    exit 1
fi
if [ "$(git branch --show-current)" != "reviewer-rerun" ]; then
    printf '%s\n' "ERROR: evaluation matrix requires branch reviewer-rerun." >&2
    exit 1
fi

run_one() {
    experiment_id=$1
    condition_path=$2
    model=$3
    adapter_path=$4
    benchmark=$5

    if [ "$STAGE" = "smoke" ]; then
        output_root="$SMOKE_OUTPUT_ROOT"
        limit_args="--limit 20"
        actual_experiment_id="$experiment_id-smoke"
    else
        output_root="$FULL_OUTPUT_ROOT"
        limit_args=""
        actual_experiment_id="$experiment_id"
    fi

    if [ "$MODE" = "greedy" ]; then
        mode_args="--mode greedy --samples 1 --temperature 0.0 --top-p 1.0 --top-k 0"
        mode_path="greedy"
    else
        mode_args="--mode self_consistency --samples 5 --temperature 0.7 --top-p 0.95 --top-k 20"
        mode_path="sc5"
    fi

    run_dir="$output_root/$condition_path/$benchmark/$mode_path"
    set -- \
        "$PYTHON" -m src.evaluate_v2 \
        --experiment-id "$actual_experiment_id" \
        --model "$model"
    if [ -n "$adapter_path" ]; then
        if [ "$ACTION" = "execute" ] && [ ! -f "$adapter_path/adapters.safetensors" ]; then
            printf '%s\n' "ERROR: adapter weights are missing: $adapter_path" >&2
            exit 1
        fi
        set -- "$@" --adapter-path "$adapter_path"
    fi
    # The following word splitting is intentional: each string contains only
    # fixed, repository-owned option/value pairs.
    # shellcheck disable=SC2086
    set -- "$@" --benchmark "$benchmark" $mode_args --max-tokens 512 --seed 42
    if [ -n "$limit_args" ]; then
        # shellcheck disable=SC2086
        set -- "$@" $limit_args
    fi
    set -- "$@" --run-dir "$run_dir"

    if [ "$ACTION" = "plan" ]; then
        printf '%s' "PLAN:"
        for argument in "$@"; do
            printf ' %s' "$argument"
        done
        printf '\n'
    else
        "$@"
    fi
}

run_condition() {
    experiment_id=$1
    condition_path=$2
    model=$3
    full_adapter=$4

    adapter_path="$full_adapter"
    if [ "$STAGE" = "smoke" ] && \
        [ "$SMOKE_USE_FULL_ADAPTERS" != "1" ] && \
        [ -n "$full_adapter" ]; then
        adapter_path="runs/reviewer_rerun/smoke/training/$condition_path/smoke_adapter"
    fi
    for benchmark in gsm8k multiarith svamp; do
        run_one \
            "$experiment_id" \
            "$condition_path" \
            "$model" \
            "$adapter_path" \
            "$benchmark"
    done
}

run_condition \
    qwen3-0.6b-base \
    qwen3_0.6b_base \
    mlx-community/Qwen3-0.6B-bf16 \
    ""
run_condition \
    qwen3-0.6b-socratic \
    qwen3_0.6b_socratic \
    mlx-community/Qwen3-0.6B-bf16 \
    runs/reviewer_rerun/training/qwen3_0.6b_socratic/adapter
run_condition \
    qwen3-0.6b-non-socratic \
    qwen3_0.6b_non_socratic \
    mlx-community/Qwen3-0.6B-bf16 \
    runs/reviewer_rerun/training/qwen3_0.6b_non_socratic/adapter
run_condition \
    llama3.2-1b-base \
    llama3.2_1b_base \
    mlx-community/Llama-3.2-1B-Instruct-MLXTuned \
    ""
run_condition \
    llama3.2-1b-socratic \
    llama3.2_1b_socratic \
    mlx-community/Llama-3.2-1B-Instruct-MLXTuned \
    runs/reviewer_rerun/training/llama3.2_1b_socratic/adapter

if [ "$INCLUDE_QWEN_17" = "1" ]; then
    run_condition \
        qwen3-1.7b-base \
        qwen3_1.7b_base \
        mlx-community/Qwen3-1.7B-4bit \
        ""
    run_condition \
        qwen3-1.7b-socratic \
        qwen3_1.7b_socratic \
        mlx-community/Qwen3-1.7B-4bit \
        runs/reviewer_rerun/training/qwen3_1.7b_socratic/adapter
    run_condition \
        qwen3-1.7b-non-socratic \
        qwen3_1.7b_non_socratic \
        mlx-community/Qwen3-1.7B-4bit \
        runs/reviewer_rerun/training/qwen3_1.7b_non_socratic/adapter
fi

printf '%s\n' \
    "Evaluation matrix $ACTION completed: stage=$STAGE mode=$MODE include_qwen_1.7b=$INCLUDE_QWEN_17 output_root=$output_root"

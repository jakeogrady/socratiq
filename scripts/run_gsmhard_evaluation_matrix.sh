#!/bin/sh

# Plan or execute the isolated GSM-Hard extension across all eight completed
# reviewer-rerun conditions. Planning is the default; execution safely resumes
# only when an existing run has the same complete configuration hash.

set -eu

ACTION="plan"
STAGE="full"
MODE="greedy"
FULL_OUTPUT_ROOT=${SOCRATIQ_GSMHARD_OUTPUT_ROOT:-runs/reviewer_rerun/evaluation_gsmhard_v1}
SMOKE_OUTPUT_ROOT=${SOCRATIQ_GSMHARD_SMOKE_OUTPUT_ROOT:-runs/reviewer_rerun/evaluation_gsmhard_v1_smoke}
PROTOCOL_EXTENSION="configs/reviewer_rerun/extensions/gsmhard_extension_v1.yaml"
ADAPTER_CHECKSUMS="configs/reviewer_rerun/gsmhard_adapters.sha256"

usage() {
    printf '%s\n' \
        "Usage: ./scripts/run_gsmhard_evaluation_matrix.sh [--full|--smoke]" \
        "       [--greedy|--sc5] [--plan|--execute]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --plan) ACTION="plan" ;;
        --execute) ACTION="execute" ;;
        --full) STAGE="full" ;;
        --smoke) STAGE="smoke" ;;
        --greedy) MODE="greedy" ;;
        --sc5) MODE="sc5" ;;
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
    printf '%s\n' "ERROR: GSM-Hard evaluation requires branch reviewer-rerun." >&2
    exit 1
fi
if [ ! -f "$PROTOCOL_EXTENSION" ]; then
    printf '%s\n' "ERROR: protocol extension is missing: $PROTOCOL_EXTENSION" >&2
    exit 1
fi
if [ "$ACTION" = "execute" ]; then
    if [ ! -f "$ADAPTER_CHECKSUMS" ]; then
        printf '%s\n' "ERROR: adapter checksum manifest is missing: $ADAPTER_CHECKSUMS" >&2
        exit 1
    fi
    shasum -a 256 -c "$ADAPTER_CHECKSUMS"
fi

run_one() {
    experiment_id=$1
    condition_path=$2
    model=$3
    adapter_path=$4

    if [ "$STAGE" = "smoke" ]; then
        output_root="$SMOKE_OUTPUT_ROOT"
        limit_args="--limit 20"
        actual_experiment_id="$experiment_id-gsmhard-smoke"
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

    run_dir="$output_root/$condition_path/gsm_hard/$mode_path"
    set -- \
        "$PYTHON" -m src.evaluate_v2 \
        --experiment-id "$actual_experiment_id" \
        --model "$model"
    if [ -n "$adapter_path" ]; then
        if [ "$ACTION" = "execute" ] && [ ! -f "$adapter_path/adapters.safetensors" ]; then
            printf '%s\n' "ERROR: final adapter weights are missing: $adapter_path" >&2
            exit 1
        fi
        set -- "$@" --adapter-path "$adapter_path"
    fi
    # The following word splitting is intentional: each value contains only
    # fixed, repository-owned option/value pairs.
    # shellcheck disable=SC2086
    set -- \
        "$@" \
        --benchmark gsm_hard \
        --protocol-extension "$PROTOCOL_EXTENSION" \
        $mode_args \
        --max-tokens 512 \
        --seed 42
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

run_one \
    qwen3-0.6b-base \
    qwen3_0.6b_base \
    mlx-community/Qwen3-0.6B-bf16 \
    ""
run_one \
    qwen3-0.6b-socratic \
    qwen3_0.6b_socratic \
    mlx-community/Qwen3-0.6B-bf16 \
    runs/reviewer_rerun/training/qwen3_0.6b_socratic/adapter
run_one \
    qwen3-0.6b-non-socratic \
    qwen3_0.6b_non_socratic \
    mlx-community/Qwen3-0.6B-bf16 \
    runs/reviewer_rerun/training/qwen3_0.6b_non_socratic/adapter
run_one \
    qwen3-1.7b-base \
    qwen3_1.7b_base \
    mlx-community/Qwen3-1.7B-4bit \
    ""
run_one \
    qwen3-1.7b-socratic \
    qwen3_1.7b_socratic \
    mlx-community/Qwen3-1.7B-4bit \
    runs/reviewer_rerun/training/qwen3_1.7b_socratic/adapter
run_one \
    qwen3-1.7b-non-socratic \
    qwen3_1.7b_non_socratic \
    mlx-community/Qwen3-1.7B-4bit \
    runs/reviewer_rerun/training/qwen3_1.7b_non_socratic/adapter
run_one \
    llama3.2-1b-base \
    llama3.2_1b_base \
    mlx-community/Llama-3.2-1B-Instruct-MLXTuned \
    ""
run_one \
    llama3.2-1b-socratic \
    llama3.2_1b_socratic \
    mlx-community/Llama-3.2-1B-Instruct-MLXTuned \
    runs/reviewer_rerun/training/llama3.2_1b_socratic/adapter

printf '%s\n' \
    "GSM-Hard matrix $ACTION completed: stage=$STAGE mode=$MODE runs=8 output_root=$output_root"

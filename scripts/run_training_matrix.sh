#!/bin/sh

# Plan or execute the reviewer-rerun LoRA training matrix.
# Planning is the default. --execute may download pinned model snapshots and
# start long-running MLX-LM jobs, but it never invokes the OpenAI API.

set -eu

ACTION="plan"
STAGE="full"
INCLUDE_QWEN_17="0"
MINIMUM_FREE_GIB="40"

usage() {
    printf '%s\n' \
        "Usage: ./scripts/run_training_matrix.sh [--full|--smoke]" \
        "       [--mandatory|--include-qwen-1.7b] [--plan|--execute]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --plan) ACTION="plan" ;;
        --execute) ACTION="execute" ;;
        --full) STAGE="full" ;;
        --smoke) STAGE="smoke" ;;
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
    printf '%s\n' "ERROR: training matrix requires branch reviewer-rerun." >&2
    exit 1
fi

if [ "$ACTION" = "execute" ]; then
    for split in \
        data/reviewer_rerun/socratic/train.jsonl \
        data/reviewer_rerun/socratic/valid.jsonl \
        data/reviewer_rerun/non_socratic/train.jsonl \
        data/reviewer_rerun/non_socratic/valid.jsonl
    do
        if [ ! -f "$split" ]; then
            printf '%s\n' "ERROR: matched training split is missing: $split" >&2
            exit 1
        fi
    done
fi

run_status() {
    "$PYTHON" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))' \
        "$1"
}

run_one() {
    config_path=$1
    experiment_id=$2
    condition_path=$3

    if [ "$STAGE" = "smoke" ]; then
        run_dir="runs/reviewer_rerun/smoke/training/$condition_path"
        set -- \
            "$PYTHON" -m src.run_training run \
            --config "$config_path" \
            --experiment-id "$experiment_id-smoke" \
            --run-dir "$run_dir" \
            --smoke-iters 20 \
            --minimum-free-gib "$MINIMUM_FREE_GIB"
    else
        run_dir="runs/reviewer_rerun/training/$condition_path"
        set -- \
            "$PYTHON" -m src.run_training run \
            --config "$config_path" \
            --experiment-id "$experiment_id" \
            --run-dir "$run_dir" \
            --minimum-free-gib "$MINIMUM_FREE_GIB"
    fi

    if [ "$ACTION" = "plan" ]; then
        printf '%s' "PLAN:"
        for argument in "$@"; do
            printf ' %s' "$argument"
        done
        printf '\n'
        return
    fi

    manifest_path="$run_dir/manifest.json"
    if [ -f "$manifest_path" ]; then
        status=$(run_status "$manifest_path")
        if [ "$status" = "completed" ]; then
            printf '%s\n' "SKIP: completed training run $experiment_id ($run_dir)"
            return
        fi
        printf '%s\n' \
            "ERROR: incomplete training directory requires archival and review: $run_dir" >&2
        printf '%s\n' \
            "Do not adapter-resume an official run; optimizer/scheduler state is not restored." >&2
        exit 1
    fi

    "$@"
}

run_one \
    configs/reviewer_rerun/qwen3_0.6b_socratic.yaml \
    qwen3-0.6b-socratic \
    qwen3_0.6b_socratic
run_one \
    configs/reviewer_rerun/qwen3_0.6b_non_socratic.yaml \
    qwen3-0.6b-non-socratic \
    qwen3_0.6b_non_socratic
run_one \
    configs/reviewer_rerun/llama3.2_1b_socratic.yaml \
    llama3.2-1b-socratic \
    llama3.2_1b_socratic

if [ "$INCLUDE_QWEN_17" = "1" ]; then
    run_one \
        configs/reviewer_rerun/qwen3_1.7b_socratic.yaml \
        qwen3-1.7b-socratic \
        qwen3_1.7b_socratic
    run_one \
        configs/reviewer_rerun/qwen3_1.7b_non_socratic.yaml \
        qwen3-1.7b-non-socratic \
        qwen3_1.7b_non_socratic
fi

printf '%s\n' "Training matrix $ACTION completed: stage=$STAGE include_qwen_1.7b=$INCLUDE_QWEN_17"

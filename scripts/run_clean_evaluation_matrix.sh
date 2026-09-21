#!/bin/sh

# Run the post-fix evaluation matrix in an isolated namespace. This wrapper
# deliberately reuses the final trained adapters for timing smokes and never
# points at the historical evaluation directories.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export SOCRATIQ_EVALUATION_OUTPUT_ROOT="runs/reviewer_rerun/evaluation_clean_v1"
export SOCRATIQ_EVALUATION_SMOKE_OUTPUT_ROOT="runs/reviewer_rerun/evaluation_clean_v1_smoke"
export SOCRATIQ_SMOKE_USE_FULL_ADAPTERS="1"

exec "$SCRIPT_DIR/run_evaluation_matrix.sh" "$@"

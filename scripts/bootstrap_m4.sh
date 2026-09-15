#!/bin/sh

# Bootstrap the locked reviewer-rerun environment on the experiment M4 Mac.
# This script performs no OpenAI requests, model downloads, or training.

set -eu

EXPECTED_UV_VERSION="0.9.18"
EXPECTED_PYTHON_VERSION="3.13.2"
EXPECTED_BRANCH="reviewer-rerun"
CHECK_ONLY=0

usage() {
    printf '%s\n' "Usage: ./scripts/bootstrap_m4.sh [--check-only]"
}

if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    if [ "$1" != "--check-only" ]; then
        usage >&2
        exit 2
    fi
    CHECK_ONLY=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

if [ "${SOCRATIQ_BOOTSTRAP_ALLOW_NON_M4:-0}" != "1" ]; then
    if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
        printf '%s\n' "ERROR: reviewer training requires an Apple Silicon Mac." >&2
        exit 1
    fi
    CHIP=$(/usr/sbin/system_profiler SPHardwareDataType | awk -F': ' '/Chip:/{print $2; exit}')
    case "$CHIP" in
        "Apple M4"*) ;;
        *)
            printf '%s\n' "ERROR: expected an Apple M4-family chip, found: ${CHIP:-unknown}." >&2
            exit 1
            ;;
    esac
else
    CHIP="platform check bypassed for bootstrap testing"
fi

BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "$EXPECTED_BRANCH" ]; then
    printf '%s\n' "ERROR: expected branch $EXPECTED_BRANCH, found ${BRANCH:-detached HEAD}." >&2
    exit 1
fi
if [ "${SOCRATIQ_BOOTSTRAP_ALLOW_DIRTY:-0}" != "1" ] && [ -n "$(git status --porcelain)" ]; then
    printf '%s\n' "ERROR: start the M4 bootstrap from a clean Git working tree." >&2
    git status --short >&2
    exit 1
fi

UV_BIN=${SOCRATIQ_UV_BIN:-uv}
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    printf '%s\n' "ERROR: uv $EXPECTED_UV_VERSION is not on PATH." >&2
    printf '%s\n' "Install the pinned version using the command in docs/reviewer_rerun_workflow.md." >&2
    exit 1
fi
UV_VERSION=$("$UV_BIN" --version)
case "$UV_VERSION" in
    "uv $EXPECTED_UV_VERSION"*) ;;
    *)
        printf '%s\n' "ERROR: expected uv $EXPECTED_UV_VERSION, found: $UV_VERSION." >&2
        exit 1
        ;;
esac

printf '%s\n' "Repository: $REPOSITORY_ROOT"
printf '%s\n' "Branch: $BRANCH"
printf '%s\n' "Commit: $(git rev-parse HEAD)"
printf '%s\n' "Chip: $CHIP"
printf '%s\n' "uv: $UV_VERSION"

if [ "$CHECK_ONLY" -eq 1 ]; then
    printf '%s\n' "Bootstrap prerequisites passed; no environment changes were made."
    exit 0
fi

"$UV_BIN" python install "$EXPECTED_PYTHON_VERSION"
"$UV_BIN" sync \
    --locked \
    --only-group rerun \
    --python "$EXPECTED_PYTHON_VERSION"

ACTUAL_PYTHON_VERSION=$(
    .venv/bin/python -c 'import platform; print(platform.python_version())'
)
if [ "$ACTUAL_PYTHON_VERSION" != "$EXPECTED_PYTHON_VERSION" ]; then
    printf '%s\n' \
        "ERROR: expected Python $EXPECTED_PYTHON_VERSION, found $ACTUAL_PYTHON_VERSION." >&2
    exit 1
fi

"$UV_BIN" lock --check
make rerun-check
make rerun-dry-run-training
.venv/bin/python -m src.run_training environment

printf '%s\n' "Fresh-M4 bootstrap passed."
printf '%s\n' "No API request, model download, training, or evaluation was started."

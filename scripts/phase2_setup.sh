#!/bin/sh

# Set up and verify one M4 Pro machine for the revision-v2 runs.
#
#   ./scripts/phase2_setup.sh --machine isik --before-freeze   Friday, on branch revision-v2
#   ./scripts/phase2_setup.sh --machine chee --before-freeze   Friday, on branch revision-v2
#   ./scripts/phase2_setup.sh --machine chee                   after the freeze, on tag protocol-v2-frozen
#
# Syncs the locked environment, rebuilds the v2 data and checks it against the
# tracked hashes, verifies the Llama checkpoint, runs the tests and config
# checks, and prints the machine's queue. It never trains or evaluates.

set -eu

EXPECTED_UV_VERSION="0.9.18"
EXPECTED_PYTHON_VERSION="3.13.2"
FREEZE_TAG="protocol-v2-frozen"
MACHINE=""
BEFORE_FREEZE=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --machine) MACHINE="${2:-}"; shift ;;
        --before-freeze) BEFORE_FREEZE=1 ;;
        *) printf '%s\n' "Usage: ./scripts/phase2_setup.sh --machine {chee|isik} [--before-freeze]" >&2; exit 2 ;;
    esac
    shift
done
case "$MACHINE" in
    chee|isik) ;;
    *) printf '%s\n' "ERROR: --machine must be chee or isik." >&2; exit 2 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

CHIP=$(/usr/sbin/system_profiler SPHardwareDataType | awk -F': ' '/Chip:/{print $2; exit}')
case "$CHIP" in
    "Apple M4"*) ;;
    *) printf '%s\n' "ERROR: expected an Apple M4-family chip, found: ${CHIP:-unknown}." >&2; exit 1 ;;
esac

# macOS rewrites the tracked .DS_Store files on its own; they carry no code.
MODIFIED=$(git status --porcelain --untracked-files=no | grep -v '\.DS_Store$' || true)
if [ -n "$MODIFIED" ]; then
    printf '%s\n' "ERROR: tracked files are modified. Do not edit code or configs; ask the PI." "$MODIFIED" >&2
    exit 1
fi
HEAD_COMMIT=$(git rev-parse HEAD)
if [ "$BEFORE_FREEZE" -eq 1 ]; then
    if [ "$(git branch --show-current)" != "revision-v2" ]; then
        printf '%s\n' "ERROR: before the freeze, run 'git checkout revision-v2' first." >&2
        exit 1
    fi
else
    TAGGED=$(git rev-parse "$FREEZE_TAG^{commit}" 2>/dev/null || true)
    if [ "$TAGGED" != "$HEAD_COMMIT" ]; then
        printf '%s\n' "ERROR: HEAD is not the frozen tag. Run 'git fetch origin --tags' and 'git checkout $FREEZE_TAG'." >&2
        exit 1
    fi
fi

UV_BIN=${SOCRATIQ_UV_BIN:-uv}
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    printf '%s\n' "ERROR: uv $EXPECTED_UV_VERSION is not on PATH." >&2
    exit 1
fi
case "$("$UV_BIN" --version)" in
    "uv $EXPECTED_UV_VERSION"*) ;;
    *) printf '%s\n' "ERROR: expected uv $EXPECTED_UV_VERSION, found: $("$UV_BIN" --version)." >&2; exit 1 ;;
esac

printf '%s\n' "Machine: $MACHINE" "Chip: $CHIP" "Commit: $HEAD_COMMIT"

"$UV_BIN" python install "$EXPECTED_PYTHON_VERSION"
"$UV_BIN" sync --locked --only-group rerun --python "$EXPECTED_PYTHON_VERSION"
"$UV_BIN" lock --check

PYTHON=".venv/bin/python"
printf '%s\n' "--- Rebuilding the v2 data and checking it against configs/revision_v2/dataset_manifest.json"
"$PYTHON" -m src.revision_v2.data build
"$PYTHON" -m src.revision_v2.data verify

printf '%s\n' "--- Verifying the Llama checkpoint"
"$PYTHON" -m src.revision_v2.llama_base verify models/revision_v2/llama3.2-1b-instruct-meta-bf16

printf '%s\n' "--- Tests, lint and configuration checks"
make phase2-check RERUN_PYTHON="$PYTHON" RERUN_RUFF=.venv/bin/ruff

printf '%s\n' "--- Queue for $MACHINE (peak shown as 8e-5; use the PI-approved value when running)"
SOCRATIQ_MACHINE="$MACHINE" "$PYTHON" -m src.revision_v2.queue plan --machine "$MACHINE" --peak 8e-5

printf '%s\n' "Setup passed on $MACHINE. No training or evaluation was started."

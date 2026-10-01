#!/bin/sh

# Revision-v2 queue for the machine "isik". Resumable: after any stop or
# restart, run the same command again; finished items are skipped.
#
#   ./scripts/phase2_queue_isik.sh run       run everything; the pilot comes first and
#                                            chooses the learning rate (8e-5, else 4e-5)
#   ./scripts/phase2_queue_isik.sh report    progress and checks, to paste into an email
#   ./scripts/phase2_queue_isik.sh plan      list items, status and expected time
#   ./scripts/phase2_queue_isik.sh package   build the return packet
#
# On any failure the queue stops: do not edit code or configs; send the log.

set -eu

MACHINE="isik"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    printf '%s\n' "ERROR: .venv is missing; follow docs/phase2_todo_isik.md." >&2
    exit 1
fi
export SOCRATIQ_MACHINE="$MACHINE"
export HF_HUB_DISABLE_TELEMETRY=1

COMMAND="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi
case "$COMMAND" in
    run)
        exec caffeinate -is "$PYTHON" -m src.revision_v2.queue run --machine "$MACHINE" "$@"
        ;;
    plan|status|report|package)
        exec "$PYTHON" -m src.revision_v2.queue "$COMMAND" --machine "$MACHINE" "$@"
        ;;
    *)
        printf '%s\n' "Usage: ./scripts/phase2_queue_isik.sh {run|report|plan|package}" >&2
        exit 2
        ;;
esac

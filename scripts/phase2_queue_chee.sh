#!/bin/sh

# Revision-v2 queue for the machine "chee". Resumable: after any stop or
# restart, run the same command again; finished items are skipped.
#
#   ./scripts/phase2_queue_chee.sh run                 run the core items (no peak needed)
#   ./scripts/phase2_queue_chee.sh run --peak 8e-5     also run the second tier; use the
#                                                      peak from isik's pilot decision
#   ./scripts/phase2_queue_chee.sh report [--peak ..]  progress and checks, to paste into an email
#   ./scripts/phase2_queue_chee.sh plan [--peak ..]    list items, status and expected time
#   ./scripts/phase2_queue_chee.sh package [--peak ..] build the return packet
#
# The Qwen3-1.7B pair always uses 1e-4; the peak only affects the second tier.
# On any failure the queue stops: do not edit code or configs; send the log.

set -eu

MACHINE="chee"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    printf '%s\n' "ERROR: .venv is missing; follow docs/phase2_todo_chee.md." >&2
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
        printf '%s\n' "Usage: ./scripts/phase2_queue_chee.sh {run|report|plan|package} [--peak 8e-5|4e-5]" >&2
        exit 2
        ;;
esac

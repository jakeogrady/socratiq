#!/bin/sh

# Revision-v2 queue for the machine "isik". Resumable: after any stop or
# restart, run the same command again; finished items are skipped.
#
#   ./scripts/phase2_queue_isik.sh pilot --peak 8e-5     Friday, before the freeze
#   ./scripts/phase2_queue_isik.sh plan --peak 8e-5      list items, status, expected time
#   ./scripts/phase2_queue_isik.sh run --peak 8e-5       after the freeze tag, whole list
#   ./scripts/phase2_queue_isik.sh report --peak 8e-5    progress and checks, to paste into an email
#   ./scripts/phase2_queue_isik.sh package --peak 8e-5   build the return packet
#
# Use the peak the PI approved (8e-5, or 4e-5 if the first pilot failed).
# On any failure the queue stops: do not edit code or configs; send the log.

set -eu

MACHINE="isik"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPOSITORY_ROOT"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    printf '%s\n' "ERROR: .venv is missing; follow setup step 2 of docs/phase2_runbook.md." >&2
    exit 1
fi
export SOCRATIQ_MACHINE="$MACHINE"
export HF_HUB_DISABLE_TELEMETRY=1

COMMAND="${1:-}"
if [ "$#" -gt 0 ]; then
    shift
fi
case "$COMMAND" in
    pilot)
        exec caffeinate -is "$PYTHON" -m src.revision_v2.queue pilot "$@"
        ;;
    run)
        exec caffeinate -is "$PYTHON" -m src.revision_v2.queue run --machine "$MACHINE" "$@"
        ;;
    plan|status|package|report)
        exec "$PYTHON" -m src.revision_v2.queue "$COMMAND" --machine "$MACHINE" "$@"
        ;;
    *)
        printf '%s\n' "Usage: ./scripts/phase2_queue_isik.sh {pilot|plan|run|report|package} --peak {8e-5|4e-5}" >&2
        exit 2
        ;;
esac

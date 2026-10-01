#!/bin/sh

# Revision-v2 queue for the machine "chee". Resumable: after any stop or
# restart, run the same command again; finished items are skipped.
#
#   ./scripts/phase2_queue_chee.sh plan --peak 8e-5      list items, status, expected time
#   ./scripts/phase2_queue_chee.sh run --peak 8e-5       after the freeze tag, whole list
#   ./scripts/phase2_queue_chee.sh report --peak 8e-5    progress and checks, to paste into an email
#   ./scripts/phase2_queue_chee.sh package --peak 8e-5   build the return packet
#
# Use the peak the PI approved after the pilot on isik (8e-5 or 4e-5). It
# applies to the second-tier Qwen3-0.6B runs; the Qwen3-1.7B pair always uses 1e-4.
# On any failure the queue stops: do not edit code or configs; send the log.

set -eu

MACHINE="chee"
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
    run)
        exec caffeinate -is "$PYTHON" -m src.revision_v2.queue run --machine "$MACHINE" "$@"
        ;;
    plan|status|package|report)
        exec "$PYTHON" -m src.revision_v2.queue "$COMMAND" --machine "$MACHINE" "$@"
        ;;
    *)
        printf '%s\n' "Usage: ./scripts/phase2_queue_chee.sh {plan|run|report|package} --peak {8e-5|4e-5}" >&2
        exit 2
        ;;
esac

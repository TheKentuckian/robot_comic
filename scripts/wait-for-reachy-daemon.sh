#!/bin/bash
# Poll the Reachy daemon's motors API until it responds or we time out.
# Replaces the legacy ExecStartPre=/bin/sleep 30.
#
# Exit codes:
#   0 — daemon is ready (motors endpoint responded)
#   1 — timed out after 10s without a response
#
# Using exit 1 on timeout (rather than 0) lets systemd treat this as a
# failed precondition and retry/fail the unit instead of blindly proceeding
# with motors still disabled.  If you prefer the old "proceed anyway"
# behaviour, change the final `exit 1` to `exit 0`.
set -u

MAX_TRIES=50       # 50 x 0.2 s = 10 s max
INTERVAL=0.2
ENDPOINT="${REACHY_DAEMON_URL:-http://127.0.0.1:8000}/api/motors/get_mode"

for i in $(seq 1 "${MAX_TRIES}"); do
    if curl -fsS "${ENDPOINT}" >/dev/null 2>&1; then
        elapsed=$(echo "$i * $INTERVAL" | bc 2>/dev/null || echo "$((i * 200))ms")
        echo "Reachy daemon ready after ${elapsed}s" >&2
        exit 0
    fi
    sleep "${INTERVAL}"
done

echo "Reachy daemon not ready after $((MAX_TRIES * 200 / 1000))s — giving up" >&2
exit 1

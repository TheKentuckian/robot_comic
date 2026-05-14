#!/usr/bin/env bash
# install-pi-journald.sh — Install the robot_comic journald drop-in on a Pi.
#
# Usage (run from the repo root):
#   sudo ./scripts/install-pi-journald.sh
#
# What it does:
#   1. Creates /etc/systemd/journald.conf.d/ if it does not exist.
#   2. Creates /var/log/journal/ so journald uses persistent on-disk storage
#      instead of the RAM-backed /run/log/journal (which is wiped on reboot).
#   3. Copies deploy/systemd/journald-robot-comic.conf to
#      /etc/systemd/journald.conf.d/robot-comic.conf.
#   4. Restarts systemd-journald to apply the new settings immediately.
#   5. Prints current disk usage so you can confirm the cap is in effect.

set -euo pipefail

# ---------------------------------------------------------------------------
# Privilege check
# ---------------------------------------------------------------------------
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (use sudo)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate the drop-in config relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DROP_IN_SRC="${REPO_ROOT}/deploy/systemd/journald-robot-comic.conf"

if [ ! -f "${DROP_IN_SRC}" ]; then
    echo "ERROR: Drop-in config not found at ${DROP_IN_SRC}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
CONF_DIR="/etc/systemd/journald.conf.d"
DEST="${CONF_DIR}/robot-comic.conf"
JOURNAL_DIR="/var/log/journal"

echo "==> Creating ${CONF_DIR}..."
mkdir -p "${CONF_DIR}"

echo "==> Creating ${JOURNAL_DIR} for persistent (non-RAM) journal storage..."
mkdir -p "${JOURNAL_DIR}"
# Ensure journald can write here
chown root:systemd-journal "${JOURNAL_DIR}" 2>/dev/null || true
chmod 2755 "${JOURNAL_DIR}" 2>/dev/null || true

echo "==> Installing drop-in config to ${DEST}..."
install -m 644 "${DROP_IN_SRC}" "${DEST}"

echo "==> Restarting systemd-journald..."
systemctl restart systemd-journald

echo "==> Current journal disk usage:"
journalctl --disk-usage

echo ""
echo "Done. Journal is now capped at 500 MB / 2-week retention (${DEST})."

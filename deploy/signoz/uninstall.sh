#!/usr/bin/env bash
# uninstall.sh — Stop the SigNoz stack and remove all data volumes.
#
# Run this on the Pi when you're done investigating and want to
# reclaim RAM and disk space:
#   bash deploy/signoz/uninstall.sh
#
# WARNING: This deletes all stored traces, metrics, and logs.
#          Use  docker compose down  (without -v) to keep data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Stopping SigNoz stack and removing volumes..."
cd "$SCRIPT_DIR"
docker compose down -v --remove-orphans

echo ""
echo "==> SigNoz removed.  Named volumes deleted:"
echo "    signoz-clickhouse"
echo "    signoz-clickhouse-user-scripts"
echo "    signoz-sqlite"
echo "    signoz-zookeeper-1"
echo ""
echo "==> To reinstall, run:  bash ${SCRIPT_DIR}/install.sh"

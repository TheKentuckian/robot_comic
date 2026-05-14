#!/usr/bin/env bash
# install.sh — Start the SigNoz observability stack on the Pi 5.
#
# Run this on the Pi:
#   bash deploy/signoz/install.sh
#
# The stack starts on-demand; it does NOT auto-start at boot.
# Use uninstall.sh (docker compose down -v) to stop and wipe data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "==> Checking Docker..."
if ! docker --version &>/dev/null; then
    echo "ERROR: Docker is not installed or not in PATH."
    echo "Install Docker on the Pi with:"
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  sudo usermod -aG docker \$USER   # then log out and back in"
    exit 1
fi

echo "==> Checking Docker Compose (plugin)..."
if ! docker compose version &>/dev/null; then
    echo "ERROR: 'docker compose' (v2 plugin) is not available."
    echo "Install it with:  sudo apt-get install docker-compose-plugin"
    exit 1
fi

# ── Kernel parameter for ClickHouse ──────────────────────────────────────────
# ClickHouse recommends vm.max_map_count >= 262144.
# We set it transiently here; for persistence add to /etc/sysctl.d/.
current_map=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)
if [ "$current_map" -lt 262144 ]; then
    echo "==> Setting vm.max_map_count=262144 (transient)..."
    sudo sysctl -w vm.max_map_count=262144
fi

# ── Launch stack ─────────────────────────────────────────────────────────────

echo "==> Starting SigNoz stack (this may take 1-2 minutes on first run while images pull)..."
cd "$SCRIPT_DIR"
docker compose up -d --remove-orphans

echo ""
echo "==> Waiting for SigNoz to become healthy..."
attempts=0
max_attempts=30
until docker inspect --format='{{.State.Health.Status}}' signoz 2>/dev/null | grep -q "healthy"; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge "$max_attempts" ]; then
        echo "WARNING: SigNoz health check timed out after ${max_attempts}s."
        echo "Check logs with:  docker compose -f ${SCRIPT_DIR}/docker-compose.yml logs signoz"
        break
    fi
    printf "."
    sleep 2
done
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────

HOSTNAME_="$(hostname).local"
echo "============================================================"
echo "  SigNoz is up!"
echo ""
echo "  UI:          http://${HOSTNAME_}:3301"
echo "  OTLP gRPC:   ${HOSTNAME_}:4317   (set OTEL_EXPORTER_OTLP_ENDPOINT)"
echo "  OTLP HTTP:   ${HOSTNAME_}:4318"
echo ""
echo "  To configure robot_comic, add to your .env:"
echo "    ROBOT_INSTRUMENTATION=remote"
echo "    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317"
echo ""
echo "  To stop and preserve data:"
echo "    docker compose -f ${SCRIPT_DIR}/docker-compose.yml down"
echo ""
echo "  To stop and WIPE all trace data:"
echo "    bash ${SCRIPT_DIR}/uninstall.sh"
echo "============================================================"

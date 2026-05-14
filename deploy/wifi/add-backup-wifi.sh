#!/bin/bash
# Add a backup WiFi network at lower priority than the primary connection.
# Run on the Pi as the user that owns the active session.
#
# Usage:
#   sudo ./deploy/wifi/add-backup-wifi.sh <SSID> <PASSWORD> [priority]
#
# Priority defaults to 5 (lower than the primary's typical 10-100).
# NetworkManager will fall back to this network when the primary is unavailable.

set -eu

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <SSID> <PASSWORD> [priority]" >&2
  echo "Example: $0 SetecMobile8a 'super-secret' 5" >&2
  exit 1
fi

SSID="$1"
PASSWORD="$2"
PRIORITY="${3:-5}"
CONNECTION_NAME="${SSID}-backup"

# Idempotent: remove an existing connection of the same name before adding.
if nmcli connection show "$CONNECTION_NAME" >/dev/null 2>&1; then
  echo "Removing existing connection '$CONNECTION_NAME'..."
  nmcli connection delete "$CONNECTION_NAME"
fi

echo "Adding WiFi connection '$CONNECTION_NAME' (priority=$PRIORITY)..."
nmcli connection add \
  type wifi \
  con-name "$CONNECTION_NAME" \
  ifname wlan0 \
  ssid "$SSID" \
  -- \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$PASSWORD" \
  connection.autoconnect yes \
  connection.autoconnect-priority "$PRIORITY"

echo ""
echo "Done. Verify with:"
echo "  nmcli connection show"
echo "  nmcli -f NAME,AUTOCONNECT-PRIORITY connection"

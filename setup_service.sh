#!/bin/bash

# SVDS Detection Service Setup Script (with log rotation)
# --------------------------------------------------------
# This script sets up:
#   - svds-detection.service (your detection pipeline)
#   - svds-reboot.service (24h auto reboot)
#   - /var/log/svds-detection.log with logrotate policy

set -euo pipefail

SERVICE_NAME="svds-detection"
REBOOT_SERVICE_NAME="svds-reboot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_SCRIPT="$SCRIPT_DIR/run_detection.sh"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REBOOT_SERVICE_FILE="/etc/systemd/system/${REBOOT_SERVICE_NAME}.service"
REBOOT_SCRIPT="$SCRIPT_DIR/reboot_system.sh"
LOG_FILE="/var/log/svds-detection.log"
LOGROTATE_CONF="/etc/logrotate.d/svds-detection"

# Run as root
if [[ $EUID -ne 0 ]]; then
    echo "⚠️  This script needs root privileges. Re-run with sudo..."
    exec sudo bash "$0" "$@"
fi

# Detect user
if [[ -n "${SUDO_USER:-}" ]]; then
    RUN_USER="$SUDO_USER"
else
    RUN_USER=$(whoami)
fi

echo "👉 Setting up services for user: $RUN_USER in $SCRIPT_DIR"

# Ensure detection script exists
if [[ ! -f "$DETECTION_SCRIPT" ]]; then
    echo "❌ Error: run_detection.sh not found in $SCRIPT_DIR"
    exit 1
fi
chmod +x "$DETECTION_SCRIPT"

# --- Create log file ---
touch "$LOG_FILE"
chown "$RUN_USER":"$RUN_USER" "$LOG_FILE"
chmod 664 "$LOG_FILE"

# --- Reboot script ---
cat > "$REBOOT_SCRIPT" << 'EOF'
#!/bin/bash
sleep 86400   # 24 hours
systemctl stop svds-detection
sleep 5
reboot
EOF
chmod +x "$REBOOT_SCRIPT"

# --- Detection service ---
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=SVDS Detection Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$DETECTION_SCRIPT
Restart=always
RestartSec=30
RuntimeMaxSec=21600
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF

# --- Reboot service ---
cat > "$REBOOT_SERVICE_FILE" <<EOF
[Unit]
Description=SVDS 24-Hour Reboot Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
ExecStart=$REBOOT_SCRIPT
Restart=always
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# --- Logrotate config ---
cat > "$LOGROTATE_CONF" <<EOF
$LOG_FILE {
    size 50M
    rotate 5
    compress
    missingok
    notifempty
    copytruncate
}
EOF

# --- Enable and start services ---
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" "$REBOOT_SERVICE_NAME"
systemctl restart "$SERVICE_NAME" "$REBOOT_SERVICE_NAME"

echo "✅ Setup complete!"
echo "  - Service: $SERVICE_NAME (logs in $LOG_FILE)"
echo "  - Service: $REBOOT_SERVICE_NAME (auto reboot every 24h)"
echo "  - Logrotate rule: $LOGROTATE_CONF"

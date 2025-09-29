#!/bin/bash
set -e

### ==============================
### Variables (edit as needed)
### ==============================
ETH_IP="Placeholder"
ETH_GW="gatewayIP"
TABLE_ID="200"
TABLE_NAME="eth0table"

WIFI_METRIC=100
ETH_METRIC=600

# External devices/IPs you want to force through eth0
FORCED_IPS=("ServerIP")

# Paths
SYSTEMD_SERVICE="/etc/systemd/system/eth0-policy-routing.service"
ROUTING_SCRIPT="/usr/local/bin/eth0-policy-routing.sh"

### ==============================
### Detect connection names
### ==============================
ETH_CONN=$(nmcli -t -f NAME,DEVICE connection show --active | grep ":eth0" | cut -d: -f1)
WIFI_CONN=$(nmcli -t -f NAME,DEVICE connection show --active | grep ":wlan0" | cut -d: -f1)

if [ -z "$ETH_CONN" ]; then
    echo "No active eth0 connection found. Exiting."
    exit 1
fi

echo "Detected Ethernet connection: $ETH_CONN"
[ -n "$WIFI_CONN" ] && echo "Detected Wi-Fi connection: $WIFI_CONN"

### ==============================
### Configure Ethernet and Wi-Fi
### ==============================
sudo nmcli connection modify "$ETH_CONN" ipv4.addresses "$ETH_IP"
sudo nmcli connection modify "$ETH_CONN" ipv4.method manual
sudo nmcli connection modify "$ETH_CONN" ipv4.route-metric "$ETH_METRIC"
sudo nmcli connection up "$ETH_CONN"

if [ -n "$WIFI_CONN" ]; then
    sudo nmcli connection modify "$WIFI_CONN" ipv4.route-metric "$WIFI_METRIC"
    sudo nmcli connection up "$WIFI_CONN"
fi

### ==============================
### Create routing helper script
### ==============================
sudo tee "$ROUTING_SCRIPT" > /dev/null << EOF
#!/bin/bash
TABLE_NAME="$TABLE_NAME"
TABLE_ID="$TABLE_ID"
ETH_IP="$ETH_IP"
ETH_GW="$ETH_GW"
FORCED_IPS=(${FORCED_IPS[@]})

# Add custom routing table if not exists
grep -q "\$TABLE_NAME" /etc/iproute2/rt_tables || echo "\$TABLE_ID \$TABLE_NAME" | sudo tee -a /etc/iproute2/rt_tables

# ==============================
# Clean up old rules and routes in this table (except current ETH_IP)
# ==============================
OLD_RULES=\$(ip rule | grep "lookup \$TABLE_NAME" | awk '{print \$1,\$2,\$3}')
for RULE in \$OLD_RULES; do
    FROM=\$(echo \$RULE | awk '{print \$3}')
    if [ "\$FROM" != "\${ETH_IP%/*}" ]; then
        sudo ip rule del from \$FROM table \$TABLE_NAME 2>/dev/null || true
    fi
done

OLD_ROUTES=\$(ip route show table \$TABLE_NAME | grep -v "\$ETH_IP")
for R in \$OLD_ROUTES; do
    sudo ip route del \$R table \$TABLE_NAME 2>/dev/null || true
done

# ==============================
# Default route for eth0 subnet
# ==============================
sudo ip route replace \$ETH_IP dev eth0 table \$TABLE_NAME
sudo ip route replace default via \$ETH_GW dev eth0 table \$TABLE_NAME

# Force all traffic from eth0 IP to use eth0table
sudo ip rule del from \${ETH_IP%/*} table \$TABLE_NAME 2>/dev/null || true
sudo ip rule add from \${ETH_IP%/*} table \$TABLE_NAME

# Force all replies to eth0 IP to go through eth0table
sudo ip rule del to \${ETH_IP%/*} table \$TABLE_NAME 2>/dev/null || true
sudo ip rule add to \${ETH_IP%/*} table \$TABLE_NAME

# Apply forced routing for each external IP
for IP in "\${FORCED_IPS[@]}"; do
    sudo ip route replace \$IP dev eth0 table \$TABLE_NAME
    sudo ip rule del to \$IP table \$TABLE_NAME 2>/dev/null || true
    sudo ip rule add to \$IP table \$TABLE_NAME
done
EOF

sudo chmod +x "$ROUTING_SCRIPT"

### ==============================
### Create systemd service
### ==============================
sudo tee "$SYSTEMD_SERVICE" > /dev/null << EOF
[Unit]
Description=Policy routing for eth0 (forced IPs)
After=network-online.target
Wants=network-online.target
Restart=on-failure
RestartSec=10

[Service]
Type=oneshot
ExecStart=$ROUTING_SCRIPT
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

### ==============================
### Enable and start systemd service
### ==============================
sudo systemctl daemon-reload
sudo systemctl enable eth0-policy-routing.service
sudo systemctl restart eth0-policy-routing.service

### ==============================
### Verification
### ==============================
echo "==== ip rule ===="
ip rule
echo "==== ip route show table $TABLE_NAME ===="
ip route show table $TABLE_NAME
echo "==== Test route get for forced IPs ===="
for IP in "${FORCED_IPS[@]}"; do
    ip route get $IP
done

echo "✅ Setup complete: eth0 subnet and forced IPs routed via eth0, internet unaffected."

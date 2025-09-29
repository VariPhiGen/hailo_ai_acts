#!/bin/bash
set -e

### ==============================
### Config (edit as needed)
### ==============================
IFACE="eth0"
STATIC_IP="Place your static ip here"
GATEWAY="Place your gateway here"
DNS="8.8.8.8 1.1.1.1"
NETPLAN_FILE="/etc/netplan/99-$IFACE.yaml"

echo "👉 Flushing old IPs and routes on $IFACE..."
sudo ip addr flush dev "$IFACE" || true
sudo ip route flush dev "$IFACE" || true

echo "👉 Writing Netplan config: $NETPLAN_FILE"
sudo tee "$NETPLAN_FILE" > /dev/null <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    $IFACE:
      dhcp4: no
      addresses:
        - $STATIC_IP
      gateway4: $GATEWAY
      nameservers:
        addresses: [$DNS]
EOF

echo "👉 Applying Netplan..."
sudo netplan apply

echo "✅ Static IP and gateway set on $IFACE"
ip addr show dev "$IFACE"
ip route show

#!/bin/bash
# ==============================================
# Raspberry Pi Nginx Media Share Setup Script
# Auto-detects user and sets up /media share
# Provides clear exit reasons if something fails
# Author: Nitish Mishra
# ==============================================

# Exit immediately if a command fails
set -e

# Custom error handling
trap 'echo "❌ ERROR: The script exited unexpectedly. Check the above logs for details."; exit 1' ERR

# Detect current user
CURRENT_USER=$(whoami)
MEDIA_DIR="/home/$CURRENT_USER/media"

echo "🚀 Starting Nginx setup for user: $CURRENT_USER"
echo "📁 Target shared folder: $MEDIA_DIR"
sleep 1

# 1️⃣ Update and install nginx
echo "🔄 Updating packages and installing Nginx..."
if ! sudo apt update -y && sudo apt install nginx -y; then
    echo "❌ Failed to install Nginx. Check your internet or package sources."
    exit 1
fi

# 2️⃣ Create shared folder
if [ ! -d "$MEDIA_DIR" ]; then
    echo "📂 Creating shared folder..."
    if ! mkdir -p "$MEDIA_DIR"; then
        echo "❌ Failed to create folder: $MEDIA_DIR"
        exit 1
    fi
else
    echo "ℹ️ Folder already exists: $MEDIA_DIR"
fi

# 3️⃣ Set permissions
echo "🔧 Setting permissions..."
if ! sudo chmod -R 755 "$MEDIA_DIR"; then
    echo "❌ Failed to set permissions on $MEDIA_DIR"
    exit 1
fi

if ! sudo chown -R www-data:www-data "$MEDIA_DIR"; then
    echo "❌ Failed to change ownership to www-data"
    exit 1
fi

# 4️⃣ Link folder to Nginx web root
echo "🔗 Linking shared folder to /var/www/html/media..."
if ! sudo rm -rf /var/www/html/media 2>/dev/null || true; then
    echo "⚠️ Could not remove old /media link, continuing..."
fi

if ! sudo ln -sf "$MEDIA_DIR" /var/www/html/media; then
    echo "❌ Failed to link $MEDIA_DIR to /var/www/html/media"
    exit 1
fi

# 5️⃣ Enable and restart nginx
echo "⚙️ Enabling and restarting Nginx..."
if ! sudo systemctl enable nginx; then
    echo "❌ Failed to enable Nginx service."
    exit 1
fi

if ! sudo systemctl restart nginx; then
    echo "❌ Failed to restart Nginx service."
    exit 1
fi

# 6️⃣ Get IP address
IP_ADDR=$(hostname -I | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    echo "❌ Unable to detect IP address. Ensure you're connected to the network."
    exit 1
fi

# 7️⃣ Success message
echo
echo "✅ Setup complete!"
echo "🌐 Access your shared media here:"
echo "👉 http://$IP_ADDR/media/"
echo
echo "📁 Save images/audio to: $MEDIA_DIR"
echo "   Example: cp image1.jpg $MEDIA_DIR/"
echo "   Example: cp sound1.wav $MEDIA_DIR/"
echo
echo "🧠 New files appear instantly — no restart required!"

#!/bin/bash
# ==============================================
# Raspberry Pi Nginx Media Share Setup Script
# Auto-detects user and sets up /media share
# Ensures permissions so Nginx can serve all files
# Provides clear exit reasons if something fails
# Author: Nitish Mishra
# ==============================================

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
sudo apt update -y
sudo apt install nginx -y

# 2️⃣ Create shared folder
if [ ! -d "$MEDIA_DIR" ]; then
    echo "📂 Creating shared folder..."
    mkdir -p "$MEDIA_DIR"
else
    echo "ℹ️ Folder already exists: $MEDIA_DIR"
fi

# 3️⃣ Fix home directory permissions for Nginx traversal
echo "🔧 Setting /home/$CURRENT_USER permissions..."
sudo chmod 711 /home/$CURRENT_USER

# 4️⃣ Set media folder permissions
echo "🔧 Setting media folder and files permissions..."
sudo chown -R www-data:www-data "$MEDIA_DIR"
sudo chmod -R 755 "$MEDIA_DIR"

# 5️⃣ Ensure Nginx web root exists
if [ ! -d "/var/www/html" ]; then
    echo "📂 /var/www/html missing, creating..."
    sudo mkdir -p /var/www/html
    sudo chown www-data:www-data /var/www/html
fi

# 6️⃣ Link folder to Nginx web root
echo "🔗 Linking shared folder to /var/www/html/media..."
sudo rm -rf /var/www/html/media 2>/dev/null || true
sudo ln -sf "$MEDIA_DIR" /var/www/html/media

# 7️⃣ Create Nginx site config with autoindex
NGINX_CONF="/etc/nginx/sites-available/media_share"
sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name _;
    location /media/ {
        alias /var/www/html/media/;
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
    }
}
EOF

# Enable site and reload Nginx
sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 8️⃣ Enable and restart nginx
echo "⚙️ Enabling and restarting Nginx..."
sudo systemctl enable nginx
sudo systemctl restart nginx

# 9️⃣ Get IP address
IP_ADDR=$(hostname -I | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    echo "❌ Unable to detect IP address. Ensure you're connected to the network."
    exit 1
fi

# 10️⃣ Success message
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

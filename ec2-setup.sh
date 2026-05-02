#!/bin/bash
set -e

echo "=== Installing Docker ==="
apt-get update -qq
apt-get install -y docker.io docker-compose git
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

echo "=== Cloning repository ==="
cd /home/ubuntu
git clone https://github.com/edgarguardado6631-rgb/mri-alzheimer-analyzer.git app
cd app

echo "=== Writing .env ==="
cat > ml/.env << 'ENVEOF'
ANTHROPIC_API_KEY=PLACEHOLDER_KEY
ALLOWED_ORIGINS=https://d138vh09lsfjit.cloudfront.net
ENVEOF

echo "=== Building and starting backend ==="
docker build -t neuroscan-backend ./ml
docker run -d \
  --name neuroscan-backend \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file ml/.env \
  neuroscan-backend

echo "=== Done! Backend starting on port 8000 ==="
echo "Check status: docker logs neuroscan-backend"

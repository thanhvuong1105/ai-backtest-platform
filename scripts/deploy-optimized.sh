#!/bin/bash
# =============================================================================
# DEPLOY OPTIMIZED CONFIGURATION
# For 8 vCPU, 32GB RAM, 20+ Workers
# =============================================================================

set -e

echo "=========================================="
echo "DEPLOYING OPTIMIZED CONFIGURATION"
echo "Server: 8 vCPU, 32GB RAM, 20+ Workers"
echo "=========================================="

# Check if running on EC2
if [ ! -d "/home/ubuntu" ]; then
    echo "Warning: This script is designed for EC2 Ubuntu servers"
fi

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Step 1: Copy optimized environment file
echo ""
echo "[1/5] Setting up environment variables..."
if [ -f "$PROJECT_ROOT/backend/.env.optimized" ]; then
    cp "$PROJECT_ROOT/backend/.env.optimized" "$PROJECT_ROOT/backend/.env"
    echo "✓ Copied .env.optimized to .env"
else
    echo "✗ Error: .env.optimized not found"
    exit 1
fi

# Step 2: Stop existing containers
echo ""
echo "[2/5] Stopping existing containers..."
docker-compose down 2>/dev/null || true
docker-compose -f docker-compose.optimized.yml down 2>/dev/null || true
echo "✓ Containers stopped"

# Step 3: Pull latest images and build
echo ""
echo "[3/5] Building optimized containers..."
docker-compose -f docker-compose.optimized.yml build --no-cache
echo "✓ Containers built"

# Step 4: Start services
echo ""
echo "[4/5] Starting optimized services..."
docker-compose -f docker-compose.optimized.yml up -d
echo "✓ Services started"

# Step 5: Verify services
echo ""
echo "[5/5] Verifying services..."
sleep 10

# Check container status
echo ""
echo "Container Status:"
docker-compose -f docker-compose.optimized.yml ps

# Check health
echo ""
echo "Service Health:"
for service in redis backend celery frontend; do
    if docker ps --filter "name=quant-brain-$service" --format "{{.Status}}" | grep -q "Up"; then
        echo "✓ $service: Running"
    else
        echo "✗ $service: Not running"
    fi
done

# Display resource usage
echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Access points:"
echo "  Frontend:  http://$(curl -s ifconfig.me):3000"
echo "  Backend:   http://$(curl -s ifconfig.me):8000"
echo "  API Docs:  http://$(curl -s ifconfig.me):8000/docs"
echo ""
echo "Monitoring:"
echo "  docker-compose -f docker-compose.optimized.yml logs -f celery"
echo "  docker stats"
echo ""
echo "To enable Flower monitoring:"
echo "  docker-compose -f docker-compose.optimized.yml --profile monitoring up -d flower"
echo ""

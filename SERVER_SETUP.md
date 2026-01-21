# Hướng Dẫn Chạy AI Backtest Platform Trên Server (Không Docker)

## Yêu Cầu Hệ Thống

### Hardware khuyến nghị (i9-12900K, RTX 3080, 64GB RAM)
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA RTX 3080+ (optional)

### Software
- Python 3.11+
- Node.js 20+
- Redis Server 7+
- Git

---

## 1. Cài Đặt Dependencies

### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs -y

# Install Redis
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Install build tools
sudo apt install build-essential -y
```

### Windows
```powershell
# Install Chocolatey (nếu chưa có)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install python311 nodejs-lts redis-64 git -y

# Start Redis
redis-server --service-start
```

---

## 2. Clone Repository

```bash
git clone https://github.com/thanhvuong1105/ai-backtest-platform.git
cd ai-backtest-platform
```

---

## 3. Setup Backend

### Tạo Virtual Environment
```bash
cd backend
python3.11 -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Tạo File .env
```bash
cp .env.example .env
```

Chỉnh sửa `.env`:
```env
# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Performance (i9-12900K, 64GB RAM)
MAX_THREAD_WORKERS=16
CELERY_CONCURRENCY=8
PARALLEL_FITNESS=true
NUMBA_NUM_THREADS=8

# Cache
INDICATOR_CACHE_SIZE=4096
DATA_CACHE_SIZE=1024
DATA_PRELOAD=true

# Timeouts
JOB_TIMEOUT_SEC=3600
```

---

## 4. Setup Frontend

```bash
cd ../frontend
npm install
```

Tạo file `.env` (nếu cần):
```env
VITE_API_URL=http://localhost:8000/api
```

---

## 5. Chạy Services

### Option A: Chạy bằng 4 Terminal riêng biệt

#### Terminal 1: Redis (nếu chưa chạy service)
```bash
redis-server
```

#### Terminal 2: Backend (FastAPI)
```bash
cd ai-backtest-platform/backend
source venv/bin/activate  # Linux/macOS
# hoặc: venv\Scripts\activate  # Windows

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 3: Celery Worker
```bash
cd ai-backtest-platform/backend
source venv/bin/activate  # Linux/macOS
# hoặc: venv\Scripts\activate  # Windows

# Linux/macOS
celery -A app.services.celery_app.celery worker --loglevel=info -Q default --concurrency=8

# Windows (phải dùng --pool=solo)
celery -A app.services.celery_app.celery worker --loglevel=info -Q default --pool=solo
```

#### Terminal 4: Frontend
```bash
cd ai-backtest-platform/frontend

# Development mode
npm run dev

# Hoặc Production build
npm run build
npm run preview
```

---

### Option B: Chạy bằng Script (Linux/macOS)

Tạo file `start.sh`:
```bash
#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}Starting AI Backtest Platform...${NC}"

# Start Redis (if not running)
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Activate venv
source backend/venv/bin/activate

# Start Celery in background
echo "Starting Celery worker..."
cd backend
celery -A app.services.celery_app.celery worker --loglevel=info -Q default --concurrency=8 &
CELERY_PID=$!
cd ..

# Start Backend
echo "Starting Backend (FastAPI)..."
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}All services started!${NC}"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait and cleanup
trap "kill $CELERY_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
```

Chạy:
```bash
chmod +x start.sh
./start.sh
```

---

### Option C: Chạy với Systemd (Production - Linux)

#### 1. Tạo service cho Backend
```bash
sudo nano /etc/systemd/system/backtest-backend.service
```

```ini
[Unit]
Description=AI Backtest Backend
After=network.target redis.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/ai-backtest-platform/backend
Environment="PATH=/path/to/ai-backtest-platform/backend/venv/bin"
ExecStart=/path/to/ai-backtest-platform/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### 2. Tạo service cho Celery
```bash
sudo nano /etc/systemd/system/backtest-celery.service
```

```ini
[Unit]
Description=AI Backtest Celery Worker
After=network.target redis.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/ai-backtest-platform/backend
Environment="PATH=/path/to/ai-backtest-platform/backend/venv/bin"
ExecStart=/path/to/ai-backtest-platform/backend/venv/bin/celery -A app.services.celery_app.celery worker --loglevel=info -Q default --concurrency=8
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### 3. Enable và Start services
```bash
sudo systemctl daemon-reload
sudo systemctl enable backtest-backend backtest-celery
sudo systemctl start backtest-backend backtest-celery

# Check status
sudo systemctl status backtest-backend
sudo systemctl status backtest-celery
```

---

## 6. Cấu Hình Nginx (Production)

```bash
sudo nano /etc/nginx/sites-available/backtest
```

```nginx
server {
    listen 80;
    server_name your_domain.com;

    # Frontend
    location / {
        root /path/to/ai-backtest-platform/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # API Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # SSE support
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/backtest /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 7. Truy Cập

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 (dev) hoặc http://localhost:4173 (preview) |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## 8. Troubleshooting

### Redis không kết nối được
```bash
# Check Redis status
redis-cli ping
# Expected: PONG

# If not running
sudo systemctl start redis-server
```

### Celery không nhận task
```bash
# Check Celery logs
celery -A app.services.celery_app.celery inspect active

# Restart Celery
pkill -f celery
celery -A app.services.celery_app.celery worker --loglevel=info -Q default --concurrency=8
```

### Port đã được sử dụng
```bash
# Find process using port
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Windows: Celery worker bị lỗi
Windows không hỗ trợ fork, phải dùng:
```powershell
celery -A app.services.celery_app.celery worker --loglevel=info -Q default --pool=solo
```

---

## 9. Performance Tuning

### Biến môi trường quan trọng

```env
# CPU Workers (số core - 2)
MAX_THREAD_WORKERS=16

# Celery concurrency (số core / 2)
CELERY_CONCURRENCY=8

# Enable parallel fitness evaluation
PARALLEL_FITNESS=true

# Numba threads
NUMBA_NUM_THREADS=8

# Cache sizes (tăng nếu có nhiều RAM)
INDICATOR_CACHE_SIZE=4096
DATA_CACHE_SIZE=1024
INDICATOR_RESULT_CACHE_SIZE=2048

# Redis connection pool
REDIS_POOL_SIZE=50
```

### Monitor Performance
```bash
# CPU/RAM usage
htop

# Redis stats
redis-cli info stats

# Celery inspect
celery -A app.services.celery_app.celery inspect stats
```

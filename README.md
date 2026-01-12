# AI Backtest Platform

A high-performance backtesting platform for trading strategies with AI-powered optimization.

## Project Structure

```
ai-backtest-platform/
├── frontend/           # React frontend (Vite + React 19)
│   ├── src/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── backend/            # FastAPI backend + Celery workers
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── api/              # API routes
│   │   └── services/         # Celery tasks, Redis helpers
│   ├── tests/                # Pytest tests
│   ├── node_api_backup/      # Archived Node.js API (reference)
│   ├── archive/              # Archived test files
│   ├── Dockerfile
│   └── requirements.txt
├── engine/             # Core Python backtest engine (DO NOT MODIFY)
│   ├── ai_agent.py
│   ├── backtest_engine.py
│   ├── chart_data.py
│   ├── run_optimizer.py
│   └── data/           # Market data CSV files
├── docker-compose.yml  # Docker orchestration
├── .env.example        # Environment variables template
└── README.md
```

## Tech Stack

- **Frontend**: React 19, Vite 7, Recharts, Lightweight Charts
- **Backend**: FastAPI, Celery, Redis
- **Engine**: Python (NumPy, Pandas) - core backtest logic
- **Infrastructure**: Docker, Docker Compose, Nginx

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-backtest-platform

# Copy environment file
cp .env.example .env
```

### 2. Build and Run

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f
```

### 3. Access Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Health**: http://localhost:8000/

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/parse-pine` | Parse Pine Script (mock) |
| POST | `/run-backtest` | Run single backtest |
| POST | `/optimize` | Run optimization |
| POST | `/ai-agent` | Run AI agent optimization |
| GET | `/ai-agent/progress/{jobId}` | Poll job progress |
| GET | `/ai-agent/progress-stream/{jobId}` | SSE progress stream |
| GET | `/ai-agent/result/{jobId}` | Get job result |
| POST | `/ai-agent/cancel/{jobId}` | Cancel running job |
| POST | `/chart-data` | Generate chart data |

## Development

### Run Backend Locally (without Docker)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (separate terminal)
celery -A app.services.celery_app.celery worker --loglevel=info
```

### Run Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
cd backend
pytest -v
```

## Environment Variables

See [.env.example](.env.example) for all configuration options:

```env
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
JOB_TIMEOUT_SEC=1800
MAX_STDOUT_SIZE=20971520
```

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Rebuild specific service
docker-compose up --build backend -d

# View logs
docker-compose logs -f celery

# Stop all services
docker-compose down

# Clean up volumes
docker-compose down -v
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Nginx     │────▶│   Backend   │
│   (React)   │     │   (Proxy)   │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐     ┌──────▼──────┐
                    │    Redis    │◀────│   Celery    │
                    │  (Broker)   │     │  (Worker)   │
                    └─────────────┘     └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Engine    │
                                        │ (Subprocess)│
                                        └─────────────┘
```

## Migration Notes

This project was migrated from Node.js/Express to Python/FastAPI:

- Original `api/server.js` → `backend/app/` (FastAPI)
- Same API contract maintained for frontend compatibility
- Engine scripts (`engine/*.py`) unchanged
- Celery replaces in-process spawning for background jobs
- Redis replaces in-memory Maps for progress/result storage

## License

MIT

# AI Backtest Platform

A full-stack trading strategy backtesting and optimization platform with AI-driven parameter search.

## Architecture

```
ai-backtest-platform/
├── frontend/              # React frontend (Vite + React 19)
├── backend/               # FastAPI backend + Celery workers
├── engine/                # Python backtesting engine (unchanged)
├── docker-compose.yml     # Docker orchestration
└── .env.example           # Environment variables template
```

### Components

- **Frontend**: React 19 SPA with Recharts and Lightweight Charts for visualization
- **Backend**: FastAPI REST API server
- **Celery Worker**: Background task processor for long-running jobs
- **Redis**: Message broker and result backend
- **Engine**: Python backtesting core (backtest_engine.py, ai_agent.py, optimizer.py)

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available

### Build and Run

```bash
# Clone the repository
git clone <repo-url>
cd ai-backtest-platform

# Copy environment file
cp .env.example .env

# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Flower (Celery monitor)**: http://localhost:5555 (optional, use `--profile monitoring`)

### Enable Celery Monitoring (Optional)

```bash
docker-compose --profile monitoring up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/parse-pine` | POST | Parse Pine Script code |
| `/run-backtest` | POST | Run single backtest |
| `/optimize` | POST | Run parameter optimization |
| `/ai-agent` | POST | Run AI-driven optimization |
| `/ai-agent/progress/{jobId}` | GET | Poll job progress |
| `/ai-agent/progress-stream/{jobId}` | GET | SSE progress stream |
| `/ai-agent/result/{jobId}` | GET | Get job result |
| `/ai-agent/cancel/{jobId}` | POST | Cancel running job |
| `/chart-data` | POST | Generate chart data |

## Development Setup

### Backend (Local)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Start Celery worker (separate terminal)
celery -A app.services.celery_app.celery worker --loglevel=info
```

### Frontend (Local)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Run Tests

```bash
cd backend
pytest -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CELERY_BROKER_URL` | Same as REDIS_URL | Celery broker URL |
| `CELERY_RESULT_BACKEND` | Same as REDIS_URL | Celery result backend URL |
| `FASTAPI_HOST` | `0.0.0.0` | FastAPI host |
| `FASTAPI_PORT` | `8000` | FastAPI port |
| `JOB_TIMEOUT_SEC` | `1800` | Max job execution time (30 min) |
| `MAX_STDOUT_SIZE` | `20971520` | Max output size (20MB) |
| `CELERY_CONCURRENCY` | `2` | Number of Celery workers |
| `PROJECT_ROOT` | `/app` | Root path for engine scripts |

## Verification Checklist

After starting with `docker-compose up --build`:

- [ ] Frontend loads at http://localhost:3000
- [ ] Backend health check: `curl http://localhost:8000/` returns `{"status": "ok"}`
- [ ] Parse Pine Script:
  ```bash
  curl -X POST http://localhost:8000/parse-pine \
    -H "Content-Type: application/json" \
    -d '{"pineCode": "// test"}'
  ```
- [ ] Submit AI Agent job:
  ```bash
  curl -X POST http://localhost:8000/ai-agent \
    -H "Content-Type: application/json" \
    -d '{
      "symbols": ["BTCUSDT"],
      "timeframes": ["1h"],
      "strategy": {
        "type": "ema_cross",
        "params": {"emaFast": [12], "emaSlow": [26]}
      }
    }'
  ```
- [ ] Check job progress: `curl http://localhost:8000/ai-agent/progress/{jobId}`
- [ ] Get job result: `curl http://localhost:8000/ai-agent/result/{jobId}`

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── health.py        # Health check endpoint
│   │   ├── parse_pine.py    # Pine Script parser
│   │   ├── jobs.py          # Job management (ai-agent, optimize, backtest)
│   │   └── chart.py         # Chart data endpoint
│   └── services/
│       ├── celery_app.py    # Celery configuration
│       ├── tasks.py         # Celery tasks
│       ├── engine_runner.py # Engine subprocess runner
│       └── progress_store.py # Redis progress storage
├── tests/                   # Pytest tests
├── Dockerfile
└── requirements.txt

frontend/
├── src/
│   ├── components/          # React components
│   ├── pages/               # Page components
│   └── api/                 # API client
├── Dockerfile
├── nginx.conf
└── package.json

engine/                      # Python backtest engine (unchanged)
├── ai_agent.py
├── backtest_engine.py
├── chart_data.py
├── optimizer.py
└── ...
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

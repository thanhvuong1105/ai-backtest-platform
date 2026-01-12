# AI Backtest Platform

<<<<<<< HEAD
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
=======
A comprehensive trading strategy backtesting and optimization platform using AI-assisted parameter selection.

## Features

- **Multi-strategy Support**: EMA Cross, RF+ST+RSI strategies
- **Real-time Optimization**: AI agent for fast strategy parameter optimization
- **Advanced Metrics**: Profit factor, Sharpe ratio, max drawdown, stability analysis
- **Web Dashboard**: Interactive optimizer dashboard built with React
- **Real-time Progress**: Server-Sent Events (SSE) for live progress tracking
>>>>>>> 3c2d5d481a9c9665e2de3a13dfda1e0ecdb6a6ba

## Project Structure

```
<<<<<<< HEAD
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
=======
ai-backtest-platform/
├── engine/              # Python backtesting engine
│   ├── ai_agent.py     # AI agent for parameter optimization
│   ├── backtest_engine.py
│   ├── strategies/     # Strategy implementations
│   └── ...
├── api/                 # Express.js API server
│   └── server.js
├── optimizer-dashboard/ # React frontend
│   └── src/
└── requirements.txt     # Python dependencies
```

## Setup Instructions

### Prerequisites

- **Node.js** 18+ (for API server and dashboard)
- **Python** 3.8+ (for backtesting engine)
- **npm** or **yarn** (for JavaScript package management)

### Windows Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-backtest-platform
   ```

2. **Create Python virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install JavaScript dependencies (API server):**
   ```bash
   cd api
   npm install
   cd ..
   ```

5. **Install JavaScript dependencies (Dashboard):**
   ```bash
   cd optimizer-dashboard
   npm install
   cd ..
   ```

### macOS/Linux Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-backtest-platform
   ```

2. **Create Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install JavaScript dependencies:**
   ```bash
   cd api && npm install && cd ..
   cd optimizer-dashboard && npm install && cd ..
   ```

## Running the Application

### 1. Start the API Server (Terminal 1)

From project root:
```bash
cd api
npm start
```
Server will run on `http://localhost:3002`

### 2. Start the Dashboard (Terminal 2)

From project root:
```bash
cd optimizer-dashboard
npm run dev
```
Dashboard will run on `http://localhost:5173` (or configured Vite port)

### 3. Verify Setup

- API Health: `http://localhost:3002/` → Should return `{"status":"ok"}`
- Dashboard: `http://localhost:5173/` → Should load the optimizer interface

## Key Technologies

### Backend
- **Express.js**: REST API and SSE streaming
- **Python 3**: Backtesting engine with threading
- **pandas/numpy**: Data processing and calculations

### Frontend
- **React**: Interactive UI
- **Vite**: Build tool and dev server
- **Server-Sent Events**: Real-time progress streaming

## API Endpoints

### Optimization
- `POST /optimize` - Run optimizer with config
- `POST /ai-agent` - Run AI agent for parameter selection
- `GET /ai-agent/progress/:jobId` - Poll progress (legacy)
- `GET /ai-agent/progress-stream/:jobId` - Stream progress (SSE, real-time)
- `GET /ai-agent/result/:jobId` - Get optimization results
- `POST /ai-agent/cancel/:jobId` - Cancel running job

### Utilities
- `POST /parse-pine` - Parse PineScript strategies
- `POST /chart-data` - Generate chart data
- `POST /run-backtest` - Run single backtest

## Configuration

### Environment Variables

**API Server** (api/.env or NODE_ENV):
- `PORT` - Server port (default: 3002)

**Dashboard** (optimizer-dashboard/.env):
- `VITE_API_BASE` - API server URL (default: http://localhost:3002)

## Performance Optimization

The AI Agent uses optimized threading:
- **ThreadPoolExecutor** with 2x CPU cores for I/O-bound data loading
- **Progress throttling** (200ms minimum between updates) to prevent log spam
- **SSE streaming** for real-time progress (replaces polling)

## Development

### Python Testing

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
python engine/ai_agent.py < config.json  # Test with config
```

### Dashboard Development

```bash
cd optimizer-dashboard
npm run dev  # Start with hot reload
npm run build  # Build for production
```

## Troubleshooting

### "Could not open requirements.txt" on Windows
- Ensure you're running from the project root directory
- Verify Python is installed: `python --version`
- Recreate virtual environment: `python -m venv venv --clear`

### SSE Connection Errors
- Check API server is running on correct port
- Verify `VITE_API_BASE` in dashboard matches API URL
- Check browser console for CORS errors

### Python Module Not Found Errors
- Verify virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python --version` (should be 3.8+)

## Performance Notes

- First run may be slower due to data loading from CSV
- Subsequent runs benefit from pandas LRU caching
- Use reasonable number of parameters to avoid excessive backtests
- Monitor system resources during large optimizations

## Future Enhancements

- [ ] WebSocket support for lower latency
- [ ] Batch result export (Excel, CSV)
- [ ] Strategy backtesting validation
- [ ] Parameter constraint optimization
- [ ] Multi-symbol portfolio optimization
>>>>>>> 3c2d5d481a9c9665e2de3a13dfda1e0ecdb6a6ba

## License

MIT
<<<<<<< HEAD
=======

## Support

For issues or questions, please check the project repository or contact the development team.
>>>>>>> 3c2d5d481a9c9665e2de3a13dfda1e0ecdb6a6ba

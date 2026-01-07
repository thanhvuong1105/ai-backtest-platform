# AI Backtest Platform

A comprehensive trading strategy backtesting and optimization platform using AI-assisted parameter selection.

## Features

- **Multi-strategy Support**: EMA Cross, RF+ST+RSI strategies
- **Real-time Optimization**: AI agent for fast strategy parameter optimization
- **Advanced Metrics**: Profit factor, Sharpe ratio, max drawdown, stability analysis
- **Web Dashboard**: Interactive optimizer dashboard built with React
- **Real-time Progress**: Server-Sent Events (SSE) for live progress tracking

## Project Structure

```
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

## License

MIT

## Support

For issues or questions, please check the project repository or contact the development team.

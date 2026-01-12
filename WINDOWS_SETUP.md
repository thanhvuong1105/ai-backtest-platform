# Windows Setup Guide

Quick reference for setting up the AI Backtest Platform on Windows.

## Step-by-Step Setup

### 1. Install Prerequisites

Ensure you have installed:
- **Python 3.8+**: Download from https://www.python.org/downloads/
  - ✅ Check "Add Python to PATH" during installation
- **Node.js 18+**: Download from https://nodejs.org/
  - ✅ Includes npm automatically
- **Git**: Download from https://git-scm.com/ (optional, for cloning)

Verify installation in PowerShell:
```powershell
python --version
node --version
npm --version
```

### 2. Clone Repository

Using Git (recommended):
```powershell
git clone https://github.com/thanhvuong1105/ai-backtest-platform.git
cd ai-backtest-platform
```

Or download as ZIP and extract manually.

### 3. Create and Activate Python Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

Expected output: `(venv)` should appear at start of each line in PowerShell.

### 4. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- pandas (data processing)
- numpy (numerical computing)
- python-dateutil (date utilities)
- pytz (timezone support)
- tzdata (timezone data)

### 5. Install API Server Dependencies

```powershell
cd api
npm install
cd ..
```

### 6. Install Dashboard Dependencies

```powershell
cd optimizer-dashboard
npm install
cd ..
```

## Running the Application

### Terminal 1: Start API Server

```powershell
cd api
npm start
```

Expected output:
```
API running on port 3002
```

### Terminal 2: Start Dashboard

```powershell
cd optimizer-dashboard
npm run dev
```

Expected output:
```
VITE v... running at:
  ➜  Local:   http://localhost:5173/
```

### Terminal 3: Keep Python Venv Active

Optional - for testing Python directly:
```powershell
venv\Scripts\activate
# Now you can run: python engine/ai_agent.py
```

## Verify Setup Works

1. **Check API**: Open browser and go to `http://localhost:3002/`
   - Should see: `{"status":"ok","message":"AI Backtest Platform API is running"}`

2. **Check Dashboard**: Open browser and go to `http://localhost:5173/`
   - Should see the optimizer interface load

3. **Test AI Agent**: In dashboard, try a simple optimization run
   - Should see progress streaming in real-time
   - JSON logs from port 3002 should appear smoothly without pausing

## Troubleshooting

### Python Virtual Environment Issues

**Problem**: `venv\Scripts\activate` not found or doesn't work
```powershell
# Recreate venv
python -m venv venv --clear
venv\Scripts\activate
```

**Problem**: "'python' is not recognized"
- Reinstall Python with "Add Python to PATH" checked
- Or add manually to PATH: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python313\`

### npm Install Errors

**Problem**: "npm: The term 'npm' is not recognized"
- Reinstall Node.js
- Restart PowerShell after installation

**Problem**: "ERR! code ERESOLVE"
```powershell
npm install --legacy-peer-deps
```

### Port Already in Use

**Problem**: "Port 3002 is already in use"
```powershell
# Find process using port 3002
netstat -ano | findstr :3002

# Kill the process (replace PID with the number found)
taskkill /PID <PID> /F
```

**Problem**: "Port 5173 is already in use"
```powershell
cd optimizer-dashboard
npm run dev -- --port 5174
```

### Python Module Not Found

**Problem**: "No module named 'pandas'"
```powershell
# Ensure venv is activated (should show (venv) in prompt)
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

### Slow Progress Logging

**Expected behavior** (after latest optimization):
- JSON logs from port 3002 should appear smoothly at regular intervals
- Progress updates throttled to max 1 per 200ms to avoid spam
- No long pauses or freezing

If you still see pauses:
1. Check CPU usage - may indicate system bottleneck
2. Ensure venv is properly activated
3. Restart both API and Python backtesting

## Common Commands

```powershell
# Activate Python environment
venv\Scripts\activate

# Deactivate Python environment
deactivate

# Install/update a package
pip install package-name

# List installed packages
pip list

# Check Python version
python --version

# Run tests (if available)
npm test

# Build for production
npm run build
```

## File Structure Reference

```
ai-backtest-platform/
├── api/
│   ├── server.js          # Main Express server
│   ├── package.json       # API dependencies
│   └── ...
├── engine/                # Python backtesting engine
│   ├── ai_agent.py        # Main AI optimization
│   ├── backtest_engine.py
│   ├── data_loader.py
│   └── ...
├── optimizer-dashboard/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── api/
│   ├── package.json       # Dashboard dependencies
│   └── vite.config.js
├── requirements.txt       # Python dependencies
├── README.md             # Full documentation
└── WINDOWS_SETUP.md      # This file
```

## Next Steps

1. Open `http://localhost:5173/` in your browser
2. Configure a trading strategy in the dashboard
3. Run optimization
4. View results in real-time with progress streaming

## Additional Resources

- **Python**: https://docs.python.org/3/
- **pandas**: https://pandas.pydata.org/
- **Node.js**: https://nodejs.org/en/docs/
- **React**: https://react.dev/
- **Express**: https://expressjs.com/

## Need Help?

- Check the main `README.md` for full documentation
- Check browser console (F12) for JavaScript errors
- Check terminal output for Python errors
- Ensure all three terminals are running (API, Dashboard, and optionally Python venv)

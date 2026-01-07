import express from "express";
import rateLimit from "express-rate-limit";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { parsePineStrategy } from "./agent/parsePine.js";

// ================== PATH SETUP ==================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// project root (ai-backtest-platform)
const PROJECT_ROOT = path.resolve(__dirname, "..");

// ================== APP ==================
const app = express();
const PORT = process.env.PORT || 3002;
const progressStore = new Map(); // jobId -> {progress,total,status,error}
const resultStore = new Map();   // jobId -> result
const childStore = new Map();    // jobId -> child process
const sseClients = new Map();    // jobId -> Set<res> for SSE streaming

// Helper: broadcast progress to all SSE clients for a job (defined early for hoisting)
function broadcastProgress(jobId, data) {
  const clients = sseClients.get(jobId);
  console.log(`📡 broadcastProgress: jobId=${jobId}, clients=${clients?.size || 0}, data=`, JSON.stringify(data));
  if (!clients || clients.size === 0) return;
  const msg = `data: ${JSON.stringify(data)}\n\n`;
  for (const client of clients) {
    try {
      client.write(msg);
    } catch (e) {
      console.warn(`⚠️ SSE write error for ${jobId}:`, e.message);
    }
  }
}

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }
  next();
});

// ================== MIDDLEWARE ==================
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 300, // allow more, progress polling can be frequent
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => req.path.startsWith("/ai-agent/progress"),
}));

app.use(express.json({ limit: "200kb" }));

function validateOptimizeCfg(cfg) {
  if (!cfg || typeof cfg !== "object") return "Config must be JSON object";
  if (!Array.isArray(cfg.symbols) || cfg.symbols.length === 0) return "symbols must be non-empty array";
  if (!Array.isArray(cfg.timeframes) || cfg.timeframes.length === 0) return "timeframes must be non-empty array";
  const strat = cfg.strategy || {};
  if (!strat.type) return "strategy.type is required";
  if (!strat.params || typeof strat.params !== "object") return "strategy.params is required";
  const params = strat.params;

  // Validate based on strategy type
  if (strat.type === "ema_cross") {
    if (!Array.isArray(params.emaFast) || !Array.isArray(params.emaSlow)) return "strategy.params.emaFast/emaSlow must be arrays";
    if (params.emaFast.length === 0 || params.emaSlow.length === 0) return "emaFast/emaSlow must be non-empty arrays";
  } else if (strat.type === "rf_st_rsi") {
    // RF+ST+RSI strategy - validate required arrays
    if (!Array.isArray(params.st_atrPeriod)) return "strategy.params.st_atrPeriod must be array";
    if (!Array.isArray(params.st_mult)) return "strategy.params.st_mult must be array";
    if (!Array.isArray(params.rf_period)) return "strategy.params.rf_period must be array";
    if (!Array.isArray(params.rf_mult)) return "strategy.params.rf_mult must be array";
  } else {
    return `Unknown strategy type: ${strat.type}`;
  }

  return null;
}

app.use((req, _, next) => {
  console.log(new Date().toISOString(), req.method, req.url);
  next();
});

// ================== HEALTH ==================
app.get("/", (_, res) => {
  res.json({
    status: "ok",
    message: "AI Backtest Platform API is running",
  });
});

// ================== PARSE PINE ==================
app.post("/parse-pine", (req, res) => {
  const { pineCode } = req.body;
  if (!pineCode) {
    return res.status(400).json({ error: "pineCode is required" });
  }

  const strategy = parsePineStrategy(pineCode);
  res.json({ success: true, strategy });
});

// ================== RUN BACKTEST ==================
app.post("/run-backtest", (req, res) => {
  const { strategy } = req.body;
  if (!strategy) {
    return res.status(400).json({ error: "strategy is required" });
  }

  const python = spawn(
    "python3",
    ["engine/backtest_engine.py"],
    {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONPATH: PROJECT_ROOT
      }
    }
  );

  let stdout = "";
  let stderr = "";

  python.stdout.on("data", d => stdout += d.toString());
  python.stderr.on("data", d => stderr += d.toString());

  python.on("close", code => {
    if (code !== 0) {
      return res.status(500).json({
        error: "Python backtest failed",
        detail: stderr
      });
    }

    try {
      res.json({
        success: true,
        result: JSON.parse(stdout)
      });
    } catch {
      res.status(500).json({
        error: "Invalid JSON from Python",
        raw: stdout
      });
    }
  });

  python.stdin.write(JSON.stringify(strategy));
  python.stdin.end();
});

// ================== RUN OPTIMIZER ==================
app.post("/optimize", (req, res) => {
  const cfg = req.body;
  const errMsg = validateOptimizeCfg(cfg);
  if (errMsg) return res.status(400).json({ error: errMsg });

  const python = spawn(
    "python3",
    ["engine/run_optimizer.py"],
    {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONPATH: PROJECT_ROOT
      }
    }
  );

  let stdout = "";
  let stderr = "";
  const MAX_OUTPUT = 20 * 1024 * 1024;
  let aborted = false;

  python.stdout.on("data", d => {
    if (aborted) return;
    stdout += d.toString();
    if (stdout.length > MAX_OUTPUT) {
      aborted = true;
      python.kill();
      return res.status(500).json({
        error: "Optimizer output too large",
        detail: "Exceeded 20MB stdout limit"
      });
    }
  });
  python.stderr.on("data", d => {
    if (aborted) return;
    stderr += d.toString();
  });

  python.on("close", code => {
    if (aborted) return;
    if (code !== 0) {
      return res.status(500).json({
        error: "Optimizer failed",
        detail: stderr
      });
    }

    try {
      res.json(JSON.parse(stdout));
    } catch {
      res.status(500).json({
        error: "Invalid JSON from optimizer",
        raw: stdout
      });
    }
  });

  python.stdin.write(JSON.stringify(cfg));
  python.stdin.end();
});

// ================== AI AGENT (optimize + chọn best) ==================
app.post("/ai-agent", (req, res) => {
  const cfg = req.body;
  const errMsg = validateOptimizeCfg(cfg);
  if (errMsg) return res.status(400).json({ error: errMsg });

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
  progressStore.set(jobId, { progress: 0, total: 0, status: "running" });

  const python = spawn(
    "python3",
    ["engine/ai_agent.py"],
    {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONPATH: PROJECT_ROOT,
        AI_PROGRESS: "1",
      }
    }
  );
  childStore.set(jobId, python);

  let rawOutput = "";
  let stderr = "";
  const MAX_OUTPUT = 20 * 1024 * 1024; // 20MB guard
  let aborted = false;

  python.stdout.on("data", d => {
    if (aborted) return;
    const chunk = d.toString();
    rawOutput += chunk;

    // Debug: log raw output
    console.log(`🐍 Python stdout (jobId=${jobId}): ${chunk.slice(0, 200)}...`);

    // Parse progress lines while collecting
    chunk.split("\n").forEach(line => {
      const s = line.trim();
      if (!s) return;
      try {
        const obj = JSON.parse(s);
        if (obj && typeof obj === "object" && "progress" in obj && "total" in obj && !("success" in obj)) {
          const progressData = {
            progress: obj.progress,
            total: obj.total,
            status: "running",
          };
          progressStore.set(jobId, progressData);
          // Broadcast to SSE clients immediately (real-time update)
          broadcastProgress(jobId, progressData);
        }
      } catch {
        // not json, ignore during streaming
      }
    });

    if (rawOutput.length > MAX_OUTPUT) {
      aborted = true;
      python.kill();
      const errorData = { progress: 0, total: 0, status: "error", error: "Output too large" };
      progressStore.set(jobId, errorData);
      broadcastProgress(jobId, errorData);
    }
  });
  python.stderr.on("data", d => {
    if (aborted) return;
    stderr += d.toString();
  });

  python.on("close", code => {
    childStore.delete(jobId);
    if (aborted) return;
    if (code !== 0) {
      const errorData = { progress: 0, total: 0, status: "error", error: stderr || "AI Agent failed" };
      progressStore.set(jobId, errorData);
      broadcastProgress(jobId, errorData);
      return;
    }

    try {
      // Find the last complete JSON object in raw output (skip progress lines)
      const lines = rawOutput.trim().split("\n");
      let finalJson = null;

      // Find the last line that is a complete JSON with "success" key (the final result)
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          if (obj && typeof obj === "object" && "success" in obj) {
            finalJson = obj;
            break;
          }
        } catch {
          // Not valid JSON, continue
        }
      }

      if (!finalJson) {
        throw new Error("No valid result JSON found");
      }

      resultStore.set(jobId, finalJson);
      const doneData = { progress: finalJson?.total || 1, total: finalJson?.total || 1, status: "done" };
      progressStore.set(jobId, doneData);
      broadcastProgress(jobId, doneData);
    } catch (e) {
      const errDetail = stderr ? `stderr: ${stderr.slice(0, 500)}` : `stdout: ${(rawOutput || "").slice(-500)}`;
      const errorData = { progress: 0, total: 0, status: "error", error: `Invalid JSON from AI Agent. ${errDetail}` };
      progressStore.set(jobId, errorData);
      broadcastProgress(jobId, errorData);
    }
  });

  python.stdin.write(JSON.stringify(cfg));
  python.stdin.end();

  res.json({ jobId });
});

// Progress endpoint (polling - legacy)
app.get("/ai-agent/progress/:jobId", (req, res) => {
  const job = progressStore.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

// ================== SSE PROGRESS STREAM (real-time, faster) ==================
// Client connects once, server pushes updates immediately when progress changes
app.get("/ai-agent/progress-stream/:jobId", (req, res) => {
  const jobId = req.params.jobId;
  const job = progressStore.get(jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.flushHeaders();

  // Register client
  if (!sseClients.has(jobId)) {
    sseClients.set(jobId, new Set());
  }
  sseClients.get(jobId).add(res);

  // Send initial state immediately
  res.write(`data: ${JSON.stringify(job)}\n\n`);

  // Heartbeat every 15s to keep connection alive
  const heartbeat = setInterval(() => {
    res.write(`: heartbeat\n\n`);
  }, 15000);

  // Cleanup on close
  req.on("close", () => {
    clearInterval(heartbeat);
    const clients = sseClients.get(jobId);
    if (clients) {
      clients.delete(res);
      if (clients.size === 0) sseClients.delete(jobId);
    }
  });
});

// Result endpoint
app.get("/ai-agent/result/:jobId", (req, res) => {
  const job = progressStore.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (job.status !== "done") return res.status(202).json({ status: job.status });
  const result = resultStore.get(req.params.jobId);
  if (!result) return res.status(404).json({ error: "Result not found" });
  res.json(result);
});

// Cancel endpoint
app.post("/ai-agent/cancel/:jobId", (req, res) => {
  const jobId = req.params.jobId;
  const child = childStore.get(jobId);
  if (child) {
    child.kill();
    childStore.delete(jobId);
  }
  const cancelData = { progress: 0, total: 0, status: "canceled", error: "Canceled by user" };
  progressStore.set(jobId, cancelData);
  broadcastProgress(jobId, cancelData);
  res.json({ success: true });
});

// ================== CHART DATA (Debug View) ==================
app.post("/chart-data", (req, res) => {
  const cfg = req.body;

  // Basic validation
  if (!cfg.symbol) return res.status(400).json({ error: "symbol is required" });
  if (!cfg.timeframe) return res.status(400).json({ error: "timeframe is required" });
  if (!cfg.strategy?.type) return res.status(400).json({ error: "strategy.type is required" });

  const python = spawn(
    "python3",
    ["engine/chart_data.py"],
    {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONPATH: PROJECT_ROOT
      }
    }
  );

  let stdout = "";
  let stderr = "";

  python.stdout.on("data", d => stdout += d.toString());
  python.stderr.on("data", d => stderr += d.toString());

  python.on("close", code => {
    if (code !== 0) {
      return res.status(500).json({
        error: "Chart data generation failed",
        detail: stderr
      });
    }

    try {
      res.json(JSON.parse(stdout));
    } catch {
      res.status(500).json({
        error: "Invalid JSON from chart_data.py",
        raw: stdout
      });
    }
  });

  python.stdin.write(JSON.stringify(cfg));
  python.stdin.end();
});

// ================== START ==================
app.listen(PORT, () => {
  console.log(`API running on port ${PORT}`);
});

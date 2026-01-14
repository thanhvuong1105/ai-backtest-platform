// API Base URL
// In Docker production: use /api (nginx proxy)
// In development: use direct URL to backend
const API_BASE = import.meta.env.VITE_API_URL || "/api";

// Default timeouts
const DEFAULT_TIMEOUT = 30000; // 30 seconds for normal requests
const LONG_TIMEOUT = 60000; // 60 seconds for long operations

// Quant Brain modes
export const QUANT_BRAIN_MODE = {
  ULTRA: "ultra", // Maximum speed, rank by FastScore (PnL/DD)
  FAST: "fast",   // Quick exploration, rank by PnL
  BRAIN: "brain", // Deep learning, rank by BrainScore
};

/**
 * Create AbortController with timeout
 */
function createTimeoutController(timeoutMs = DEFAULT_TIMEOUT) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return { controller, timeoutId };
}

async function postJson(path, body, timeoutMs = DEFAULT_TIMEOUT) {
  const { controller, timeoutId } = createTimeoutController(timeoutMs);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }

    return res.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(`Request timeout after ${timeoutMs}ms`);
    }
    throw error;
  }
}

async function getJson(path, timeoutMs = DEFAULT_TIMEOUT) {
  const { controller, timeoutId } = createTimeoutController(timeoutMs);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(await res.text());
    }

    return res.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(`Request timeout after ${timeoutMs}ms`);
    }
    throw error;
  }
}

export async function runOptimize(cfg) {
  return postJson("/optimize", cfg, LONG_TIMEOUT);
}

export async function runAiAgent(cfg) {
  return postJson("/ai-agent", cfg, LONG_TIMEOUT);
}

export async function startAiAgent(cfg) {
  return postJson("/ai-agent", cfg, LONG_TIMEOUT); // returns {jobId}
}

export async function getAiAgentProgress(jobId) {
  return getJson(`/ai-agent/progress/${jobId}`, 10000); // 10s timeout for polling
}

export async function getAiAgentResult(jobId) {
  const { controller, timeoutId } = createTimeoutController(LONG_TIMEOUT);

  try {
    const res = await fetch(`${API_BASE}/ai-agent/result/${jobId}`, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (res.status === 202) return { status: "running" };
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === "AbortError") {
      throw new Error(`Request timeout after ${LONG_TIMEOUT}ms`);
    }
    throw error;
  }
}

export async function cancelAiAgent(jobId) {
  return postJson(`/ai-agent/cancel/${jobId}`, {}, 10000); // 10s timeout for cancel
}

/**
 * Start Quant AI Brain optimization
 * Self-learning optimization with long-term memory
 *
 * Supports three modes:
 * - "ultra": Maximum speed, rank by FastScore (PnL/DD), minimal validation
 * - "fast": Quick exploration, rank by PnL, no robustness, no memory write
 * - "brain": Deep learning, rank by BrainScore, robustness required, writes to memory
 *
 * @param {Object} cfg - Configuration object
 * @param {string} cfg.mode - "ultra", "fast", or "brain" (default: "fast")
 */
export async function startQuantBrain(cfg) {
  return postJson("/quant-brain", cfg, LONG_TIMEOUT); // returns {jobId}
}

/**
 * Start Quant Brain in ULTRA mode (maximum speed)
 * - Rank by FastScore = PnL / MaxDD
 * - Minimal coherence check
 * - No robustness, no memory write
 * - Early termination at 50% DD
 */
export async function startQuantBrainUltra(cfg) {
  return startQuantBrain({ ...cfg, mode: "ultra" });
}

/**
 * Start Quant Brain in FAST mode (quick exploration)
 * - Rank by PnL (highest profit)
 * - No robustness testing
 * - No memory write
 */
export async function startQuantBrainFast(cfg) {
  return startQuantBrain({ ...cfg, mode: "fast" });
}

/**
 * Start Quant Brain in BRAIN mode (deep learning)
 * - Rank by BrainScore (risk-adjusted)
 * - Robustness testing required
 * - Writes to ParamMemory
 */
export async function startQuantBrainDeep(cfg) {
  return startQuantBrain({ ...cfg, mode: "brain" });
}

/**
 * Create SSE EventSource for real-time progress streaming
 * @param {string} jobId - Job ID to stream progress for
 * @param {object} callbacks - Event callbacks
 * @param {function} callbacks.onProgress - Called with progress data {progress, total, status}
 * @param {function} callbacks.onDone - Called when job completes with final data
 * @param {function} callbacks.onError - Called on error with error message
 * @returns {EventSource} - EventSource instance (call .close() to cleanup)
 */
export function createProgressStream(jobId, callbacks = {}) {
  const { onProgress, onDone, onError } = callbacks;

  const url = `${API_BASE}/ai-agent/progress-stream/${jobId}`;
  const eventSource = new EventSource(url);

  eventSource.onopen = () => {
    console.log(`[SSE] Connected to progress stream for job ${jobId}`);
  };

  eventSource.addEventListener("progress", (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onProgress) {
        onProgress(data);
      }
    } catch (e) {
      console.error("[SSE] Failed to parse progress event:", e);
    }
  });

  eventSource.addEventListener("done", (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onDone) {
        onDone(data);
      }
      eventSource.close();
    } catch (e) {
      console.error("[SSE] Failed to parse done event:", e);
    }
  });

  eventSource.onerror = (error) => {
    console.error("[SSE] EventSource error:", error);
    if (onError) {
      onError(error);
    }
    // Don't close on error - EventSource will auto-reconnect
  };

  return eventSource;
}

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:3002";

async function postJson(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }

  return res.json();
}

export async function runOptimize(cfg) {
  return postJson("/optimize", cfg);
}

export async function runAiAgent(cfg) {
  return postJson("/ai-agent", cfg);
}

export async function startAiAgent(cfg) {
  return postJson("/ai-agent", cfg); // returns {jobId}
}

// Legacy polling (fallback)
export async function getAiAgentProgress(jobId) {
  const res = await fetch(`${API_BASE}/ai-agent/progress/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * SSE Progress Stream - Real-time updates (much faster than polling)
 * @param {string} jobId - Job ID to track
 * @param {function} onProgress - Callback when progress updates: (data) => void
 * @param {function} onError - Callback on error: (error) => void
 * @returns {function} cleanup - Call to close the connection
 */
export function subscribeAiAgentProgress(jobId, onProgress, onError) {
  const url = `${API_BASE}/ai-agent/progress-stream/${jobId}`;
  const eventSource = new EventSource(url);
  let hasReceivedData = false;
  let errorCount = 0;

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      hasReceivedData = true;
      errorCount = 0; // Reset error count on successful message
      onProgress(data);

      // Auto-close when job is done/error/canceled
      if (data.status === "done" || data.status === "error" || data.status === "canceled") {
        eventSource.close();
      }
    } catch (e) {
      console.warn("SSE parse error:", e);
    }
  };

  eventSource.onerror = (err) => {
    errorCount++;
    console.warn(`SSE error (count=${errorCount}, hasData=${hasReceivedData}):`, err);

    // Only trigger fallback if:
    // 1. Never received any data (connection failed from start), OR
    // 2. Multiple consecutive errors (connection truly broken)
    if (!hasReceivedData || errorCount >= 3) {
      eventSource.close();
      if (onError) onError(err);
    }
    // Otherwise, let SSE auto-reconnect
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

export async function getAiAgentResult(jobId) {
  const res = await fetch(`${API_BASE}/ai-agent/result/${jobId}`);
  if (res.status === 202) return { status: "running" };
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelAiAgent(jobId) {
  const res = await fetch(`${API_BASE}/ai-agent/cancel/${jobId}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

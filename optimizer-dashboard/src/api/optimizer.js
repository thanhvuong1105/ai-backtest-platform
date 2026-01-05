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

export async function getAiAgentProgress(jobId) {
  const res = await fetch(`${API_BASE}/ai-agent/progress/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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

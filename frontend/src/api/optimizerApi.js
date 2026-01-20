// API Base URL
// In Docker production: use /api (nginx proxy)
// In development: use direct URL to backend
// For AWS EC2 deployment: use the public IP
const API_BASE = import.meta.env.VITE_API_URL || "http://54.206.158.194:8000/api";

export async function runOptimize(cfg) {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }

  return res.json();
}

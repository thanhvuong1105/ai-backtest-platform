/**
 * Input:
 *  results = backend result.all
 * Output:
 *  [
 *    { time: "...", strat_0: 0, strat_1: 0 },
 *    { time: "...", strat_0: 0.3, strat_1: -0.1 },
 *    ...
 *  ]
 */

/**
 * Tính equity curve từ trades và properties (client-side).
 */
export function computeEquityCurveFromTrades(trades = [], properties = {}) {
  const {
    initialCapital = 1000000,
    orderSize = { value: 1000, type: "fixed" },
    commission = { value: 0, type: "percent" },
    slippage = 0,
  } = properties || {};

  let equity = Number(initialCapital) || 0;
  const curve = [];

  const sorted = [...trades].sort(
    (a, b) =>
      Number(a.exit_time || a.exitTime || a.time || 0) -
      Number(b.exit_time || b.exitTime || b.time || 0)
  );

  for (const t of sorted) {
    const side = (t.side || "").toLowerCase();
    const entry = Number(t.entry_price || t.entry || 0);
    const exit = Number(t.exit_price || t.exit || 0);
    if (!entry || !exit) continue;

    const positionSize =
      (orderSize?.type || "percent") === "percent"
        ? equity * (Number(orderSize?.value || 0) / 100)
        : Number(orderSize?.value || 0);
    if (positionSize <= 0) continue;

    const qty = positionSize / entry;
    const gross =
      (side === "short" ? entry - exit : exit - entry) *
      qty;

    const comm =
      (commission?.type || "percent") === "percent"
        ? positionSize * (Number(commission?.value || 0) / 100)
        : Number(commission?.value || 0);

    const slip = Number(slippage || 0) * qty; // tick value ~1 quote per tick
    const net = gross - comm - slip;

    equity += net;
    curve.push({
      time: Number(t.exit_time || t.exitTime || t.time || 0),
      equity,
    });
  }

  return curve;
}

/**
 * Parse time string to timestamp (milliseconds).
 * Supports formats: "YYYY-MM-DD HH:mm:ss" or epoch ms
 * Backend outputs UTC+7 (Ho Chi Minh) time without timezone suffix.
 * We parse it consistently across all strategies for proper alignment.
 */
function parseTimeToMs(timeStr) {
  if (typeof timeStr === "number") return timeStr;
  if (!timeStr) return 0;
  // Handle "YYYY-MM-DD HH:mm:ss" format
  // Treat as UTC+7 by appending timezone offset for consistent parsing
  const normalized = timeStr.replace(" ", "T");
  // If no timezone info, append +07:00 (UTC+7 Ho Chi Minh)
  const withTz = normalized.includes("+") || normalized.includes("Z")
    ? normalized
    : normalized + "+07:00";
  const d = new Date(withTz);
  return isNaN(d.getTime()) ? 0 : d.getTime();
}

/**
 * Get timeframe in minutes from string like "15m", "1h", "4h", "1D"
 */
function getTfMinutes(tf) {
  if (!tf) return 60; // default 1h
  const match = tf.match(/^(\d+)(m|h|d|D|w|W)$/i);
  if (!match) return 60;
  const val = parseInt(match[1], 10);
  const unit = match[2].toLowerCase();
  switch (unit) {
    case "m": return val;
    case "h": return val * 60;
    case "d": return val * 1440;
    case "w": return val * 10080;
    default: return 60;
  }
}

/**
 * Build series cho chart, hỗ trợ chế độ %PnL hoặc Equity.
 * mode = "pct" | "equity"
 *
 * FIX: Sử dụng TIME-BASED alignment thay vì INDEX-based
 * - Tạo global timeline từ TF nhỏ nhất
 * - Forward-fill equity cho mỗi strategy theo timeline chung
 */
export function buildEquitySeries(results, mode = "pct", fallbackProperties) {
  if (!results || results.length === 0) return [];

  const valid = results
    .map((r, idx) => ({
      ...r,
      // Match Dashboard.jsx key format: strategyId || `s${idx + 1}` (1-indexed)
      _chartKey: r.strategyId || `s${idx + 1}`,
    }))
    .map((r) => {
      if (mode === "equity" && (!r.equityCurve || r.equityCurve.length === 0)) {
        if (Array.isArray(r.trades) && r.trades.length) {
          return {
            ...r,
            equityCurve: computeEquityCurveFromTrades(r.trades, fallbackProperties),
          };
        }
      }
      return r;
    })
    .map((r) => {
      if (mode !== "equity") return r;
      const initCap = fallbackProperties?.initialCapital;
      if (!initCap || !r.equityCurve || !r.equityCurve.length) return r;
      const startEq = r.equityCurve[0]?.equity || 0;
      if (!startEq || startEq === initCap) return r;
      const scale = initCap / startEq;
      return {
        ...r,
        equityCurve: r.equityCurve.map((p) => ({
          ...p,
          equity: Number((p.equity * scale).toFixed(2)),
        })),
      };
    })
    .filter((r) => Array.isArray(r.equityCurve) && r.equityCurve.length > 0);

  if (valid.length === 0) return [];

  // ===== 1. Tìm TF nhỏ nhất để tạo global timeline =====
  const tfMinutesArr = valid.map((r) => {
    const tf = r.meta?.timeframe || r.timeframe || "1h";
    return getTfMinutes(tf);
  });
  const smallestTfMinutes = Math.min(...tfMinutesArr);

  // ===== 2. Tìm global start/end time từ tất cả strategies =====
  let globalStartMs = Infinity;
  let globalEndMs = -Infinity;

  valid.forEach((r) => {
    if (!r.equityCurve || r.equityCurve.length === 0) return;
    const firstMs = parseTimeToMs(r.equityCurve[0]?.time);
    const lastMs = parseTimeToMs(r.equityCurve[r.equityCurve.length - 1]?.time);
    if (firstMs > 0 && firstMs < globalStartMs) globalStartMs = firstMs;
    if (lastMs > 0 && lastMs > globalEndMs) globalEndMs = lastMs;
  });

  if (globalStartMs === Infinity || globalEndMs === -Infinity) {
    return [];
  }

  // ===== 3. Tạo global timeline với step = smallest TF =====
  // Limit max points to prevent performance issues with very long periods
  const MAX_POINTS = 5000;
  let stepMs = smallestTfMinutes * 60 * 1000;
  const estimatedPoints = Math.ceil((globalEndMs - globalStartMs) / stepMs);

  // If too many points, increase step size to fit within limit
  if (estimatedPoints > MAX_POINTS) {
    stepMs = Math.ceil((globalEndMs - globalStartMs) / MAX_POINTS);
  }

  const globalTimeline = [];
  for (let t = globalStartMs; t <= globalEndMs; t += stepMs) {
    globalTimeline.push(t);
  }

  if (globalTimeline.length === 0) {
    return [];
  }

  // ===== 4. Convert mỗi strategy equity curve sang Map<timestamp, equity> =====
  const strategyMaps = valid.map((r) => {
    const map = new Map();
    (r.equityCurve || []).forEach((p) => {
      const ms = parseTimeToMs(p.time);
      if (ms > 0) {
        map.set(ms, p.equity);
      }
    });
    const timestamps = Array.from(map.keys()).sort((a, b) => a - b);
    const firstTimestamp = timestamps[0] || globalStartMs;
    return {
      key: r._chartKey,
      map,
      startEquity: r.equityCurve[0]?.equity || 1,
      timestamps,
      firstTimestamp, // Track when this strategy actually starts
    };
  });

  // ===== 5. Build output với TIME-based alignment + forward-fill =====
  const output = [];

  // Helper to format time in UTC+7 (Ho Chi Minh)
  const formatTimeHcm = (ms) => {
    const d = new Date(ms);
    // Add 7 hours to UTC to get UTC+7
    const hcm = new Date(d.getTime() + 7 * 60 * 60 * 1000);
    return hcm.toISOString().replace("T", " ").slice(0, 19);
  };

  for (const t of globalTimeline) {
    const point = {
      time: formatTimeHcm(t),
    };

    for (const { key, map, startEquity, timestamps, firstTimestamp } of strategyMaps) {
      // Skip points before this strategy starts (leave undefined for Recharts)
      if (t < firstTimestamp) {
        // Don't set point[key] - line will start from first valid point
        continue;
      }

      let equity;

      if (map.has(t)) {
        // Exact match - use this equity value
        equity = map.get(t);
      } else {
        // Forward-fill: find the most recent equity value before this time
        // Binary search for the largest timestamp <= t
        let lo = 0;
        let hi = timestamps.length - 1;
        let bestIdx = -1;
        while (lo <= hi) {
          const mid = Math.floor((lo + hi) / 2);
          if (timestamps[mid] <= t) {
            bestIdx = mid;
            lo = mid + 1;
          } else {
            hi = mid - 1;
          }
        }

        if (bestIdx >= 0) {
          equity = map.get(timestamps[bestIdx]);
        } else {
          // Should not happen since we check t >= firstTimestamp above
          continue;
        }
      }

      // Calculate value based on mode
      const value =
        mode === "equity"
          ? Number(equity)
          : ((equity - startEquity) / startEquity) * 100;

      point[key] = Number(value.toFixed(2));
    }

    output.push(point);
  }

  return output;
}

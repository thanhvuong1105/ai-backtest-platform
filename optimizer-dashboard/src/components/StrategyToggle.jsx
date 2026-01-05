import { useState } from "react";

export default function StrategyToggle({ results, visible, setVisible, onSelect }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const formatLabel = (r, idx) => {
    const params = r.params || {};
    if (r.strategy === "rf_st_rsi" || r.strategyType === "rf_st_rsi") {
      const parts = [];
      if (params.st_atrPeriod) parts.push(`ST-ATR ${params.st_atrPeriod}`);
      if (params.st_mult) parts.push(`ST-Mult ${params.st_mult}`);
      if (params.rf_period) parts.push(`RF-Per ${params.rf_period}`);
      if (params.rf_mult) parts.push(`RF-Mult ${params.rf_mult}`);
      return parts.length ? parts.join(" · ") : "RF+ST+RSI";
    }
    if (params.emaFast && params.emaSlow) {
      return `EMA ${params.emaFast}/${params.emaSlow}`;
    }
    return `s${idx + 1}`;
  };

  // Map param keys to friendly labels
  const paramLabels = {
    // Entry Settings
    showDualFlip: "Enable Dual Flip",
    showRSI: "Enable RSI Divergence",
    st_useATR: "ST use ATR",
    st_atrPeriod: "ST ATR Period",
    st_mult: "ST Multiplier",
    rf_period: "RF Period",
    rf_mult: "RF Multiplier",
    rsi_length: "RSI Length",
    rsi_ma_length: "MA Length on RSI",
    // Stop Loss
    sl_st_useATR: "SL ST use ATR",
    sl_st_atrPeriod: "SL ST ATR Period",
    sl_st_mult: "SL ST Mult",
    sl_rf_period: "SL RF Period",
    sl_rf_mult: "SL RF Mult",
    // Take Profit - Dual Flip
    tp_dual_st_atrPeriod: "TP Dual ST ATR Period",
    tp_dual_st_mult: "TP Dual ST Mult",
    tp_dual_rr_mult: "TP Dual R:R Mult",
    // Take Profit - RSI
    tp_rsi_st_atrPeriod: "TP RSI ST ATR Period",
    tp_rsi_st_mult: "TP RSI ST Mult",
    tp_rsi_rr_mult: "TP RSI R:R Mult",
    // EMA
    emaFast: "EMA Fast",
    emaSlow: "EMA Slow",
    // Debug
    debug: "Debug",
  };

  const fmt = (v, digits = 2) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits);
  };

  const fmtMoney = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  const buildTooltip = (r) => {
    const lines = [];
    const params = r.params || {};
    const summary = r.summary || {};

    // Strategy params
    lines.push("=== Params ===");
    Object.entries(params).forEach(([k, v]) => {
      const label = paramLabels[k] || k;
      const displayValue = typeof v === "boolean" ? (v ? "Yes" : "No") : v;
      lines.push(`${label}: ${displayValue}`);
    });

    // Summary metrics - same as StrategyTable
    lines.push("");
    lines.push("=== Metrics ===");
    lines.push(`Symbol: ${r.symbol || "-"}`);
    lines.push(`TF: ${r.timeframe || "-"}`);
    lines.push(`Total PnL: ${fmtMoney(summary.netProfit)} (${fmt(summary.netProfitPct)}%)`);
    lines.push(`PF: ${fmt(summary.profitFactor)}`);
    lines.push(`WR: ${fmt(summary.winrate, 1)}%`);
    lines.push(`DD: ${fmt(summary.maxDrawdownPct)}%`);
    lines.push(`Score: ${fmt(summary.score)}`);
    if (summary.totalTrades != null) lines.push(`Trades: ${summary.totalTrades}`);

    return lines.join("\n");
  };

  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
      {results.map((r, i) => {
        const key = r.strategyId || `s${i + 1}`;
        const color = palette[i % palette.length];
        const active = visible[key];
        return (
          <div
            key={key}
            style={{ position: "relative", display: "inline-block" }}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            <button
              onClick={() =>
                {
                  setVisible(v => ({ ...v, [key]: !v[key] }));
                  onSelect?.(key);
                }
              }
              style={{
                padding: "6px 10px",
                borderRadius: 999,
                border: `1px solid ${active ? color : "#334155"}`,
                background: active ? "rgba(255,255,255,0.08)" : "transparent",
                color: active ? color : "#e5e7eb",
                cursor: "pointer",
              }}
            >
              #{r.rank || i + 1} {key} {r.symbol} {r.timeframe} · {formatLabel(r, i)}
            </button>
            {hoveredIdx === i && (
              <div
                style={{
                  position: "absolute",
                  top: "100%",
                  left: 0,
                  marginTop: 8,
                  padding: "10px 14px",
                  background: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: 8,
                  color: "#e5e7eb",
                  fontSize: 12,
                  whiteSpace: "pre",
                  zIndex: 1000,
                  minWidth: 180,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                }}
              >
                {buildTooltip(r)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

const palette = [
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
  "#eab308",
  "#6366f1",
];

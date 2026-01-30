import { useState } from "react";

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

export default function StrategyTable({
  rows = [],
  bestKey,
  onSelectRow,
  pageSize = 20,
  currentPage = 1,
  // New props for Memory selection
  selectedForMemory = {},  // { strategyId: true/false }
  onToggleMemorySelect,    // (strategyId) => void
}) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (!rows.length) return <p style={{ opacity: 0.7 }}>No strategies</p>;

  const mono = { fontFamily: "JetBrains Mono, IBM Plex Mono, monospace" };
  const formatLabel = (r) => {
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
    return r.strategyId || r.strategy || "Strategy";
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

    // Summary metrics - same format as table columns
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

  const isBest = (r) => {
    if (!bestKey) return false;
    return bestKey.symbol === r.symbol &&
      bestKey.timeframe === r.timeframe &&
      JSON.stringify(bestKey.params || {}) === JSON.stringify(r.params || {});
  };

  const headerStyle = {
    textAlign: "left",
    padding: "10px 12px",
    fontSize: 12,
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#94a3b8",
  };

  const rowBase = {
    padding: "9px 12px",
    fontSize: 13,
  };

  const pfColor = (pf) => (pf >= 1.5 ? "#22c55e" : pf >= 1 ? "#a5e0b9" : "#f97316");
  const ddColor = (dd) => (dd > 20 ? "#ef4444" : "#f59e0b");
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

  const start = (currentPage - 1) * pageSize;
  const paged = rows.slice(start, start + pageSize);

  // Checkbox style
  const checkboxStyle = {
    width: 16,
    height: 16,
    cursor: "pointer",
    accentColor: "#a855f7",
  };

  return (
    <table width="100%" style={{ borderCollapse: "collapse" }}>
      <thead>
        <tr>
          {onToggleMemorySelect && <th style={{ ...headerStyle, width: 40, textAlign: "center" }}>Sel</th>}
          <th style={headerStyle}>Symbol</th>
          <th style={headerStyle}>TF</th>
          <th style={headerStyle}>Total PnL</th>
          <th style={headerStyle}>PF</th>
          <th style={headerStyle}>WR</th>
          <th style={headerStyle}>DD</th>
          <th style={headerStyle}>Trades</th>
          <th style={headerStyle}>Score</th>
        </tr>
      </thead>
      <tbody>
        {paged.map((r, i) => {
          const best = isBest(r);
          const rankLabel = r.rank ? `#${r.rank} ` : "";
          const globalIdx = start + i;
          return (
            <tr
              key={globalIdx}
              onClick={() => onSelectRow?.(r, globalIdx)}
              onMouseEnter={() => setHoveredIdx(globalIdx)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{
                background: selectedForMemory[r.strategyId]
                  ? "rgba(168,85,247,0.1)"
                  : i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent",
                borderLeft: best ? "3px solid #22c55e" : selectedForMemory[r.strategyId] ? "3px solid #a855f7" : "3px solid transparent",
                boxShadow: best ? "0 6px 18px rgba(34,197,94,0.15)" : "none",
                cursor: onSelectRow ? "pointer" : "default",
                position: "relative",
              }}
            >
              {onToggleMemorySelect && (
                <td style={{ ...rowBase, textAlign: "center" }} onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={!!selectedForMemory[r.strategyId]}
                    onChange={() => onToggleMemorySelect(r.strategyId)}
                    style={checkboxStyle}
                    title="Select for Add to Memory"
                  />
                </td>
              )}
              <td style={{ ...rowBase, position: "relative" }}>
                {best ? "★ " : ""}
                {rankLabel}
                {r.symbol} {r.timeframe} · {formatLabel(r)}
                {hoveredIdx === globalIdx && (
                  <div
                    style={{
                      position: "absolute",
                      top: "100%",
                      left: 0,
                      marginTop: 4,
                      padding: "10px 14px",
                      background: "#1e293b",
                      border: "1px solid #334155",
                      borderRadius: 8,
                      color: "#e5e7eb",
                      fontSize: 12,
                      whiteSpace: "pre",
                      zIndex: 1000,
                      minWidth: 200,
                      boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                    }}
                  >
                    {buildTooltip(r)}
                  </div>
                )}
              </td>
              <td style={{ ...rowBase, color: "#cbd5e1" }}>{r.timeframe}</td>
              <td style={{ ...rowBase, ...mono }}>
                {`${fmtMoney(r.summary?.netProfit)} (${fmt(r.summary?.netProfitPct, 2)}%)`}
              </td>
              <td style={{ ...rowBase, ...mono, color: pfColor(r.summary.profitFactor) }}>
                {fmt(r.summary.profitFactor)}
              </td>
              <td style={{ ...rowBase, ...mono, color: "#22c55e" }}>
                {fmt(r.summary.winrate, 1)}%
              </td>
              <td style={{ ...rowBase, ...mono, color: ddColor(r.summary.maxDrawdownPct) }}>
                {fmt(r.summary.maxDrawdownPct)}%
              </td>
              <td style={{ ...rowBase, ...mono }}>
                {r.summary?.totalTrades ?? "-"}
              </td>
              <td style={{ ...rowBase, ...mono }}>
                {fmt(r.summary.score)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

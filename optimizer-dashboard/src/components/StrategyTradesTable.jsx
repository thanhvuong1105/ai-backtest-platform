export default function StrategyTradesTable({
  trades = [],
  initialEquity = 1000000,
  onSelectTrade,
  selectedIndex,
}) {
  /**
   * Chuẩn hóa dữ liệu: tính Cumulative PnL, Drawdown, Net PnL %
   */
  const processed = normalizeTrades(trades, initialEquity);

  if (!processed.length) {
    return (
      <div
        style={{
          padding: "12px 14px",
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          color: "rgba(255,255,255,0.7)",
        }}
      >
        Chưa có trade hoàn chỉnh
      </div>
    );
  }

  return (
    <div
      style={{
        maxHeight: 420,
        overflow: "auto",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(15,20,35,0.6)",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 13,
        }}
      >
        <thead
          style={{
            position: "sticky",
            top: 0,
            background: "rgba(11,16,33,0.95)",
            zIndex: 2,
          }}
        >
          <tr>
            {[
              "Trade #",
              "Type",
              "Date / Time",
              "Signal",
              "Entry Price",
              "Exit Price",
              "Position size",
              "Net P&L",
              "Commission",
              "Drawdown",
              "Cumulative P&L",
            ].map((h) => (
              <th
                key={h}
                style={{
                  textAlign: "left",
                  padding: "10px 12px",
                  fontWeight: 700,
                  color: "#9ca3af",
                  borderBottom: "1px solid rgba(255,255,255,0.08)",
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {processed.map((t, i) => {
            const pnlColor = t.pnlValue >= 0 ? "#22c55e" : "#ef4444";
            const signalLabel = t.signal || "Long";
            const signalIsLong = signalLabel.toLowerCase() === "long";
            const signalIsShort = signalLabel.toLowerCase() === "short";
            const rowSelected = selectedIndex === i;
            return (
              <tr
                key={t.tradeNo}
                onClick={() => onSelectTrade?.(t.raw, i)}
                style={{
                  background: rowSelected
                    ? "rgba(34,197,94,0.12)"
                    : i % 2 === 0
                    ? "rgba(255,255,255,0.02)"
                    : "transparent",
                  cursor: "pointer",
                }}
              >
                <td style={{ ...cell, fontWeight: 700, fontFamily: mono }}>
                  {t.tradeNo}
                </td>
                <td style={{ ...cell, color: "#e5e7eb" }}>Exit</td>
                <td style={{ ...cell, color: "#d1d5db" }}>{t.time}</td>
                <td style={cell}>
                  <span
                    style={{
                      padding: "4px 10px",
                      borderRadius: 999,
                      fontWeight: 700,
                      fontSize: 12,
                      background: signalIsLong
                        ? "rgba(34,197,94,.16)"
                        : signalIsShort
                        ? "rgba(239,68,68,.16)"
                        : "rgba(255,255,255,.1)",
                      color: signalIsLong
                        ? "#22c55e"
                        : signalIsShort
                        ? "#ef4444"
                        : "#e5e7eb",
                    }}
                  >
                    {signalLabel}
                  </span>
                </td>
                <td style={cell}>{fmt(t.entryPrice)}</td>
                <td style={cell}>{fmt(t.exitPrice)}</td>
                <td style={cell}>
                  <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.3 }}>
                    <span style={{ fontWeight: 600 }}>{fmt(t.positionSize)}</span>
                    <span style={{ fontSize: 12, opacity: 0.75 }}>
                      {t.notional !== null ? formatNotional(t.notional) : "--"}
                    </span>
                  </div>
                </td>
                <td
                  style={{
                    ...cell,
                    color: pnlColor,
                    fontWeight: 700,
                    fontFamily: mono,
                  }}
                >
                  {`${fmt(t.pnlValue)} (${(t.pnlPct || 0).toFixed(2)}%)`}
                </td>
                <td style={{ ...cell, fontFamily: mono }}>
                  {t.commission !== null ? `${fmt(t.commission)} USDT` : "--"}
                </td>
                <td
                  style={{
                    ...cell,
                    color: t.ddPct < 0 ? "#ef4444" : "#9ca3af",
                    fontFamily: mono,
                  }}
                >
                  {`${t.ddPct.toFixed(2)}%`}
                </td>
                <td
                  style={{
                    ...cell,
                    color: t.cumPnl >= 0 ? "#22c55e" : "#ef4444",
                    fontFamily: mono,
                    fontWeight: 700,
                  }}
                >
                  {`${fmt(t.cumPnl)} (${(t.cumPnlPct || 0).toFixed(2)}%)`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const cell = {
  padding: "10px 12px",
  whiteSpace: "nowrap",
};

const mono = "JetBrains Mono, IBM Plex Mono, monospace";

function fmt(v) {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  return Number(v).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
}

function formatNotional(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  const abs = Math.abs(v);
  let val = v;
  let suffix = "";
  if (abs >= 1_000_000) {
    val = v / 1_000_000;
    suffix = " M";
  } else if (abs >= 1_000) {
    val = v / 1_000;
    suffix = " K";
  }
  return `${val.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix} USDT`;
}

function parseTime(str) {
  if (!str) return "-";
  // Backend returns time in UTC+7 format "YYYY-MM-DD HH:MM:SS"
  // Display as-is since it's already in Ho Chi Minh timezone
  return str;
}

function normalizeTrades(trades, initialEquity) {
  if (!trades || !trades.length) return [];

  const completed = trades.filter((t) => t.exit_time);
  if (!completed.length) return [];

  // Chronological order to compute cumulative correctly
  const sortedAsc = [...completed].sort((a, b) => {
    const ta = new Date(a.exit_time).getTime();
    const tb = new Date(b.exit_time).getTime();
    return ta - tb;
  });

  let cumPnl = 0;
  let peak = initialEquity;

  const computedAsc = sortedAsc.map((t) => {
    const pnlValue = Number(t.pnl || 0);
    const notional =
      t.entry_price && t.size ? Number(t.entry_price) * Number(t.size) : null;
    const pnlPct =
      notional && notional !== 0 ? (pnlValue / notional) * 100 : 0;

    cumPnl += pnlValue;
    const equityNow = initialEquity + cumPnl;
    peak = Math.max(peak, equityNow);
    const ddPct = peak ? ((equityNow - peak) / peak) * 100 : 0;

    return {
      time: parseTime(t.exit_time),
      signal: t.side || "Long",
      entryPrice: Number(t.entry_price),
      exitPrice: Number(t.exit_price),
      positionSize: t.size,
      notional,
      pnlValue,
      commission: Number(t.entry_fee || 0) + Number(t.exit_fee || 0),
      pnlPct,
      ddPct,
      cumPnl,
      cumPnlPct: initialEquity ? (cumPnl / initialEquity) * 100 : 0,
      raw: t,
    };
  });

  // Reverse for display (newest first) and number descending
  const reversed = [...computedAsc].reverse();
  return reversed.map((row, idx) => ({
    tradeNo: reversed.length - idx,
    ...row,
  }));
}

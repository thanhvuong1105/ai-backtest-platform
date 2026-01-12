export default function DebugTradeTable({ trades }) {
  const exportCSV = () => {
    if (!trades?.length) return;

    const headers = [
      "ID",
      "Entry Time",
      "Exit Time",
      "Entry Price",
      "Exit Price",
      "Side",
      "PnL %",
      "PnL USDT",
      "Fee",
      "Cumulative PnL",
    ];

    const rows = enrichedTrades.map((t) => [
      t.id,
      t.entry_time,
      t.exit_time,
      t.entry_price,
      t.exit_price,
      t.side,
      t.pnl,
      t.pnl_usdt,
      t.fee,
      t.cum_pnl_usdt,
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((r) => r.join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `trades_${Date.now()}.csv`;
    link.click();
  };

  if (!trades?.length) {
    return (
      <div style={{ color: "#9ca3af", padding: "20px", textAlign: "center" }}>
        No trades found
      </div>
    );
  }

  const winTrades = trades.filter((t) => (t.pnl_pct ?? t.pnl) > 0).length;
  const loseTrades = trades.filter((t) => (t.pnl_pct ?? t.pnl) < 0).length;
  const totalPnl = trades.reduce((sum, t) => sum + (t.pnl_pct ?? 0), 0);
  // Tính cumulative PnL theo thời gian (tăng dần) rồi hiển thị giảm dần
  const enrichedTrades = (() => {
    const asc = [...trades].sort((a, b) => {
      const ta = new Date(a.exit_time || a.entry_time || 0).getTime();
      const tb = new Date(b.exit_time || b.entry_time || 0).getTime();
      return ta - tb;
    });
    let cum = 0;
    return asc.map((t) => {
      const pnlUsdt = Number(t.pnl_usdt ?? t.pnl ?? 0);
      cum += pnlUsdt;
      const notional = t.notional ?? (t.entry_price && t.size ? Number(t.entry_price) * Number(t.size) : null);
      const pnlPct = t.pnl_pct !== undefined ? Number(t.pnl_pct) : (notional ? (pnlUsdt / notional) * 100 : 0);
      return { ...t, cum_pnl_usdt: cum, pnl_pct: pnlPct, pnl_usdt: pnlUsdt };
    });
  })();

  const sortedTrades = [...enrichedTrades].sort((a, b) => {
    const ta = new Date(a.exit_time || a.entry_time || 0).getTime();
    const tb = new Date(b.exit_time || b.entry_time || 0).getTime();
    return tb - ta;
  });

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "16px",
        }}
      >
        <div style={{ display: "flex", gap: "16px", color: "#d1d5db" }}>
          <span>
            Total: <strong>{trades.length}</strong>
          </span>
          <span style={{ color: "#22c55e" }}>
            Win: <strong>{winTrades}</strong>
          </span>
          <span style={{ color: "#ef4444" }}>
            Lose: <strong>{loseTrades}</strong>
          </span>
          <span style={{ color: totalPnl >= 0 ? "#22c55e" : "#ef4444" }}>
            Total PnL: <strong>{totalPnl.toFixed(2)}%</strong>
          </span>
        </div>
        <button
          onClick={exportCSV}
          style={{
            background: "#3b82f6",
            color: "white",
            border: "none",
            padding: "8px 16px",
            borderRadius: "6px",
            cursor: "pointer",
            fontSize: "14px",
          }}
        >
          Export CSV
        </button>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "14px",
          }}
        >
          <thead>
            <tr style={{ borderBottom: "1px solid #2d2d44" }}>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Entry Time</th>
              <th style={thStyle}>Exit Time</th>
              <th style={thStyle}>Entry Price</th>
              <th style={thStyle}>Exit Price</th>
              <th style={thStyle}>Side</th>
              <th style={thStyle}>PnL %</th>
              <th style={thStyle}>PnL USDT</th>
              <th style={thStyle}>Fee</th>
              <th style={thStyle}>Cumulative PnL</th>
            </tr>
          </thead>
          <tbody>
            {sortedTrades.map((trade, index) => (
              <tr
                key={trade.id}
                style={{
                  borderBottom: "1px solid #2d2d44",
                  background: trade.id % 2 === 0 ? "#1a1a2e" : "transparent",
                }}
              >
                <td style={tdStyle}>{sortedTrades.length - index}</td>
                <td style={tdStyle}>{formatTime(trade.entry_time)}</td>
                <td style={tdStyle}>
                  {trade.exit_time === "Open" ? (
                    <span style={{ color: "#f59e0b" }}>Open</span>
                  ) : (
                    formatTime(trade.exit_time)
                  )}
                </td>
                <td style={tdStyle}>{trade.entry_price?.toFixed(2)}</td>
                <td style={tdStyle}>
                  {trade.exit_time === "Open" ? "-" : trade.exit_price?.toFixed(2)}
                </td>
                <td style={tdStyle}>
                  <span
                    style={{
                      background: "#22c55e20",
                      color: "#22c55e",
                      padding: "2px 8px",
                      borderRadius: "4px",
                    }}
                  >
                    {trade.side}
                  </span>
                </td>
                <td
                  style={{
                    ...tdStyle,
                    color: trade.pnl >= 0 ? "#22c55e" : "#ef4444",
                    fontWeight: "600",
                  }}
                >
                  {trade.pnl >= 0 ? "+" : ""}
                  {trade.pnl?.toFixed(2)}%
                </td>
                <td
                  style={{
                    ...tdStyle,
                    color: trade.pnl_usdt >= 0 ? "#22c55e" : "#ef4444",
                  }}
                >
                  {trade.pnl_usdt >= 0 ? "+" : ""}
                  {trade.pnl_usdt?.toFixed(2)}
                </td>
                <td style={{ ...tdStyle, color: "#9ca3af" }}>
                  {(trade.fee ?? 0).toFixed(2)}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    color: (trade.cum_pnl_usdt || 0) >= 0 ? "#22c55e" : "#ef4444",
                    fontWeight: "600",
                  }}
                >
                  {(trade.cum_pnl_usdt || 0) >= 0 ? "+" : ""}
                  {(trade.cum_pnl_usdt || 0).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const thStyle = {
  textAlign: "left",
  padding: "12px 8px",
  color: "#9ca3af",
  fontWeight: "500",
  whiteSpace: "nowrap",
};

const tdStyle = {
  padding: "12px 8px",
  color: "#d1d5db",
  whiteSpace: "nowrap",
};

function formatTime(timeStr) {
  if (!timeStr) return "-";
  // Backend returns time in UTC+7 format "YYYY-MM-DD HH:MM:SS"
  // Display as-is since it's already in Ho Chi Minh timezone
  return timeStr;
}

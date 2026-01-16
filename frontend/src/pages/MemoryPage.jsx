import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { getMemoryStats, getMemoryGenomes } from "../api/optimizer";
import { buildEquitySeries } from "../utils/equityTransform";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// Color palette for equity curves (same as Dashboard)
const COLORS = [
  "#22c55e", "#3b82f6", "#a855f7", "#f59e0b", "#ef4444",
  "#06b6d4", "#ec4899", "#84cc16", "#14b8a6", "#f97316",
];

export default function MemoryPage() {
  const [stats, setStats] = useState(null);
  const [genomes, setGenomes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedGenome, setSelectedGenome] = useState(null);
  const [sortBy] = useState("pf");  // Fixed to PF only
  const [overlayLimit, setOverlayLimit] = useState("5");
  const [visibleSeries, setVisibleSeries] = useState({});

  // Filters
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("30m");

  // Use ref to track previous genomes for rank comparison (avoid infinite loop)
  const prevGenomesRef = useRef([]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsRes, genomesRes] = await Promise.all([
        getMemoryStats(),
        getMemoryGenomes(symbol, timeframe, 100),
      ]);

      if (statsRes.success) {
        setStats(statsRes);
      }
      if (genomesRes.success) {
        const newGenomes = genomesRes.genomes || [];

        // Sort new genomes by PF to get their new ranks
        const sortedNew = [...newGenomes].sort((a, b) => (b.pf || 0) - (a.pf || 0));

        // Build a map of genome_hash -> previous rank from saved ref
        const prevRankMap = {};
        const sortedOld = [...prevGenomesRef.current].sort((a, b) => (b.pf || 0) - (a.pf || 0));
        sortedOld.forEach((g, idx) => {
          if (g.genome_hash) {
            prevRankMap[g.genome_hash] = idx + 1;
          }
        });

        // Add rank info to genomes
        const genomesWithRank = sortedNew.map((g, idx) => {
          const newRank = idx + 1;
          const prevRank = prevRankMap[g.genome_hash];
          return {
            ...g,
            currentRank: newRank,
            previousRank: prevRank || null,
            isNew: !prevRank,
          };
        });

        // Save current genomes to ref for next comparison
        prevGenomesRef.current = genomesWithRank;

        setGenomes(genomesWithRank);

        // Initialize visible series for top genomes
        const vis = {};
        genomesWithRank.slice(0, 5).forEach((_, idx) => {
          vis[`s${idx + 1}`] = true;
        });
        setVisibleSeries(vis);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Sort genomes by PF (Profit Factor) - descending
  const sortedGenomes = useMemo(() => {
    const arr = [...genomes];
    arr.sort((a, b) => (b.pf || 0) - (a.pf || 0));
    return arr.map((g, idx) => ({ ...g, displayRank: idx + 1 }));
  }, [genomes]);

  // Filter genomes for chart based on overlay limit
  const chartGenomes = useMemo(() => {
    if (overlayLimit === "all") return sortedGenomes;
    const n = parseInt(overlayLimit, 10);
    return sortedGenomes.slice(0, n);
  }, [sortedGenomes, overlayLimit]);

  // Check if we have real equity data
  const hasRealEquityData = useMemo(() => {
    return chartGenomes.some((g) => g.equityCurve && g.equityCurve.length > 0);
  }, [chartGenomes]);

  // Build equity data using buildEquitySeries (same as Dashboard)
  const equityData = useMemo(() => {
    if (!chartGenomes.length) return [];

    // Transform genomes to match buildEquitySeries expected format
    const resultsForChart = chartGenomes.map((g, idx) => ({
      strategyId: `s${idx + 1}`,
      equityCurve: g.equityCurve || [],
      timeframe: timeframe,
      meta: { timeframe: timeframe },
    }));

    // Use buildEquitySeries for consistent chart rendering
    const series = buildEquitySeries(resultsForChart, "pct");

    // If no real data, return empty (don't create fake data)
    if (!series || series.length === 0) {
      return [];
    }

    return series;
  }, [chartGenomes, timeframe]);

  // Calculate Y domain for chart
  const { yDomain, tickFmt } = useMemo(() => {
    const visibleKeys = Object.entries(visibleSeries)
      .filter(([, v]) => v)
      .map(([k]) => k);

    let min = Infinity;
    let max = -Infinity;

    equityData.forEach((p) => {
      visibleKeys.forEach((k) => {
        const v = Number(p[k]);
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      });
    });

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = -10;
      max = 10;
    }

    const pad = (max - min) * 0.1 || 5;
    return {
      yDomain: [min - pad, max + pad],
      tickFmt: (v) => `${Number(v).toFixed(1)}%`,
    };
  }, [equityData, visibleSeries]);

  // Rank change indicator
  const getRankChange = (genome) => {
    if (genome.isNew || genome.previousRank === null) {
      return { type: "new", display: "NEW", color: "#fbbf24" };
    }
    const change = genome.previousRank - genome.currentRank;
    if (change > 0) {
      return { type: "up", display: `▲${change}`, color: "#4ade80" };
    } else if (change < 0) {
      return { type: "down", display: `▼${Math.abs(change)}`, color: "#f87171" };
    }
    return { type: "same", display: "-", color: "#6b7280" };
  };

  const formatPct = (val) => {
    if (val === undefined || val === null) return "-";
    return `${val.toFixed(1)}%`;
  };

  const formatMoney = (val) => {
    if (val === undefined || val === null) return "-";
    return `$${Number(val).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  // Format date string to full format (DD/MM/YYYY)
  const formatDateShort = (dateStr) => {
    if (!dateStr) return "-";
    // Handle "YYYY-MM-DD" or "YYYY-MM-DD HH:mm:ss" format
    const parts = dateStr.split(" ")[0].split("-");
    if (parts.length >= 3) {
      return `${parts[2]}/${parts[1]}/${parts[0]}`;  // DD/MM/YYYY
    }
    return dateStr.slice(0, 10);
  };

  return (
    <div style={pageStyle}>
      <div style={{ maxWidth: 1800, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700 }}>
              Quant Brain Memory
            </h1>
            <p style={{ margin: "8px 0 0", opacity: 0.7, fontSize: 14 }}>
              Bảng xếp hạng genomes từ các lần tối ưu hóa (BRAIN mode)
            </p>
          </div>

          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            {/* Symbol filter */}
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              style={selectStyle}
            >
              {(stats?.symbols ? Object.keys(stats.symbols) : ["BTCUSDT"]).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>

            {/* Timeframe filter */}
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              style={selectStyle}
            >
              {(stats?.timeframes ? Object.keys(stats.timeframes) : ["30m"]).map((tf) => (
                <option key={tf} value={tf}>{tf}</option>
              ))}
            </select>

            <button onClick={fetchData} style={refreshBtn} disabled={loading}>
              {loading ? "Loading..." : "Refresh"}
            </button>
          </div>
        </div>

        {/* Stats bar */}
        {stats && (
          <div style={statsBar}>
            <div style={statItem}>
              <span style={{ opacity: 0.7 }}>Total Genomes:</span>
              <span style={{ fontWeight: 700, color: "#22c55e" }}>{stats.total_genomes}</span>
            </div>
            <div style={statItem}>
              <span style={{ opacity: 0.7 }}>Symbols:</span>
              <span>{Object.keys(stats.symbols || {}).join(", ") || "-"}</span>
            </div>
            <div style={statItem}>
              <span style={{ opacity: 0.7 }}>Timeframes:</span>
              <span>{Object.keys(stats.timeframes || {}).join(", ") || "-"}</span>
            </div>
          </div>
        )}

        {error && (
          <div style={{ color: "#ef4444", marginBottom: 16, padding: 12, background: "rgba(239,68,68,0.1)", borderRadius: 8 }}>
            {error}
          </div>
        )}

        {/* Main content */}
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 20 }}>
          {/* Left: Equity Chart */}
          <div style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16 }}>Equity / PnL % (overlay)</h2>
              <div style={{ display: "flex", gap: 8 }}>
                {["5", "10", "20", "50", "all"].map((n) => (
                  <button
                    key={n}
                    onClick={() => {
                      setOverlayLimit(n);
                      // Reset visibility
                      const vis = {};
                      const limit = n === "all" ? sortedGenomes.length : parseInt(n, 10);
                      sortedGenomes.slice(0, limit).forEach((_, idx) => {
                        vis[`s${idx + 1}`] = true;
                      });
                      setVisibleSeries(vis);
                    }}
                    style={{
                      ...chipBtn,
                      background: overlayLimit === n ? "rgba(34,197,94,0.2)" : chipBtn.background,
                      color: overlayLimit === n ? "#22c55e" : "#e5e7eb",
                    }}
                  >
                    {n === "all" ? "All" : `Top ${n}`}
                  </button>
                ))}
              </div>
            </div>

            <div style={chartContainer}>
              {equityData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equityData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <XAxis
                      dataKey="time"
                      tick={{ fontSize: 10, fill: "#64748b" }}
                      tickFormatter={(v) => {
                        if (!v) return "";
                        // Show only date part for cleaner display
                        return v.slice(0, 10);
                      }}
                      interval="preserveStartEnd"
                      minTickGap={50}
                    />
                    <YAxis
                      domain={yDomain}
                      tickFormatter={tickFmt}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 11, fill: "#64748b" }}
                      width={60}
                    />
                    <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                    <Tooltip
                      content={({ payload, label }) => {
                        if (!payload || !payload.length) return null;
                        const items = payload
                          .filter((p) => p && p.value !== undefined)
                          .map((p) => {
                            const idx = parseInt(p.dataKey.replace("s", ""), 10) - 1;
                            const genome = chartGenomes[idx];
                            return {
                              id: p.dataKey,
                              value: p.value,
                              color: p.stroke,
                              genome,
                              rank: idx + 1,
                            };
                          })
                          .sort((a, b) => b.value - a.value);

                        return (
                          <div style={tooltipStyle}>
                            <div style={{ fontSize: 11, marginBottom: 6, color: "#94a3b8" }}>{label}</div>
                            {items.slice(0, 10).map((item) => (
                              <div key={item.id} style={{ display: "flex", gap: 8, fontSize: 12, alignItems: "center", marginBottom: 2 }}>
                                <span style={{ width: 8, height: 8, borderRadius: "50%", background: item.color }} />
                                <span style={{ color: "#94a3b8" }}>#{item.rank}</span>
                                <span style={{ color: item.value >= 0 ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                                  {item.value >= 0 ? "+" : ""}{item.value.toFixed(2)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    {chartGenomes.map((genome, idx) => {
                      const key = `s${idx + 1}`;
                      const visible = visibleSeries[key];
                      const isTop1 = idx === 0;
                      return (
                        <Line
                          key={key}
                          type="monotone"
                          dataKey={key}
                          stroke={COLORS[idx % COLORS.length]}
                          strokeWidth={isTop1 ? 2.5 : 1.5}
                          opacity={visible ? (isTop1 ? 1 : 0.7) : 0.1}
                          dot={false}
                          connectNulls={false}
                          hide={!visible}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.5 }}>
                  {loading ? "Loading chart data..." : (
                    hasRealEquityData ? "Processing equity data..." : (
                      <div style={{ textAlign: "center" }}>
                        <div>Chưa có equity data</div>
                        <div style={{ fontSize: 12, marginTop: 8 }}>
                          Chạy Quant Brain ở <span style={{ color: "#22c55e" }}>BRAIN mode</span> để lưu equity curves
                        </div>
                      </div>
                    )
                  )}
                </div>
              )}
            </div>

            {/* Legend */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 12 }}>
              {chartGenomes.slice(0, 10).map((genome, idx) => {
                const key = `s${idx + 1}`;
                const visible = visibleSeries[key];
                return (
                  <button
                    key={key}
                    onClick={() => setVisibleSeries((v) => ({ ...v, [key]: !v[key] }))}
                    style={{
                      ...legendBtn,
                      opacity: visible ? 1 : 0.4,
                      borderColor: COLORS[idx % COLORS.length],
                    }}
                  >
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS[idx % COLORS.length] }} />
                    <span>#{idx + 1} {genome.id}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right: Leaderboard */}
          <div style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16 }}>Bảng xếp hạng Genomes</h2>
              <span style={{ fontSize: 12, opacity: 0.6 }}>Sorted by PF</span>
            </div>

            {/* Table */}
            <div style={tableContainer}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "rgba(255,255,255,0.05)" }}>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>▲▼</th>
                    <th style={thStyle}>ID</th>
                    <th style={thStyle}>Total PNL</th>
                    <th style={thStyle}>PF</th>
                    <th style={thStyle}>WR</th>
                    <th style={thStyle}>DD</th>
                    <th style={thStyle}>Trades</th>
                    <th style={thStyle}>Score</th>
                    <th style={thStyle}>Range</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedGenomes.map((g, idx) => {
                    const rankChange = getRankChange(g);
                    const isSelected = selectedGenome?.genome_hash === g.genome_hash;
                    const isTop1 = idx === 0;

                    return (
                      <tr
                        key={g.genome_hash || idx}
                        onClick={() => setSelectedGenome(isSelected ? null : g)}
                        style={{
                          cursor: "pointer",
                          background: isSelected
                            ? "rgba(59,130,246,0.2)"
                            : isTop1
                            ? "rgba(255,215,0,0.08)"
                            : idx % 2 === 0
                            ? "transparent"
                            : "rgba(255,255,255,0.02)",
                          borderBottom: "1px solid rgba(255,255,255,0.05)",
                          borderLeft: isTop1 ? "3px solid #fbbf24" : "3px solid transparent",
                        }}
                      >
                        <td style={tdStyle}>
                          {isTop1 && <span style={{ marginRight: 4 }}>⭐</span>}
                          #{idx + 1}
                        </td>
                        <td style={{ ...tdStyle, color: rankChange.color, fontWeight: 600 }}>
                          {rankChange.type === "new" ? (
                            <span style={newBadge}>NEW</span>
                          ) : (
                            rankChange.display
                          )}
                        </td>
                        <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11 }}>{g.id}</td>
                        <td style={tdStyle}>
                          <div style={{ color: (g.netProfitPct || 0) >= 0 ? "#22c55e" : "#ef4444" }}>
                            {formatMoney(g.netProfit)}
                          </div>
                          <div style={{ fontSize: 10, opacity: 0.7 }}>
                            ({formatPct(g.netProfitPct)})
                          </div>
                        </td>
                        <td style={{ ...tdStyle, color: (g.pf || 0) >= 1.5 ? "#22c55e" : "#e5e7eb" }}>
                          {g.pf?.toFixed(2) || "-"}
                        </td>
                        <td style={tdStyle}>{formatPct(g.winrate)}</td>
                        <td style={{ ...tdStyle, color: "#f87171" }}>{formatPct(g.maxDD)}</td>
                        <td style={tdStyle}>{g.totalTrades || "-"}</td>
                        <td style={{ ...tdStyle, fontWeight: 600, color: "#22c55e" }}>
                          {g.score?.toFixed(2) || "-"}
                        </td>
                        <td style={{ ...tdStyle, fontSize: 10, opacity: 0.8 }}>
                          {g.backtest_start && g.backtest_end ? (
                            <div>
                              <div>{formatDateShort(g.backtest_start)}</div>
                              <div style={{ opacity: 0.6 }}>→ {formatDateShort(g.backtest_end)}</div>
                            </div>
                          ) : "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {sortedGenomes.length === 0 && !loading && (
                <div style={{ padding: 32, textAlign: "center", opacity: 0.6 }}>
                  Chưa có genome nào được lưu.
                  <br />
                  <span style={{ fontSize: 12 }}>
                    Chạy Quant Brain ở <span style={{ color: "#22c55e" }}>BRAIN mode</span> để bắt đầu!
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Genome Detail Modal */}
        {selectedGenome && (
          <div style={modalOverlay} onClick={() => setSelectedGenome(null)}>
            <div style={modalContent} onClick={(e) => e.stopPropagation()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <h3 style={{ margin: 0 }}>
                  Genome #{selectedGenome.id} Details
                </h3>
                <button onClick={() => setSelectedGenome(null)} style={closeBtn}>✕</button>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                {/* Performance Metrics */}
                <div>
                  <h4 style={sectionTitle}>Performance Metrics</h4>
                  <div style={detailGrid}>
                    <DetailRow label="Symbol" value={symbol} />
                    <DetailRow label="Timeframe" value={timeframe} />
                    <DetailRow label="Total PnL" value={formatMoney(selectedGenome.netProfit)} color="#22c55e" />
                    <DetailRow label="PnL %" value={formatPct(selectedGenome.netProfitPct)} color="#22c55e" />
                    <DetailRow label="Profit Factor" value={selectedGenome.pf?.toFixed(2)} />
                    <DetailRow label="Win Rate" value={formatPct(selectedGenome.winrate)} />
                    <DetailRow label="Max Drawdown" value={formatPct(selectedGenome.maxDD)} color="#f87171" />
                    <DetailRow label="Total Trades" value={selectedGenome.totalTrades} />
                    <DetailRow label="Score" value={selectedGenome.score?.toFixed(4)} color="#22c55e" />
                    <DetailRow label="Robustness" value={selectedGenome.robustness?.toFixed(2)} />
                  </div>
                </div>

                {/* Rank & Period */}
                <div>
                  <h4 style={sectionTitle}>Rank Info</h4>
                  <div style={detailGrid}>
                    <DetailRow label="Current Rank" value={`#${selectedGenome.currentRank || "-"}`} />
                    <DetailRow
                      label="Previous Rank"
                      value={selectedGenome.previousRank ? `#${selectedGenome.previousRank}` : "NEW"}
                    />
                    <DetailRow label="Created" value={selectedGenome.timestampStr || "-"} />
                    <DetailRow label="Test Count" value={selectedGenome.test_count || 1} />
                  </div>

                  <h4 style={{ ...sectionTitle, marginTop: 16 }}>Backtest Period</h4>
                  <div style={detailGrid}>
                    <DetailRow label="Start" value={selectedGenome.backtest_start || "-"} />
                    <DetailRow label="End" value={selectedGenome.backtest_end || "-"} />
                  </div>
                </div>
              </div>

              {/* Strategy Parameters */}
              <div style={{ marginTop: 20 }}>
                <h4 style={sectionTitle}>Strategy Parameters</h4>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                  {selectedGenome.genome && Object.entries(selectedGenome.genome).map(([block, params]) => (
                    <div key={block} style={paramBlock}>
                      <div style={{ fontWeight: 600, color: "#60a5fa", marginBottom: 8, fontSize: 13 }}>
                        {block}
                      </div>
                      {typeof params === "object" && params !== null ? (
                        Object.entries(params).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                            <span style={{ opacity: 0.7 }}>{k}:</span>
                            <span style={{ color: "#22c55e" }}>
                              {typeof v === "number" ? v.toFixed(2) : String(v)}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div style={{ fontSize: 11 }}>{String(params)}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Market Profile */}
              {selectedGenome.market_profile && Object.keys(selectedGenome.market_profile).length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <h4 style={sectionTitle}>Market Profile (when saved)</h4>
                  <div style={{ ...detailGrid, display: "flex", gap: 20, flexWrap: "wrap" }}>
                    {Object.entries(selectedGenome.market_profile).map(([k, v]) => (
                      <div key={k} style={{ minWidth: 100 }}>
                        <span style={{ opacity: 0.7, fontSize: 11 }}>{k}: </span>
                        <span style={{ fontWeight: 600, color: "#60a5fa" }}>
                          {typeof v === "number" ? v.toFixed(4) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Helper component
function DetailRow({ label, value, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
      <span style={{ opacity: 0.7 }}>{label}</span>
      <span style={{ fontWeight: 600, color: color || "#e5e7eb" }}>{value || "-"}</span>
    </div>
  );
}

// Styles
const pageStyle = {
  minHeight: "100vh",
  width: "100%",
  padding: "32px 24px 60px",
  background: "radial-gradient(1200px 700px at 20% 0%, rgba(59,130,246,0.25), rgba(2,6,23,0.95)), linear-gradient(160deg, #0b1224 0%, #0f172a 45%, #0a0f1f 100%)",
  color: "#e5e7eb",
};

const panel = {
  background: "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))",
  borderRadius: 16,
  padding: 20,
  border: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
};

const statsBar = {
  display: "flex",
  gap: 24,
  padding: "12px 20px",
  background: "rgba(255,255,255,0.03)",
  borderRadius: 12,
  marginBottom: 20,
  border: "1px solid rgba(255,255,255,0.06)",
};

const statItem = {
  display: "flex",
  gap: 8,
  fontSize: 13,
};

const selectStyle = {
  padding: "10px 16px",
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.12)",
  background: "rgba(255,255,255,0.06)",
  color: "#e5e7eb",
  cursor: "pointer",
  minWidth: 120,
};

const refreshBtn = {
  padding: "10px 20px",
  borderRadius: 8,
  border: "1px solid rgba(34,197,94,0.3)",
  background: "rgba(34,197,94,0.1)",
  color: "#22c55e",
  cursor: "pointer",
  fontWeight: 600,
};

const chipBtn = {
  padding: "6px 12px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.1)",
  color: "#e5e7eb",
  cursor: "pointer",
  fontSize: 12,
};

const chartContainer = {
  height: 400,
  borderRadius: 12,
  background: "rgba(0,0,0,0.2)",
  padding: 12,
};

const legendBtn = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "6px 10px",
  borderRadius: 6,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.1)",
  color: "#e5e7eb",
  cursor: "pointer",
  fontSize: 11,
};

const tableContainer = {
  maxHeight: 500,
  overflowY: "auto",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 8,
};

const thStyle = {
  padding: "10px 8px",
  textAlign: "left",
  fontWeight: 600,
  borderBottom: "1px solid rgba(255,255,255,0.1)",
  position: "sticky",
  top: 0,
  background: "#0f172a",
};

const tdStyle = {
  padding: "10px 8px",
  textAlign: "left",
};

const newBadge = {
  background: "linear-gradient(135deg, #fbbf24, #f59e0b)",
  color: "#000",
  padding: "2px 6px",
  borderRadius: 4,
  fontSize: 10,
  fontWeight: 700,
};

const tooltipStyle = {
  background: "#0b1021",
  border: "1px solid rgba(255,255,255,0.15)",
  borderRadius: 8,
  padding: 10,
  color: "#e5e7eb",
};

const modalOverlay = {
  position: "fixed",
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: "rgba(0,0,0,0.8)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
};

const modalContent = {
  background: "linear-gradient(180deg, #1a1a2e, #0f0f1a)",
  borderRadius: 16,
  padding: 24,
  maxWidth: 900,
  width: "90%",
  maxHeight: "85vh",
  overflowY: "auto",
  border: "1px solid rgba(255,255,255,0.1)",
  boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
};

const closeBtn = {
  background: "rgba(255,255,255,0.1)",
  border: "none",
  color: "#e5e7eb",
  width: 32,
  height: 32,
  borderRadius: "50%",
  cursor: "pointer",
  fontSize: 16,
};

const sectionTitle = {
  margin: "0 0 12px",
  fontSize: 14,
  fontWeight: 600,
  color: "#94a3b8",
};

const detailGrid = {
  background: "rgba(255,255,255,0.03)",
  borderRadius: 8,
  padding: 12,
};

const paramBlock = {
  background: "rgba(255,255,255,0.03)",
  borderRadius: 8,
  padding: 12,
  border: "1px solid rgba(255,255,255,0.05)",
};

import { useEffect, useState, useCallback } from "react";
import { getMemoryStats, getMemoryGenomes } from "../api/optimizer";

const panel = {
  background: "rgba(255,255,255,0.03)",
  borderRadius: 16,
  padding: 16,
  border: "1px solid rgba(255,255,255,0.08)",
};

const legendChip = {
  fontSize: 11,
  padding: "4px 8px",
  borderRadius: 6,
  background: "rgba(255,255,255,0.08)",
  border: "none",
  color: "#d1d5db",
  cursor: "pointer",
};

export default function MemoryPanel({ symbol = "BTCUSDT", timeframe = "30m" }) {
  const [stats, setStats] = useState(null);
  const [genomes, setGenomes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const [selectedGenome, setSelectedGenome] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsRes, genomesRes] = await Promise.all([
        getMemoryStats(),
        getMemoryGenomes(symbol, timeframe, 10),
      ]);

      if (statsRes.success) {
        setStats(statsRes);
      }
      if (genomesRes.success) {
        setGenomes(genomesRes.genomes || []);
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

  const formatScore = (score) => {
    if (!score) return "-";
    return score.toFixed(2);
  };

  const formatPct = (val) => {
    if (val === undefined || val === null) return "-";
    return `${val.toFixed(1)}%`;
  };

  return (
    <div style={{ ...panel, marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer"
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <h3 style={{ margin: 0, fontSize: 14, letterSpacing: 0.3, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 16 }}>🧠</span>
          Memory
          {stats && (
            <span style={{
              fontSize: 11,
              padding: "2px 8px",
              borderRadius: 10,
              background: stats.total_genomes > 0 ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.1)",
              color: stats.total_genomes > 0 ? "#22c55e" : "#9ca3af"
            }}>
              {stats.total_genomes} genomes
            </span>
          )}
        </h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button
            onClick={(e) => { e.stopPropagation(); fetchData(); }}
            style={legendChip}
            disabled={loading}
          >
            {loading ? "..." : "Refresh"}
          </button>
          <span style={{ fontSize: 12, opacity: 0.6 }}>{expanded ? "▼" : "▶"}</span>
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: 12 }}>
          {error && (
            <p style={{ color: "#ef4444", fontSize: 12 }}>{error}</p>
          )}

          {stats && stats.total_genomes === 0 && (
            <p style={{ fontSize: 12, opacity: 0.6, textAlign: "center", padding: 16 }}>
              Chưa có genome nào được lưu.<br/>
              Chạy Quant Brain ở BRAIN mode để bắt đầu học!
            </p>
          )}

          {genomes.length > 0 && (
            <>
              <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 8 }}>
                Top genomes cho {symbol} {timeframe}:
              </div>

              <div style={{
                maxHeight: 300,
                overflowY: "auto",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 8
              }}>
                <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: "rgba(255,255,255,0.05)" }}>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>Score</th>
                      <th style={thStyle}>PF</th>
                      <th style={thStyle}>WR</th>
                      <th style={thStyle}>DD</th>
                      <th style={thStyle}>Profit</th>
                      <th style={thStyle}>Trades</th>
                      <th style={thStyle}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {genomes.map((g, idx) => (
                      <tr
                        key={g.genome_hash || idx}
                        onClick={() => setSelectedGenome(selectedGenome?.id === g.id ? null : g)}
                        style={{
                          cursor: "pointer",
                          background: selectedGenome?.id === g.id
                            ? "rgba(59,130,246,0.2)"
                            : idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.02)",
                          borderBottom: "1px solid rgba(255,255,255,0.05)"
                        }}
                      >
                        <td style={tdStyle}>{idx + 1}</td>
                        <td style={{ ...tdStyle, color: "#22c55e", fontWeight: 600 }}>
                          {formatScore(g.score)}
                        </td>
                        <td style={tdStyle}>{g.pf?.toFixed(2) || "-"}</td>
                        <td style={tdStyle}>{formatPct(g.winrate)}</td>
                        <td style={{ ...tdStyle, color: "#ef4444" }}>{formatPct(g.maxDD)}</td>
                        <td style={{
                          ...tdStyle,
                          color: g.netProfitPct > 0 ? "#22c55e" : "#ef4444"
                        }}>
                          {formatPct(g.netProfitPct)}
                        </td>
                        <td style={tdStyle}>{g.totalTrades || "-"}</td>
                        <td style={{ ...tdStyle, opacity: 0.6 }}>{g.timestampStr || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Selected Genome Details */}
              {selectedGenome && (
                <div style={{
                  marginTop: 12,
                  padding: 12,
                  background: "rgba(59,130,246,0.1)",
                  borderRadius: 8,
                  border: "1px solid rgba(59,130,246,0.3)"
                }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
                    Genome #{selectedGenome.id} Parameters:
                  </div>
                  <div style={{ fontSize: 10, opacity: 0.8, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {selectedGenome.genome && Object.entries(selectedGenome.genome).map(([block, params]) => (
                      <div key={block}>
                        <div style={{ fontWeight: 600, color: "#60a5fa", marginBottom: 4 }}>{block}:</div>
                        {typeof params === "object" && params !== null ? (
                          Object.entries(params).map(([k, v]) => (
                            <div key={k} style={{ paddingLeft: 8 }}>
                              {k}: <span style={{ color: "#22c55e" }}>{typeof v === "number" ? v.toFixed(2) : String(v)}</span>
                            </div>
                          ))
                        ) : (
                          <div style={{ paddingLeft: 8 }}>{String(params)}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* Stats breakdown */}
          {stats && stats.total_genomes > 0 && (
            <div style={{ marginTop: 12, fontSize: 10, opacity: 0.6 }}>
              <div>Symbols: {Object.entries(stats.symbols || {}).map(([s, c]) => `${s}(${c})`).join(", ") || "-"}</div>
              <div>Timeframes: {Object.entries(stats.timeframes || {}).map(([t, c]) => `${t}(${c})`).join(", ") || "-"}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: "8px 6px",
  textAlign: "left",
  fontWeight: 600,
  borderBottom: "1px solid rgba(255,255,255,0.1)"
};

const tdStyle = {
  padding: "6px",
  textAlign: "left"
};

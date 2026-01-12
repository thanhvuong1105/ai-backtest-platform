import { useState, useRef, useEffect } from "react";
import Header from "../components/Header";
import Section from "../components/Section";
import DebugChart from "../components/DebugChart";
import DebugTradeTable from "../components/DebugTradeTable";

// API Base URL - use /api prefix for nginx proxy in Docker
const API_URL = import.meta.env.VITE_API_URL || "/api";

// Strategy schemas
const STRATEGY_SCHEMAS = {
  ema_cross: {
    name: "EMA Cross",
    params: [
      { key: "emaFast", label: "EMA Fast", type: "int", default: 12, min: 1, max: 200 },
      { key: "emaSlow", label: "EMA Slow", type: "int", default: 26, min: 1, max: 500 },
    ],
  },
  rf_st_rsi: {
    name: "RF + ST + RSI Divergence",
    groups: [
      {
        label: "Entry Settings",
        params: [
          { key: "showDualFlip", label: "Enable Dual Flip Entry", type: "bool", default: true },
          { key: "showRSI", label: "Enable RSI Divergence Entry", type: "bool", default: true },
          { key: "st_atrPeriod", label: "ST ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_src", label: "ST Source", type: "select", default: "hl2", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "st_mult", label: "ST ATR Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "st_useATR", label: "ST use ATR?", type: "bool", default: true },
          { key: "rf_src", label: "RF Source", type: "select", default: "close", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "rf_period", label: "RF Period", type: "int", default: 100, min: 1, max: 500 },
          { key: "rf_mult", label: "RF Multiplier", type: "float", default: 3.0, min: 0.1, max: 20, step: 0.1 },
          { key: "rsi_length", label: "RSI Length", type: "int", default: 14, min: 1, max: 100 },
          { key: "rsi_ma_length", label: "MA Length on RSI", type: "int", default: 6, min: 1, max: 50 },
        ],
      },
      {
        label: "Stop Loss",
        params: [
          { key: "sl_st_atrPeriod", label: "ST SL ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "sl_st_src", label: "ST SL Source", type: "select", default: "hl2", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "sl_st_mult", label: "ST SL Mult", type: "float", default: 4.0, min: 0.1, max: 20, step: 0.1 },
          { key: "sl_st_useATR", label: "ST SL use ATR?", type: "bool", default: true },
          { key: "sl_rf_period", label: "RF SL Period", type: "int", default: 100, min: 1, max: 500 },
          { key: "sl_rf_mult", label: "RF SL Multiplier", type: "float", default: 7.0, min: 0.1, max: 20, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - Dual Flip",
        params: [
          { key: "tp_dual_st_atrPeriod", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "tp_dual_st_mult", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "tp_dual_rr_mult", label: "TP R:R Mult", type: "float", default: 1.3, min: 0.1, max: 10, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - RSI",
        params: [
          { key: "tp_rsi_st_atrPeriod", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "tp_rsi_st_mult", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "tp_rsi_rr_mult", label: "TP R:R Mult", type: "float", default: 1.3, min: 0.1, max: 10, step: 0.1 },
        ],
      },
      {
        label: "Debug",
        params: [
          { key: "debug", label: "Enable Debug Logging", type: "bool", default: false },
        ],
      },
    ],
  },
};

// Get default params for a strategy
function getDefaultParams(strategyType) {
  const schema = STRATEGY_SCHEMAS[strategyType];
  if (!schema) return {};

  const params = {};

  if (schema.params) {
    schema.params.forEach((p) => {
      params[p.key] = p.default;
    });
  }

  if (schema.groups) {
    schema.groups.forEach((group) => {
      group.params.forEach((p) => {
        params[p.key] = p.default;
      });
    });
  }

  return params;
}

// Helper functions for date range
const todayIso = () => new Date().toISOString().slice(0, 10);
const presetFrom = (days) => {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
};

export default function DebugView() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [strategyType, setStrategyType] = useState("rf_st_rsi");
  const [strategyParams, setStrategyParams] = useState(getDefaultParams("rf_st_rsi"));
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [expandedGroups, setExpandedGroups] = useState({});
  const abortControllerRef = useRef(null);
  const progressIntervalRef = useRef(null);
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [baseCurrency, setBaseCurrency] = useState("USDT");
  const [orderSize, setOrderSize] = useState(100);
  const [pyramiding, setPyramiding] = useState(1);
  const [commission, setCommission] = useState(0);
  const [slippage, setSlippage] = useState(0);
  const [rangePreset, setRangePreset] = useState("ALL");
  const [rangeFrom, setRangeFrom] = useState("");
  const [rangeTo, setRangeTo] = useState("");

  const handleStrategyChange = (newType) => {
    setStrategyType(newType);
    setStrategyParams(getDefaultParams(newType));
    setChartData(null);
  };

  const handleParamChange = (key, value, type) => {
    let parsedValue = value;
    if (type === "int") parsedValue = parseInt(value, 10) || 0;
    else if (type === "float") parsedValue = parseFloat(value) || 0;
    else if (type === "bool") parsedValue = value === true || value === "true";

    setStrategyParams((prev) => ({ ...prev, [key]: parsedValue }));
  };

  const toggleGroup = (label) => {
    setExpandedGroups((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  const computeRange = () => {
    if (rangePreset === "ALL" || (!rangeFrom && !rangeTo)) {
      return null;
    }
    return {
      rangeType: rangePreset,
      from: rangeFrom || null,
      to: rangeTo || null,
    };
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const fetchChartData = async () => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setLoadingProgress(0);
    setError(null);

    // Start linear progress animation (0 to 100% over ~3 seconds)
    let progress = 0;
    const progressStep = 2; // 2% per tick
    const intervalMs = 60; // 60ms per tick = ~3 seconds total
    progressIntervalRef.current = setInterval(() => {
      progress += progressStep;
      if (progress >= 100) {
        progress = 100;
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      setLoadingProgress(progress);
    }, intervalMs);

    try {
      const config = {
        symbol,
        timeframe,
        strategy: {
          type: strategyType,
          params: strategyParams,
        },
        capital: { initial: initialCapital, orderPct: orderSize },
        risk: { pyramiding, commission, slippage },
        range: computeRange(),
      };

      const res = await fetch(`${API_URL}/chart-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
        signal,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Failed to fetch chart data");
      }

      const data = await res.json();
      setLoadingProgress(100);
      setChartData(data);
    } catch (err) {
      if (err.name === "AbortError") {
        setError("Request cancelled");
      } else {
        setError(err.message);
      }
    } finally {
      // Clear progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      setLoading(false);
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    setLoading(false);
    setLoadingProgress(0);
  };

  const schema = STRATEGY_SCHEMAS[strategyType];

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f1a", color: "#fff" }}>
      <Header />

      <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
        <h1 style={{ fontSize: "24px", marginBottom: "24px" }}>
          Debug View - Strategy Verification
        </h1>

        {/* Config Panel */}
        <Section title="Configuration">
          {/* Data settings - MOVED FIRST */}
          <div style={{ marginBottom: "20px", paddingBottom: "20px", borderBottom: "1px solid #2d2d44" }}>
            <h3 style={{ fontSize: "14px", color: "#9ca3af", marginBottom: "12px", textTransform: "uppercase", fontWeight: "600" }}>
              Data
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: "16px",
              }}
            >
              <div>
                <label style={labelStyle}>Symbol</label>
                <select
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  style={inputStyle}
                >
                  <option value="BTCUSDT">BTCUSDT</option>
                  <option value="ETHUSDT">ETHUSDT</option>
                </select>
              </div>

              <div>
                <label style={labelStyle}>Timeframe</label>
                <select
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                  style={inputStyle}
                >
                  <option value="5m">5m</option>
                  <option value="15m">15m</option>
                  <option value="30m">30m</option>
                  <option value="1h">1h</option>
                  <option value="4h">4h</option>
                  <option value="1d">1d</option>
                </select>
              </div>

              <div>
                <label style={labelStyle}>Strategy</label>
                <select
                  value={strategyType}
                  onChange={(e) => handleStrategyChange(e.target.value)}
                  style={inputStyle}
                >
                  <option value="ema_cross">EMA Cross</option>
                  <option value="rf_st_rsi">RF + ST + RSI Divergence</option>
                </select>
              </div>
            </div>
          </div>

          {/* Capital & Size settings - MOVED SECOND */}
          <div style={{ marginBottom: "20px", paddingBottom: "20px", borderBottom: "1px solid #2d2d44" }}>
            <h3 style={{ fontSize: "14px", color: "#9ca3af", marginBottom: "12px", textTransform: "uppercase", fontWeight: "600" }}>
              Capital & Size
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "16px",
              }}
            >
              <div>
                <label style={labelStyle}>Initial Capital</label>
                <input
                  type="number"
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(parseInt(e.target.value) || 0)}
                  style={inputStyle}
                  min="1000"
                  step="10000"
                />
              </div>
              <div>
                <label style={labelStyle}>Base Currency</label>
                <select
                  value={baseCurrency}
                  onChange={(e) => setBaseCurrency(e.target.value)}
                  style={inputStyle}
                >
                  <option value="USDT">USDT</option>
                  <option value="USD">USD</option>
                  <option value="EUR">EUR</option>
                  <option value="BUSD">BUSD</option>
                </select>
              </div>
              <div>
                <label style={labelStyle}>Default Order Size</label>
                <div style={{ display: "flex", gap: "8px" }}>
                  <input
                    type="number"
                    value={orderSize}
                    onChange={(e) => setOrderSize(Math.min(100, Math.max(1, parseInt(e.target.value) || 0)))}
                    style={{...inputStyle, flex: 1}}
                    min="1"
                    max="100"
                  />
                  <div style={{...inputStyle, background: "#2d2d44", display: "flex", alignItems: "center", justifyContent: "center", flexBasis: "70px"}}>
                    % of equity
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Execution / Risk */}
          <div style={{ marginBottom: "20px", paddingBottom: "20px", borderBottom: "1px solid #2d2d44" }}>
            <h3 style={{ fontSize: "14px", color: "#9ca3af", marginBottom: "12px", textTransform: "uppercase", fontWeight: "600" }}>
              Execution / Risk
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "16px",
              }}
            >
              <div>
                <label style={labelStyle}>Pyramiding</label>
                <input
                  type="number"
                  value={pyramiding}
                  onChange={(e) => setPyramiding(Math.max(1, parseInt(e.target.value) || 1))}
                  style={inputStyle}
                  min="1"
                  step="1"
                />
              </div>
              <div>
                <label style={labelStyle}>Commission (%)</label>
                <input
                  type="number"
                  value={commission}
                  onChange={(e) => setCommission(parseFloat(e.target.value) || 0)}
                  style={inputStyle}
                  step="0.01"
                  min="0"
                />
              </div>
              <div>
                <label style={labelStyle}>Slippage (ticks)</label>
                <input
                  type="number"
                  value={slippage}
                  onChange={(e) => setSlippage(parseFloat(e.target.value) || 0)}
                  style={inputStyle}
                  step="0.01"
                  min="0"
                />
              </div>
            </div>
          </div>

          {/* Range settings */}
          <div style={{ marginBottom: "20px", paddingBottom: "20px", borderBottom: "1px solid #2d2d44" }}>
            <h3 style={{ fontSize: "14px", color: "#9ca3af", marginBottom: "12px", textTransform: "uppercase", fontWeight: "600" }}>
              Range
            </h3>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
              {[
                { key: "7D", label: "7D" },
                { key: "30D", label: "30D" },
                { key: "90D", label: "90D" },
                { key: "365D", label: "365D" },
                { key: "2Y", label: "2Y" },
                { key: "3Y", label: "3Y" },
                { key: "ALL", label: "All" },
              ].map((opt) => (
                <button
                  key={opt.key}
                  onClick={() => {
                    setRangePreset(opt.key);
                    if (opt.key === "ALL") {
                      setRangeFrom("");
                      setRangeTo("");
                    } else if (opt.key === "7D") {
                      setRangeFrom(presetFrom(7));
                      setRangeTo(todayIso());
                    } else if (opt.key === "30D") {
                      setRangeFrom(presetFrom(30));
                      setRangeTo(todayIso());
                    } else if (opt.key === "90D") {
                      setRangeFrom(presetFrom(90));
                      setRangeTo(todayIso());
                    } else if (opt.key === "365D") {
                      setRangeFrom(presetFrom(365));
                      setRangeTo(todayIso());
                    } else if (opt.key === "2Y") {
                      setRangeFrom(presetFrom(730));
                      setRangeTo(todayIso());
                    } else if (opt.key === "3Y") {
                      setRangeFrom(presetFrom(1095));
                      setRangeTo(todayIso());
                    }
                  }}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "6px",
                    border: "1px solid #2d2d44",
                    background: rangePreset === opt.key ? "rgba(34,197,94,0.15)" : "#1a1a2e",
                    color: rangePreset === opt.key ? "#22c55e" : "#e5e7eb",
                    cursor: "pointer",
                    fontSize: "13px",
                  }}
                >
                  {opt.label}
                </button>
              ))}
              <input
                type="date"
                placeholder="dd/mm/yyyy"
                value={rangeFrom}
                onChange={(e) => {
                  setRangeFrom(e.target.value);
                  setRangePreset("CUSTOM");
                }}
                style={dateInputStyle}
              />
              <span style={{ color: "#9ca3af" }}>to</span>
              <input
                type="date"
                placeholder="dd/mm/yyyy"
                value={rangeTo}
                onChange={(e) => {
                  setRangeTo(e.target.value);
                  setRangePreset("CUSTOM");
                }}
                style={dateInputStyle}
              />
            </div>
          </div>

          {/* Strategy params - Simple (no groups) */}
          {schema?.params && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: "16px",
                marginBottom: "16px",
              }}
            >
              {schema.params.map((param) => (
                <ParamInput
                  key={param.key}
                  param={param}
                  value={strategyParams[param.key]}
                  onChange={handleParamChange}
                />
              ))}
            </div>
          )}

          {/* Strategy params - Grouped */}
          {schema?.groups && (
            <div style={{ marginBottom: "16px" }}>
              {schema.groups.map((group) => (
                <div
                  key={group.label}
                  style={{
                    marginBottom: "12px",
                    border: "1px solid #2d2d44",
                    borderRadius: "8px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    onClick={() => toggleGroup(group.label)}
                    style={{
                      background: "#1a1a2e",
                      padding: "12px 16px",
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontWeight: "600", fontSize: "14px" }}>
                      {group.label}
                    </span>
                    <span style={{ color: "#9ca3af" }}>
                      {expandedGroups[group.label] ? "▼" : "▶"}
                    </span>
                  </div>

                  {expandedGroups[group.label] && (
                    <div
                      style={{
                        padding: "16px",
                        background: "#0f0f1a",
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                        gap: "12px",
                      }}
                    >
                      {group.params.map((param) => (
                        <ParamInput
                          key={param.key}
                          param={param}
                          value={strategyParams[param.key]}
                          onChange={handleParamChange}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            <button
              onClick={fetchChartData}
              disabled={loading}
              style={{
                background: loading ? "#4b5563" : "linear-gradient(110deg, #8b5cf6, #6366f1)",
                color: "white",
                border: "none",
                padding: "12px 24px",
                borderRadius: "8px",
                cursor: loading ? "not-allowed" : "pointer",
                fontSize: "16px",
                fontWeight: "600",
                boxShadow: loading ? "none" : "0 4px 14px rgba(139,92,246,0.3)",
              }}
            >
              {loading ? "Running..." : "Run Backtest & Show Chart"}
            </button>

            {/* Loading progress bar and cancel button */}
            {loading && (
              <div style={{ display: "flex", alignItems: "center", gap: "12px", minWidth: "200px" }}>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
                  <div style={{ fontSize: "12px", color: "#9ca3af" }}>
                    {loadingProgress < 100 ? `Loading... ${Math.round(loadingProgress)}%` : "Complete!"}
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: "6px",
                      borderRadius: "999px",
                      background: "rgba(255,255,255,0.1)",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${loadingProgress}%`,
                        height: "100%",
                        background: "linear-gradient(90deg, #8b5cf6, #6366f1)",
                        transition: "width 0.2s ease",
                        borderRadius: "999px",
                      }}
                    />
                  </div>
                </div>
                <button
                  onClick={handleCancel}
                  style={{
                    padding: "8px 14px",
                    borderRadius: "6px",
                    border: "1px solid rgba(239,68,68,0.5)",
                    background: "rgba(239,68,68,0.15)",
                    color: "#f87171",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: "500",
                  }}
                >
                  Cancel
                </button>
              </div>
            )}
          </div>

          {error && (
            <div
              style={{
                marginTop: "16px",
                padding: "12px",
                background: "#ef444420",
                border: "1px solid #ef4444",
                borderRadius: "8px",
                color: "#ef4444",
              }}
            >
              {error}
            </div>
          )}
        </Section>

        {/* Chart */}
        {chartData && (
          <>
            <Section
              title={`${chartData.symbol} · ${chartData.timeframe.toUpperCase()} · ${schema?.name || strategyType}`}
            >
              <div style={{ marginBottom: "16px" }}>
                <div
                  style={{
                    display: "flex",
                    gap: "24px",
                    color: "#9ca3af",
                    fontSize: "14px",
                    flexWrap: "wrap",
                  }}
                >
                  {strategyType === "ema_cross" && (
                    <>
                      <span>
                        <span
                          style={{
                            display: "inline-block",
                            width: "12px",
                            height: "3px",
                            background: "#f59e0b",
                            marginRight: "8px",
                            verticalAlign: "middle",
                          }}
                        />
                        EMA Fast
                      </span>
                      <span>
                        <span
                          style={{
                            display: "inline-block",
                            width: "12px",
                            height: "3px",
                            background: "#3b82f6",
                            marginRight: "8px",
                            verticalAlign: "middle",
                          }}
                        />
                        EMA Slow
                      </span>
                    </>
                  )}
                  {strategyType === "rf_st_rsi" && (
                    <>
                      <span>
                        <span
                          style={{
                            display: "inline-block",
                            width: "12px",
                            height: "3px",
                            background: "#26A69A",
                            marginRight: "8px",
                            verticalAlign: "middle",
                          }}
                        />
                        Range Filter
                      </span>
                      <span>
                        <span
                          style={{
                            display: "inline-block",
                            width: "12px",
                            height: "3px",
                            background: "#f59e0b",
                            marginRight: "8px",
                            verticalAlign: "middle",
                          }}
                        />
                        SuperTrend
                      </span>
                    </>
                  )}
                  <span>
                    <span style={{ color: "#2962FF", marginRight: "4px" }}>
                      ▲
                    </span>
                    Long
                  </span>
                  <span>
                    <span style={{ color: "#26A69A", marginRight: "4px" }}>
                      ▼
                    </span>
                    TP
                  </span>
                  <span>
                    <span style={{ color: "#EF5350", marginRight: "4px" }}>
                      ▼
                    </span>
                    Exit/SL
                  </span>
                </div>
              </div>
              <DebugChart
                candles={chartData.candles}
                indicators={chartData.indicators}
                markers={chartData.markers}
              />
            </Section>

            {/* Summary - metrics from FILTERED trades */}
            {chartData.summary && (
              <Section title={`Backtest Summary ${rangePreset !== "ALL" ? `(${rangePreset})` : "(All Time)"}`}>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                    gap: "16px",
                  }}
                >
                  <StatCard
                    label="Net Profit"
                    value={`${chartData.summary.netProfitPct?.toFixed(2)}%`}
                    color={
                      chartData.summary.netProfitPct >= 0 ? "#22c55e" : "#ef4444"
                    }
                  />
                  <StatCard
                    label="Total Trades"
                    value={chartData.summary.totalTrades}
                  />
                  <StatCard
                    label="Win Rate"
                    value={`${chartData.summary.winrate?.toFixed(1)}%`}
                    color="#22c55e"
                  />
                  <StatCard
                    label="Profit Factor"
                    value={chartData.summary.profitFactor?.toFixed(2)}
                    color={
                      chartData.summary.profitFactor >= 1
                        ? "#22c55e"
                        : "#ef4444"
                    }
                  />
                  <StatCard
                    label="Max Drawdown"
                    value={`${chartData.summary.maxDrawdownPct?.toFixed(2)}%`}
                    color="#ef4444"
                  />
                  <StatCard
                    label="Final Equity"
                    value={`$${chartData.summary.finalEquity?.toFixed(2)}`}
                  />
                </div>
              </Section>
            )}

            {/* Trade List - FILTERED by range */}
            <Section title={`Trade List ${rangePreset !== "ALL" ? `(Filtered: ${rangePreset})` : "(All Time)"}`}>
              <div style={{
                marginBottom: "12px",
                padding: "10px 14px",
                background: "rgba(139,92,246,0.1)",
                border: "1px solid rgba(139,92,246,0.3)",
                borderRadius: "6px",
                fontSize: "13px",
                color: "#a78bfa"
              }}>
                <strong>Note:</strong> Chart shows ALL signals from history. Trade List below is filtered by your selected range.
                {rangePreset !== "ALL" && ` Showing trades from ${rangeFrom || "start"} to ${rangeTo || "now"}.`}
              </div>
              <DebugTradeTable trades={chartData.trades} />
            </Section>
          </>
        )}
      </div>
    </div>
  );
}

function ParamInput({ param, value, onChange }) {
  const { key, label, type, options, min, max, step } = param;

  if (type === "bool") {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <input
          type="checkbox"
          checked={value === true}
          onChange={(e) => onChange(key, e.target.checked, type)}
          style={{ width: "18px", height: "18px", cursor: "pointer" }}
        />
        <label style={{ color: "#d1d5db", fontSize: "13px" }}>{label}</label>
      </div>
    );
  }

  if (type === "select") {
    return (
      <div>
        <label style={labelStyle}>{label}</label>
        <select
          value={value}
          onChange={(e) => onChange(key, e.target.value, type)}
          style={inputStyle}
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  // int or float
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(key, e.target.value, type)}
        style={inputStyle}
        min={min}
        max={max}
        step={step || (type === "float" ? 0.1 : 1)}
      />
    </div>
  );
}

function StatCard({ label, value, color = "#d1d5db" }) {
  return (
    <div
      style={{
        background: "#1a1a2e",
        padding: "16px",
        borderRadius: "8px",
        textAlign: "center",
      }}
    >
      <div style={{ color: "#9ca3af", fontSize: "12px", marginBottom: "4px" }}>
        {label}
      </div>
      <div style={{ color, fontSize: "20px", fontWeight: "600" }}>{value}</div>
    </div>
  );
}

const labelStyle = {
  display: "block",
  color: "#9ca3af",
  fontSize: "12px",
  marginBottom: "4px",
};

const inputStyle = {
  width: "100%",
  padding: "10px 12px",
  background: "#1a1a2e",
  border: "1px solid #2d2d44",
  borderRadius: "6px",
  color: "#fff",
  fontSize: "14px",
};

const dateInputStyle = {
  padding: "6px 10px",
  background: "#1a1a2e",
  border: "1px solid #2d2d44",
  borderRadius: "6px",
  color: "#fff",
  fontSize: "13px",
};

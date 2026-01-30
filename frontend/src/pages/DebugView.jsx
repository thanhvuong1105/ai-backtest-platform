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
  rf_st_rsi_combined: {
    name: "RF + ST + RSI Combined (Long & Short)",
    groups: [
      {
        label: "Enable Settings",
        params: [
          { key: "enableLong", label: "Enable Long Trading", type: "bool", default: true },
          { key: "enableShort", label: "Enable Short Trading", type: "bool", default: true },
        ],
      },
      {
        label: "Entry Settings",
        params: [
          { key: "showEntryLong", label: "Enable Dual Flip Long", type: "bool", default: true },
          { key: "showEntryShort", label: "Enable Dual Flip Short", type: "bool", default: true },
          { key: "showEntryRSI_L", label: "Enable RSI Long", type: "bool", default: true },
          { key: "showEntryRSI_S", label: "Enable RSI Short", type: "bool", default: true },
          { key: "dualFlipBarsLong", label: "Dual Flip Bars (Long)", type: "int", default: 8, min: 1, max: 50 },
          { key: "dualFlipBarsShort", label: "Dual Flip Bars (Short)", type: "int", default: 12, min: 1, max: 50 },
          { key: "st_atrPeriod", label: "ST ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_src", label: "ST Source", type: "select", default: "hl2", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "st_mult", label: "ST ATR Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "st_changeATR", label: "ST use ATR?", type: "bool", default: true },
          { key: "rf_src_in", label: "RF Source", type: "select", default: "close", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "rf_period", label: "RF Period", type: "int", default: 100, min: 1, max: 500 },
          { key: "rf_mult", label: "RF Multiplier", type: "float", default: 3.0, min: 0.1, max: 20, step: 0.1 },
          { key: "lenRSI", label: "RSI Length", type: "int", default: 14, min: 1, max: 100 },
          { key: "lenMA", label: "MA Length on RSI", type: "int", default: 6, min: 1, max: 50 },
        ],
      },
      {
        label: "Stop Loss - Long",
        params: [
          { key: "st_sl_atrPeriod_L", label: "ST SL ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_sl_src_L", label: "ST SL Source", type: "select", default: "hl2", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "st_sl_mult_L", label: "ST SL Mult", type: "float", default: 3.0, min: 0.1, max: 20, step: 0.1 },
          { key: "st_sl_useATR_L", label: "ST SL use ATR?", type: "bool", default: true },
          { key: "rf_sl_period_L", label: "RF SL Period", type: "int", default: 20, min: 1, max: 500 },
          { key: "rf_sl_mult_L", label: "RF SL Multiplier", type: "float", default: 15.0, min: 0.1, max: 50, step: 0.1 },
        ],
      },
      {
        label: "Stop Loss - Short",
        params: [
          { key: "st_sl_atrPeriod_S", label: "ST SL ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_sl_src_S", label: "ST SL Source", type: "select", default: "hl2", options: ["close", "hl2", "hlc3", "ohlc4"] },
          { key: "st_sl_mult_S", label: "ST SL Mult", type: "float", default: 2.0, min: 0.1, max: 20, step: 0.1 },
          { key: "st_sl_useATR_S", label: "ST SL use ATR?", type: "bool", default: true },
          { key: "rf_sl_period_S", label: "RF SL Period", type: "int", default: 20, min: 1, max: 500 },
          { key: "rf_sl_mult_S", label: "RF SL Multiplier", type: "float", default: 3.0, min: 0.1, max: 50, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - Dual Flip Long",
        params: [
          { key: "st_tp_dual_period_L", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_tp_dual_mult_L", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "rr_mult_dual_L", label: "TP R:R Mult", type: "float", default: 4.0, min: 0.1, max: 20, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - Dual Flip Short",
        params: [
          { key: "st_tp_dual_period_S", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_tp_dual_mult_S", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "rr_mult_dual_S", label: "TP R:R Mult", type: "float", default: 0.75, min: 0.1, max: 20, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - RSI Long",
        params: [
          { key: "st_tp_rsi1_period_L", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_tp_rsi1_mult_L", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "rr_mult_rsi1_L", label: "TP R:R Mult", type: "float", default: 4.0, min: 0.1, max: 20, step: 0.1 },
        ],
      },
      {
        label: "Take Profit - RSI Short",
        params: [
          { key: "st_tp_rsi1_period_S", label: "ST TP ATR Period", type: "int", default: 10, min: 1, max: 100 },
          { key: "st_tp_rsi1_mult_S", label: "ST TP Mult", type: "float", default: 2.0, min: 0.1, max: 10, step: 0.1 },
          { key: "rr_mult_rsi1_S", label: "TP R:R Mult", type: "float", default: 0.75, min: 0.1, max: 20, step: 0.1 },
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

  // Legend visibility state - all hidden by default, user clicks to show
  const [visibility, setVisibility] = useState({
    rangeFilter: false,
    superTrend: false,
    rfSlShort: false,
    stSlShort: false,
    stTpRsiShort: false,
    long: false,
    short: false,
    tp: false,
    exitSl: false,
  });

  const toggleVisibility = (key) => {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

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
      // Use backendType if available (for strategies that map to same backend)
      const backendStrategyType = schema?.backendType || strategyType;

      const config = {
        symbol,
        timeframe,
        strategy: {
          type: backendStrategyType,
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

      <div style={{ padding: "24px", maxWidth: "1800px", margin: "0 auto" }}>
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
                  <option value="rf_st_rsi">RF + ST + RSI Divergence (Long Only)</option>
                  <option value="rf_st_rsi_combined">RF + ST + RSI Combined (Long & Short)</option>
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
              {/* Clickable Legend */}
              <div style={{ marginBottom: "16px" }}>
                <div
                  style={{
                    display: "flex",
                    gap: "16px",
                    color: "#9ca3af",
                    fontSize: "13px",
                    flexWrap: "wrap",
                    alignItems: "center",
                  }}
                >
                  <span style={{ color: "#6b7280", fontSize: "12px" }}>Click to toggle:</span>
                  {strategyType === "ema_cross" && (
                    <>
                      <LegendItem
                        color="#f59e0b"
                        label="EMA Fast"
                        type="line"
                        active={true}
                      />
                      <LegendItem
                        color="#3b82f6"
                        label="EMA Slow"
                        type="line"
                        active={true}
                      />
                    </>
                  )}
                  {(strategyType === "rf_st_rsi" || strategyType === "rf_st_rsi_combined") && (
                    <>
                      <LegendItem
                        color="#26A69A"
                        label="Range Filter"
                        type="line"
                        active={visibility.rangeFilter}
                        onClick={() => toggleVisibility("rangeFilter")}
                      />
                      <LegendItem
                        color="#f59e0b"
                        label="SuperTrend"
                        type="line"
                        active={visibility.superTrend}
                        onClick={() => toggleVisibility("superTrend")}
                      />
                    </>
                  )}
                  {strategyType === "rf_st_rsi_combined" && (
                    <>
                      <LegendItem
                        color="#f97316"
                        label="RF SL Short"
                        type="line"
                        active={visibility.rfSlShort}
                        onClick={() => toggleVisibility("rfSlShort")}
                      />
                      <LegendItem
                        color="#ea580c"
                        label="ST SL Short"
                        type="line"
                        active={visibility.stSlShort}
                        onClick={() => toggleVisibility("stSlShort")}
                      />
                      <LegendItem
                        color="#0ea5e9"
                        label="ST TP RSI S"
                        type="line"
                        active={visibility.stTpRsiShort}
                        onClick={() => toggleVisibility("stTpRsiShort")}
                      />
                    </>
                  )}
                  <LegendItem
                    color="#00d4ff"
                    label="Long"
                    type="arrow"
                    direction="up"
                    active={visibility.long}
                    onClick={() => toggleVisibility("long")}
                  />
                  {strategyType === "rf_st_rsi_combined" && (
                    <LegendItem
                      color="#ff00ff"
                      label="Short"
                      type="arrow"
                      direction="down"
                      active={visibility.short}
                      onClick={() => toggleVisibility("short")}
                    />
                  )}
                  <LegendItem
                    color="#ffd700"
                    label="TP"
                    type="arrow"
                    direction="down"
                    active={visibility.tp}
                    onClick={() => toggleVisibility("tp")}
                  />
                  <LegendItem
                    color="#dc143c"
                    label="Exit/SL"
                    type="arrow"
                    direction="down"
                    active={visibility.exitSl}
                    onClick={() => toggleVisibility("exitSl")}
                  />
                </div>
              </div>
              <DebugChart
                candles={chartData.candles}
                indicators={chartData.indicators}
                markers={chartData.markers}
                debugMarkers={chartData.debug?.markers}
                visibility={visibility}
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

            {/* Debug Events - RF/ST Flip Signals */}
            {chartData.debug?.events && chartData.debug.events.length > 0 && (
              <Section title="Debug Events - RF/ST Flip Signals">
                <DebugEventsTable events={chartData.debug.events} />
              </Section>
            )}
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

// Debug Events Table Component
function DebugEventsTable({ events }) {
  const [filterType, setFilterType] = useState("ALL");
  const [filterMarker, setFilterMarker] = useState("ALL");
  const [expanded, setExpanded] = useState(null);

  // Filter options
  const eventTypes = ["ALL", "RF_FLIP_BUY", "RF_FLIP_SELL", "ST_FLIP_BUY", "ST_FLIP_SELL", "DUALFLIP_LONG_SIGNAL", "DUALFLIP_SHORT_SIGNAL", "RSI_BULL_DIV", "RSI_BEAR_DIV"];
  const markerTypes = ["ALL", "BUY", "SELL"];

  // Apply filters
  const filteredEvents = events.filter(e => {
    if (filterType !== "ALL" && e.event_type !== filterType) return false;
    if (filterMarker !== "ALL") {
      if (filterMarker === "BUY" && e.marker_type !== "BUY") return false;
      if (filterMarker === "SELL" && e.marker_type !== "SELL") return false;
    }
    return true;
  });

  // Format time from timestamp
  const formatTime = (time) => {
    if (typeof time === "number") {
      const d = new Date(time);
      return d.toISOString().replace("T", " ").slice(0, 19);
    }
    return String(time);
  };

  // Get badge style for marker type
  const getMarkerBadge = (markerType, tag) => {
    if (!markerType) return null;

    const isBuy = markerType === "BUY";
    const baseStyle = {
      padding: "2px 8px",
      borderRadius: "4px",
      fontSize: "11px",
      fontWeight: "600",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px",
    };

    if (isBuy) {
      return (
        <span style={{
          ...baseStyle,
          background: "rgba(34, 197, 94, 0.2)",
          color: "#22c55e",
          border: "1px solid rgba(34, 197, 94, 0.4)",
        }}>
          <span style={{ fontSize: "10px" }}>▲</span> Buy {tag && `(${tag})`}
        </span>
      );
    } else {
      return (
        <span style={{
          ...baseStyle,
          background: "rgba(239, 68, 68, 0.2)",
          color: "#ef4444",
          border: "1px solid rgba(239, 68, 68, 0.4)",
        }}>
          <span style={{ fontSize: "10px" }}>▼</span> Sell {tag && `(${tag})`}
        </span>
      );
    }
  };

  // Get event type badge
  const getEventTypeBadge = (eventType, source) => {
    const colors = {
      "RF": { bg: "rgba(38, 166, 154, 0.2)", color: "#26A69A", border: "rgba(38, 166, 154, 0.4)" },
      "ST": { bg: "rgba(245, 158, 11, 0.2)", color: "#f59e0b", border: "rgba(245, 158, 11, 0.4)" },
      "DUALFLIP": { bg: "rgba(99, 102, 241, 0.2)", color: "#6366f1", border: "rgba(99, 102, 241, 0.4)" },
      "RSI": { bg: "rgba(168, 85, 247, 0.2)", color: "#a855f7", border: "rgba(168, 85, 247, 0.4)" },
    };

    const c = colors[source] || { bg: "rgba(156, 163, 175, 0.2)", color: "#9ca3af", border: "rgba(156, 163, 175, 0.4)" };

    return (
      <span style={{
        padding: "2px 6px",
        borderRadius: "4px",
        fontSize: "10px",
        fontWeight: "500",
        background: c.bg,
        color: c.color,
        border: `1px solid ${c.border}`,
      }}>
        {eventType}
      </span>
    );
  };

  return (
    <div>
      {/* Filters */}
      <div style={{
        display: "flex",
        gap: "16px",
        marginBottom: "16px",
        flexWrap: "wrap",
        alignItems: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <label style={{ color: "#9ca3af", fontSize: "13px" }}>Event Type:</label>
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            style={{
              padding: "6px 10px",
              background: "#1a1a2e",
              border: "1px solid #2d2d44",
              borderRadius: "6px",
              color: "#fff",
              fontSize: "13px",
            }}
          >
            {eventTypes.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <label style={{ color: "#9ca3af", fontSize: "13px" }}>Marker:</label>
          <div style={{ display: "flex", gap: "4px" }}>
            {markerTypes.map(m => (
              <button
                key={m}
                onClick={() => setFilterMarker(m)}
                style={{
                  padding: "6px 12px",
                  borderRadius: "6px",
                  border: "1px solid #2d2d44",
                  background: filterMarker === m
                    ? (m === "BUY" ? "rgba(34,197,94,0.15)" : m === "SELL" ? "rgba(239,68,68,0.15)" : "rgba(99,102,241,0.15)")
                    : "#1a1a2e",
                  color: filterMarker === m
                    ? (m === "BUY" ? "#22c55e" : m === "SELL" ? "#ef4444" : "#6366f1")
                    : "#e5e7eb",
                  cursor: "pointer",
                  fontSize: "12px",
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <span style={{ color: "#9ca3af", fontSize: "13px" }}>
          Showing {filteredEvents.length} of {events.length} events
        </span>
      </div>

      {/* Events Table */}
      <div style={{
        maxHeight: "500px",
        overflowY: "auto",
        border: "1px solid #2d2d44",
        borderRadius: "8px",
      }}>
        <table style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "13px",
        }}>
          <thead>
            <tr style={{ background: "#1a1a2e", position: "sticky", top: 0 }}>
              <th style={thStyle}>Bar</th>
              <th style={thStyle}>Time</th>
              <th style={thStyle}>Price</th>
              <th style={thStyle}>Marker</th>
              <th style={thStyle}>Event Type</th>
              <th style={thStyle}>Source</th>
              <th style={thStyle}>Details</th>
            </tr>
          </thead>
          <tbody>
            {filteredEvents.slice(0, 500).map((evt, idx) => (
              <tr
                key={idx}
                style={{
                  background: evt.marker_type ? (evt.marker_type === "BUY" ? "rgba(34,197,94,0.05)" : "rgba(239,68,68,0.05)") : "transparent",
                  borderBottom: "1px solid #2d2d44",
                  cursor: "pointer",
                }}
                onClick={() => setExpanded(expanded === idx ? null : idx)}
              >
                <td style={tdStyle}>{evt.bar_index}</td>
                <td style={tdStyle}>{formatTime(evt.time)}</td>
                <td style={tdStyle}>{evt.price?.toFixed(2)}</td>
                <td style={tdStyle}>{getMarkerBadge(evt.marker_type, evt.tag)}</td>
                <td style={tdStyle}>{getEventTypeBadge(evt.event_type, evt.source)}</td>
                <td style={tdStyle}>{evt.source}</td>
                <td style={tdStyle}>
                  {expanded === idx ? (
                    <div style={{
                      background: "#0f0f1a",
                      padding: "8px",
                      borderRadius: "4px",
                      fontSize: "11px",
                      maxWidth: "400px",
                    }}>
                      <pre style={{ margin: 0, whiteSpace: "pre-wrap", color: "#9ca3af" }}>
                        {JSON.stringify(evt.snapshot, null, 2)}
                      </pre>
                    </div>
                  ) : (
                    <span style={{ color: "#6366f1" }}>▶ Click to expand</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredEvents.length > 500 && (
        <div style={{
          marginTop: "8px",
          padding: "8px",
          background: "rgba(245,158,11,0.1)",
          border: "1px solid rgba(245,158,11,0.3)",
          borderRadius: "6px",
          fontSize: "12px",
          color: "#f59e0b",
        }}>
          Showing first 500 events. Use filters to narrow down results.
        </div>
      )}
    </div>
  );
}

const thStyle = {
  padding: "12px 8px",
  textAlign: "left",
  color: "#9ca3af",
  fontWeight: "500",
  borderBottom: "1px solid #2d2d44",
};

const tdStyle = {
  padding: "10px 8px",
  color: "#e5e7eb",
  verticalAlign: "top",
};

// Legend Item Component - clickable to toggle visibility
function LegendItem({ color, label, type = "line", direction = "up", active = true, onClick }) {
  const isClickable = !!onClick;

  return (
    <span
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "6px",
        cursor: isClickable ? "pointer" : "default",
        background: active ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.2)",
        border: `1px solid ${active ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.05)"}`,
        opacity: active ? 1 : 0.4,
        transition: "all 0.15s ease",
        userSelect: "none",
      }}
    >
      {type === "line" ? (
        <span
          style={{
            display: "inline-block",
            width: "14px",
            height: "3px",
            background: active ? color : "#666",
            borderRadius: "2px",
          }}
        />
      ) : (
        <span
          style={{
            color: active ? color : "#666",
            fontSize: "10px",
            lineHeight: 1,
          }}
        >
          {direction === "up" ? "▲" : "▼"}
        </span>
      )}
      <span style={{ color: active ? "#e5e7eb" : "#666" }}>{label}</span>
    </span>
  );
}

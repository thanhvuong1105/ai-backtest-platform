import { useEffect, useMemo, useState } from "react";
import { runAiAgent, startAiAgent, getAiAgentProgress, getAiAgentResult, cancelAiAgent } from "../api/optimizer";

import Header from "../components/Header";
import StrategyTable from "../components/StrategyTable";
import StrategyTradesTable from "../components/StrategyTradesTable";
import TradeFilterBar from "../components/TradeFilterBar";
import Section from "../components/Section";
import StrategyToggle from "../components/StrategyToggle";
import { buildEquitySeries, computeEquityCurveFromTrades } from "../utils/equityTransform";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Dashboard() {
  const strategyConfigs = [
    {
      id: "ema",
      name: "EMA Cross Sweep",
      subtitle: "EMA crossover optimizer",
      type: "ema_cross",
      supported: true,
      defaults: {
        symbol: "BTCUSDT",
        timeframes: ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h"],
        emaFastRange: { start: 1, end: 20, step: 2 },
        emaSlowRange: { start: 10, end: 50, step: 5 },
        minTrades: 1,
        minPF: 0.5,
        maxDD: 100,
        properties: {
          initialCapital: 1000000,
          baseCurrency: "USDT",
          orderSize: { value: 1000, type: "fixed" }, // 1000 USDT cho 1 vị thế mặc định
          pyramiding: 2,
          commission: { value: 0, type: "percent" },
          slippage: 0,
        },
      },
    },
    {
      id: "rf_st_rsi",
      name: "RF + ST + RSI Divergence",
      subtitle: "Range Filter + SuperTrend + RSI Divergence",
      type: "rf_st_rsi",
      supported: true,
      defaults: {
        symbol: "BTCUSDT",
        timeframes: ["1h", "4h", "1D"],
        st_atrPeriod: [8, 10, 12],
        st_mult: [1.8, 2.0, 2.2],
        rf_period: [80, 100, 120],
        rf_mult: [2.8, 3.0, 3.2],
        minTrades: 1,
        minPF: 0.5,
        maxDD: 50,
        properties: {
          initialCapital: 1000000,
          baseCurrency: "USDT",
          orderSize: { value: 100, type: "percent" },
          pyramiding: 1,
          commission: { value: 0.04, type: "percent" },
          slippage: 0.01,
        },
      },
    },
    {
      id: "smc",
      name: "SMC Minimal (BOS / OB / FVG)",
      subtitle: "SOLUSDT · 5m",
      type: "smc_minimal",
      supported: false, // chưa nối backend
      defaults: {
        symbol: "SOLUSDT",
        timeframes: ["5m"],
        bos: true,
        ob: true,
        fvg: true,
        rr: 2,
      },
    },
    {
      id: "bb",
      name: "Trend + Momentum + BB Breakout (EMA Exit)",
      subtitle: "BTCUSDT · 1h",
      type: "bb_breakout",
      supported: false,
      defaults: {
        symbol: "BTCUSDT",
        timeframes: ["1h"],
        bbLength: 20,
        bbDev: 2,
        emaExit: 50,
      },
    },
  ];

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [agentComment, setAgentComment] = useState("");

  const tfOptions = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1D"];
  const [selectedStrategyId, setSelectedStrategyId] = useState(strategyConfigs[0].id);
  const [strategyInputs, setStrategyInputs] = useState(() => {
    const init = {};
    strategyConfigs.forEach((s) => (init[s.id] = { ...s.defaults }));
    return init;
  });

  const todayIso = () => new Date().toISOString().slice(0, 10);
  const presetFrom = (days) => {
    const d = new Date();
    d.setDate(d.getDate() - days);
    return d.toISOString().slice(0, 10);
  };

  const [rangePreset, setRangePreset] = useState("ALL"); // 7D | 30D | 90D | ALL | CUSTOM
  const [rangeFrom, setRangeFrom] = useState("");
  const [rangeTo, setRangeTo] = useState("");
  const [scanning, setScanning] = useState(false);
  const [sortBy, setSortBy] = useState("score"); // score | pnl | dd | pf | wr
  const [overlayLimit, setOverlayLimit] = useState("5"); // "5" | "10" | "20" | "50" | "all"
  const [viewMode, setViewMode] = useState("pct"); // pct | equity (hiện cố định pct)
  const [pageSize] = useState(20);
  const [currentPageOverlay, setCurrentPageOverlay] = useState(1);
  const [currentPageBots, setCurrentPageBots] = useState(1);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState({ percent: 0, status: "idle" });
  const [canceling, setCanceling] = useState(false);

  const [selectedTrade, setSelectedTrade] = useState(null);
  const [selectedTradeIndex, setSelectedTradeIndex] = useState(null);
  const [selectedBotId, setSelectedBotId] = useState(null);
  const [tradesBotId, setTradesBotId] = useState(null);

  const [tradeFilters, setTradeFilters] = useState({
    long: true,
    short: true,
    win: true,
    loss: true,
  });

  const [visibleSeries, setVisibleSeries] = useState({});
  const [expandedParamGroups, setExpandedParamGroups] = useState({
    "Entry Settings": true,
    "Stop Loss": false,
    "Take Profit - Dual Flip": false,
    "Take Profit - RSI": false,
    "Debug": false,
  });

  const toggleParamGroup = (label) => {
    setExpandedParamGroups((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  const currentStrategy = strategyConfigs.find((s) => s.id === selectedStrategyId);
  const currentInputs = strategyInputs[selectedStrategyId] || {};

  const updateInput = (key, value) => {
    setStrategyInputs((prev) => ({
      ...prev,
      [selectedStrategyId]: {
        ...prev[selectedStrategyId],
        [key]: value,
      },
    }));
  };

  const updateNested = (key, subKey, value) => {
    setStrategyInputs((prev) => ({
      ...prev,
      [selectedStrategyId]: {
        ...prev[selectedStrategyId],
        [key]: { ...(prev[selectedStrategyId]?.[key] || {}), [subKey]: value },
      },
    }));
  };

  const rangeInvalid = useMemo(() => {
    if (rangePreset === "ALL") return false;
    if (rangeFrom && rangeTo && rangeFrom > rangeTo) return true;
    return false;
  }, [rangePreset, rangeFrom, rangeTo]);

  const computeRange = () => {
    switch (rangePreset) {
      case "7D":
        return { rangeType: "7D", from: presetFrom(7), to: todayIso() };
      case "30D":
        return { rangeType: "30D", from: presetFrom(30), to: todayIso() };
      case "90D":
        return { rangeType: "90D", from: presetFrom(90), to: todayIso() };
      case "365D":
        return { rangeType: "365D", from: presetFrom(365), to: todayIso() };
      case "2Y":
        return { rangeType: "2Y", from: presetFrom(730), to: todayIso() };
      case "3Y":
        return { rangeType: "3Y", from: presetFrom(1095), to: todayIso() };
      case "CUSTOM":
        return { rangeType: "CUSTOM", from: rangeFrom || null, to: rangeTo || null };
      case "ALL":
      default:
        return { rangeType: "ALL", from: null, to: null };
    }
  };

  const buildArray = (start, end, step) => {
    const arr = [];
    for (let v = start; v <= end; v += step) arr.push(v);
    return arr;
  };

  const handleRun = async () => {
    if (!currentStrategy?.supported || !["ema_cross", "rf_st_rsi"].includes(currentStrategy?.type)) {
      setError("Strategy này chưa được backend hỗ trợ trên UI hiện tại.");
      return;
    }
    if (rangeInvalid) {
      setError("Range không hợp lệ (from > to).");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedTrade(null);
    setSelectedTradeIndex(null);

    const tfNormalized = (currentInputs.timeframes || [])
      .map((t) => t.trim())
      .filter(Boolean)
      .map((t) => (/^\d+$/.test(t) ? `${t}m` : t)); // nếu chỉ nhập số, mặc định phút

    let params = {};

    if (currentStrategy.type === "ema_cross") {
      params = {
        emaFast: buildArray(
          Number(currentInputs.emaFastRange?.start),
          Number(currentInputs.emaFastRange?.end),
          Number(currentInputs.emaFastRange?.step)
        ),
        emaSlow: buildArray(
          Number(currentInputs.emaSlowRange?.start),
          Number(currentInputs.emaSlowRange?.end),
          Number(currentInputs.emaSlowRange?.step)
        ),
      };
    } else if (currentStrategy.type === "rf_st_rsi") {
      // Helper to convert range input {start, end, step} to array
      const rangeToArray = (input, defaults) => {
        if (Array.isArray(input)) return input;
        if (input && typeof input === "object" && "start" in input) {
          return buildArray(
            Number(input.start ?? defaults[0]),
            Number(input.end ?? defaults[1]),
            Number(input.step ?? defaults[2])
          );
        }
        return defaults;
      };

      params = {
        // Entry Settings
        showDualFlip: currentInputs.showDualFlip ?? true,
        showRSI: currentInputs.showRSI ?? true,
        st_useATR: currentInputs.st_useATR ?? true,
        st_atrPeriod: rangeToArray(currentInputs.st_atrPeriod, [8, 14, 2]),
        st_mult: rangeToArray(currentInputs.st_mult, [1.5, 3.0, 0.5]),
        rf_period: rangeToArray(currentInputs.rf_period, [80, 120, 20]),
        rf_mult: rangeToArray(currentInputs.rf_mult, [2.5, 4.0, 0.5]),
        rsi_length: rangeToArray(currentInputs.rsi_length, [10, 18, 2]),
        rsi_ma_length: rangeToArray(currentInputs.rsi_ma_length, [4, 8, 2]),
        // Stop Loss
        sl_st_useATR: currentInputs.sl_st_useATR ?? true,
        sl_st_atrPeriod: rangeToArray(currentInputs.sl_st_atrPeriod, [8, 14, 2]),
        sl_st_mult: rangeToArray(currentInputs.sl_st_mult, [3.0, 5.0, 0.5]),
        sl_rf_period: rangeToArray(currentInputs.sl_rf_period, [80, 120, 20]),
        sl_rf_mult: rangeToArray(currentInputs.sl_rf_mult, [5.0, 9.0, 1.0]),
        // Take Profit - Dual Flip
        tp_dual_st_atrPeriod: rangeToArray(currentInputs.tp_dual_st_atrPeriod, [8, 14, 2]),
        tp_dual_st_mult: rangeToArray(currentInputs.tp_dual_st_mult, [1.5, 3.0, 0.5]),
        tp_dual_rr_mult: rangeToArray(currentInputs.tp_dual_rr_mult, [1.0, 2.0, 0.3]),
        // Take Profit - RSI
        tp_rsi_st_atrPeriod: rangeToArray(currentInputs.tp_rsi_st_atrPeriod, [8, 14, 2]),
        tp_rsi_st_mult: rangeToArray(currentInputs.tp_rsi_st_mult, [1.5, 3.0, 0.5]),
        tp_rsi_rr_mult: rangeToArray(currentInputs.tp_rsi_rr_mult, [1.0, 2.0, 0.3]),
        // Debug
        debug: currentInputs.debug ?? false,
      };
    }

    const cfg = {
      symbols: [currentInputs.symbol],
      timeframes: tfNormalized,
      strategy: {
        type: currentStrategy.type,
        params,
      },
      filters: {
        minPF: Number(currentInputs.minPF),
      minTrades: Number(currentInputs.minTrades),
      maxDD: Number(currentInputs.maxDD),
    },
    minTFAgree: 1,
    properties: {
      initialCapital: Number(currentInputs.properties?.initialCapital ?? 0),
      baseCurrency: currentInputs.properties?.baseCurrency || "USD",
      orderSize: {
        value: Number(currentInputs.properties?.orderSize?.value ?? 0),
        type: currentInputs.properties?.orderSize?.type || "percent",
      },
      pyramiding: Number(currentInputs.properties?.pyramiding ?? 0),
      commission: {
        value: Number(currentInputs.properties?.commission?.value ?? 0),
        type: currentInputs.properties?.commission?.type || "percent",
      },
      slippage: Number(currentInputs.properties?.slippage ?? 0),
    },
    range: computeRange(),
  };

    try {
      const startResp = await startAiAgent(cfg);
      const jid = startResp.jobId;
      setJobId(jid);
      setProgress({ percent: 0, status: "running" });

      const poll = async () => {
        try {
          const prog = await getAiAgentProgress(jid);
          const percent = prog.total ? Math.min(100, Math.round((prog.progress / prog.total) * 100)) : 0;
          setProgress({ percent, status: prog.status || "running" });
          if (prog.status === "done") {
            const data = await getAiAgentResult(jid);
            const baseRuns = (data.all && data.all.length ? data.all : data.top || []).map((r, idx) => ({
              ...r,
              strategyId: `s${idx + 1}`,
            }));
            setResult({
              ...data,
              runs: baseRuns,
            });
            if (baseRuns.length) {
              setSelectedBotId(baseRuns[0].strategyId);
              setTradesBotId(baseRuns[0].strategyId);
            }
            setAgentComment(data.comment || "");

            const vis = {};
            baseRuns.slice(0, 5).forEach((r) => {
              vis[r.strategyId] = true;
            });
            setVisibleSeries(vis);
            setLoading(false);
            return;
          }
          if (prog.status === "error") {
            setError(prog.error || "AI Agent failed");
            setLoading(false);
            setJobId(null);
            return;
          }
          if (prog.status === "canceled") {
            setError("Đã hủy bởi người dùng");
            setLoading(false);
            setJobId(null);
            return;
          }
          setTimeout(poll, 1500);
        } catch (err) {
          setError(err.message || "AI Agent failed");
          setLoading(false);
          setJobId(null);
        }
      };
      poll();
    } catch (e) {
      setError(e.message || "AI Agent failed");
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    setCanceling(true);
    try {
      await cancelAiAgent(jobId);
      setProgress({ percent: progress.percent, status: "canceled" });
      setLoading(false);
      setJobId(null);
    } catch (e) {
      setError(e.message || "Cancel failed");
      setLoading(false);
      setJobId(null);
    } finally {
      setCanceling(false);
    }
  };

  /**
   * ===== Equity curve (base) =====
   */
  const equityData = useMemo(() => {
    if (!result?.best?.equityCurve) return [];
    const curve = result.best.equityCurve;
    if (!curve.length) return [];

    const base = curve[0].equity;

    return curve.map((p, i) => ({
      index: i,
      pnlPct: ((p.equity - base) / base) * 100,
      time: p.time,
    }));
  }, [result]);

  /**
   * ===== Highlight equity for selected trade =====
   */
  const highlightedEquity = useMemo(() => {
    if (!selectedTrade || !equityData.length) return [];

    const { entry_time, exit_time } = selectedTrade;

    return equityData.filter(
      (p) => p.time >= entry_time && p.time <= exit_time
    );
  }, [selectedTrade, equityData]);

  /**
   * ===== D1.8: Filter trades =====
   */
  const selectedBot = useMemo(() => {
    if (!result?.runs?.length) return null;
    const fromSelection = selectedBotId
      ? result.runs.find((r) => r.strategyId === selectedBotId)
      : null;
    if (fromSelection) return fromSelection;
    const best = result.best;
    if (best) {
      const match = result.runs.find(
        (r) =>
          r.symbol === best.symbol &&
          r.timeframe === best.timeframe &&
          JSON.stringify(r.params || {}) === JSON.stringify(best.params || {})
      );
      if (match) return match;
    }
    return result.runs[0];
  }, [result, selectedBotId]);

  useEffect(() => {
    if (!tradesBotId && selectedBot) {
      setTradesBotId(selectedBot.strategyId);
    }
  }, [tradesBotId, selectedBot]);

  useEffect(() => {
    setSelectedTrade(null);
    setSelectedTradeIndex(null);
  }, [tradesBotId]);

  const tradeBotOptions = useMemo(
    () =>
      (result?.runs || []).map((r, idx) => ({
        value: r.strategyId,
        label: `#${idx + 1} ${r.strategyId} ${r.symbol} ${r.timeframe}`,
      })),
    [result?.runs]
  );

  const tradesBot = useMemo(() => {
    if (tradesBotId && result?.runs) {
      const found = result.runs.find((r) => r.strategyId === tradesBotId);
      if (found) return found;
    }
    return selectedBot;
  }, [tradesBotId, result?.runs, selectedBot]);

  const filteredTrades = useMemo(() => {
    if (!tradesBot?.trades) return [];

    return tradesBot.trades
      .filter((t) => {
        const isLong = t.side?.toLowerCase() === "long";
        const isWin = t.pnl >= 0;

        if (!tradeFilters.long && isLong) return false;
        if (!tradeFilters.short && !isLong) return false;
        if (!tradeFilters.win && isWin) return false;
        if (!tradeFilters.loss && !isWin) return false;

        return true;
      })
      .sort((a, b) =>
        (b.exit_time || b.entry_time || 0).localeCompare(
          a.exit_time || a.entry_time || 0
        )
      );
  }, [tradesBot, tradeFilters]);

  const bestKey = result?.best
    ? {
        symbol: result.best.symbol,
        timeframe: result.best.timeframe,
        params: result.best.params,
      }
    : null;

  const handleHighlightBest = () => {
    const runs = result?.all && result.all.length ? result.all : result?.top || [];
    if (!runs.length || !bestKey) return;
    const idx = runs.findIndex(
      (r) =>
        r.symbol === bestKey.symbol &&
        r.timeframe === bestKey.timeframe &&
        JSON.stringify(r.params || {}) === JSON.stringify(bestKey.params || {})
    );
    if (idx === -1) return;
    const key = `s${idx + 1}`;
    setVisibleSeries({ [key]: true });
    setSelectedBotId(key);
  };

  const sortedRuns = useMemo(() => {
    const runs = result?.runs || result?.all || result?.top || [];
    const arr = [...runs];
    const getSummary = (r) => r.summary || {};
    arr.sort((a, b) => {
      const sa = getSummary(a);
      const sb = getSummary(b);
      switch (sortBy) {
        case "pnl":
          return (sb.netProfit || 0) - (sa.netProfit || 0);
        case "dd":
          return (sa.maxDrawdownPct || 0) - (sb.maxDrawdownPct || 0); // ít DD hơn trước
        case "pf":
          return (sb.profitFactor || 0) - (sa.profitFactor || 0);
        case "wr":
          return (sb.winrate || 0) - (sa.winrate || 0);
        case "score":
        default:
          return (sb.score || 0) - (sa.score || 0);
      }
    });
    // giữ nguyên strategyId để đồng bộ chart/legend/table
    return arr.map((r, idx) => ({ ...r, rank: idx + 1 }));
  }, [result, sortBy]);

  // Khi thay đổi sort hoặc result mới, reset visible (top N)
  useEffect(() => {
    const runs = sortedRuns;
    if (!runs.length) return;
    setCurrentPageOverlay(1);
    setCurrentPageBots(1);
  }, [result]);

  useEffect(() => {
    setCurrentPageOverlay(1);
  }, [overlayLimit]);

  const overlayBase = useMemo(() => {
    if (overlayLimit === "all") return sortedRuns;
    const n = Number(overlayLimit || 0);
    return sortedRuns.slice(0, n || sortedRuns.length);
  }, [sortedRuns, overlayLimit]);

  const overlayPageSize = 20; // chỉ ảnh hưởng danh sách toggle, không ảnh hưởng chart

  useEffect(() => {
    if (!overlayBase.length) return;
    const vis = {};
    overlayBase.forEach((r) => {
      vis[r.strategyId] = true;
    });
    setVisibleSeries(vis);
    setCurrentPageOverlay(1);
  }, [overlayBase]);

  const pagedOverlayRuns = useMemo(() => {
    const start = (currentPageOverlay - 1) * overlayPageSize;
    return overlayBase.slice(start, start + overlayPageSize);
  }, [overlayBase, currentPageOverlay, overlayPageSize]);

  const equityOverlay = useMemo(() => {
    const runs = overlayBase;
    if (!runs.length) return [];
    // luôn dùng %PnL overlay để trục Y hiển thị tăng trưởng đúng
    return buildEquitySeries(runs, "pct", currentInputs.properties);
  }, [overlayBase, currentInputs.properties]);

  const startEquityMap = useMemo(() => {
    const map = {};
    pagedOverlayRuns.forEach((r, idx) => {
      const key = r.strategyId || `s${idx + 1}`;
      const startEq = r.equityCurve?.[0]?.equity || 0;
      map[key] = startEq;
    });
    return map;
  }, [pagedOverlayRuns]);

    const { yDomain, tickFmt } = useMemo(() => {
    const keys = Object.entries(visibleSeries)
      .filter(([, v]) => v)
      .map(([k]) => k);
    let min = Infinity;
    let max = -Infinity;
    equityOverlay.forEach((p) => {
      keys.forEach((k) => {
        const v = Number(p[k]);
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      });
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = -1;
      max = 1;
    }
    if (min === max) {
      const pad = Math.abs(min) * 0.1 || 1;
      min -= pad;
      max += pad;
    } else {
      const pad = (max - min) * 0.05;
      min -= pad;
      max += pad;
    }
    const tickFormatter = (v) => `${Number(v).toFixed(2)}%`;
    return { yDomain: [min, max], tickFmt: tickFormatter };
  }, [equityOverlay, visibleSeries]);

  const bestRun = useMemo(() => {
    if (!result?.best) return sortedRuns[0];
    const key = {
      symbol: result.best.symbol,
      timeframe: result.best.timeframe,
      params: result.best.params,
    };
    return (
      sortedRuns.find(
        (r) =>
          r.symbol === key.symbol &&
          r.timeframe === key.timeframe &&
          JSON.stringify(r.params || {}) === JSON.stringify(key.params || {})
      ) || sortedRuns[0]
    );
  }, [result, sortedRuns]);

  const bestEquityCurve = useMemo(() => {
    if (!bestRun) return [];
    if (bestRun.equityCurve && bestRun.equityCurve.length) return bestRun.equityCurve;
    if (bestRun.trades && bestRun.trades.length) {
      return computeEquityCurveFromTrades(bestRun.trades, currentInputs.properties);
    }
    return [];
  }, [bestRun, currentInputs.properties]);

  const equityKPI = useMemo(() => {
    if (!bestEquityCurve.length) return null;
    const initial = currentInputs?.properties?.initialCapital || bestEquityCurve[0]?.equity;
    const equities = bestEquityCurve.map((p) => p.equity);
    const finalEq = equities[equities.length - 1];
    const maxEq = Math.max(...equities);
    const minEq = Math.min(...equities);
    const growth = initial ? ((finalEq - initial) / initial) * 100 : 0;
    return { finalEq, maxEq, minEq, growth };
  }, [bestEquityCurve, currentInputs]);

  const renderInput = (label, value, setter, step = 1) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <label style={{ fontSize: 13, opacity: 0.75 }}>{label}</label>
      <input
        type="number"
        value={value ?? ""}
        onChange={(e) => setter(Number(e.target.value))}
        step={step}
        style={inputStyle}
      />
    </div>
  );

  const badge = (text) => (
    <span
      style={{
        padding: "6px 12px",
        borderRadius: 999,
        background: "rgba(255,255,255,0.08)",
        fontSize: 12,
      }}
    >
      {text}
    </span>
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        padding: "32px 18px 60px",
        background:
          "radial-gradient(1200px 700px at 20% 0%, rgba(59,130,246,0.25), rgba(2,6,23,0.95)), linear-gradient(160deg, #0b1224 0%, #0f172a 45%, #0a0f1f 100%)",
        color: "#e5e7eb",
      }}
    >
      <div style={{ maxWidth: 1800, margin: "0 auto" }}>
        <Header onRun={handleRun} loading={loading} />

        {error && (
          <p style={{ color: "#ef4444", marginBottom: 12 }}>{error}</p>
        )}

        {/* Removed top run control bar as requested */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "300px 1fr",
            gap: 20,
            alignItems: "stretch",
          }}
        >
          {/* ================= LEFT: STRATEGY LIST ================= */}
          <div style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ margin: 0, fontSize: 16, letterSpacing: 0.3 }}>Strategy</h2>
              <button style={{ ...legendChip, padding: "6px 10px" }}>+ New</button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 10 }}>
              {strategyConfigs.map((s) => {
                const active = s.id === selectedStrategyId;
                return (
                  <button
                    key={s.id}
                    onClick={() => setSelectedStrategyId(s.id)}
                    style={{
                      textAlign: "left",
                      padding: "10px 12px",
                      borderRadius: 14,
                      border: active ? "1px solid rgba(34,197,94,0.6)" : "1px solid rgba(255,255,255,0.08)",
                      background: active
                        ? "linear-gradient(180deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05))"
                        : "rgba(255,255,255,0.03)",
                      color: "#e5e7eb",
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ fontWeight: 700 }}>{s.name}</div>
                    <div style={{ fontSize: 12, opacity: 0.75 }}>{s.subtitle}</div>
                    {!s.supported && (
                      <div style={{ marginTop: 6, fontSize: 11, color: "#f59e0b" }}>Chưa hỗ trợ backend</div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* ================= INPUT FULL WIDTH (remaining) ================= */}
          <div style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
              <h2 style={{ margin: 0, fontSize: 18, letterSpacing: 0.3 }}>Input</h2>
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span style={{ fontSize: 12, opacity: 0.75 }}>Range</span>
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
                      ...chip,
                      background: rangePreset === opt.key ? "rgba(34,197,94,0.2)" : chip.background,
                      color: rangePreset === opt.key ? "#22c55e" : "#e5e7eb",
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
                <input
                  type="date"
                  value={rangeFrom}
                  onChange={(e) => {
                    setRangeFrom(e.target.value);
                    setRangePreset("CUSTOM");
                  }}
                  style={dateInput}
                />
                <span style={{ opacity: 0.6 }}>to</span>
                <input
                  type="date"
                  value={rangeTo}
                  onChange={(e) => {
                    setRangeTo(e.target.value);
                    setRangePreset("CUSTOM");
                  }}
                  style={dateInput}
                />
              </div>
            </div>

            <div
              style={{
                marginTop: 14,
                display: "grid",
                gridTemplateColumns: "repeat(3, minmax(0,1fr))",
                gap: 12,
              }}
            >
              {/* Col 1: Market / Context */}
              <div style={inputColumn}>
                <div style={groupTitle}>Market / Context</div>
                <label style={label}>Symbol</label>
                <input
                  value={currentInputs.symbol || ""}
                  onChange={(e) => updateInput("symbol", e.target.value.toUpperCase())}
                  style={inputStyle}
                />
                <label style={{ ...label, marginTop: 10 }}>Timeframes (comma)</label>
                <input
                  value={(currentInputs.timeframes || []).join(",")}
                  onChange={(e) =>
                    updateInput(
                      "timeframes",
                      e.target.value
                        .split(",")
                        .map((s) => s.trim())
                        .filter(Boolean)
                    )
                  }
                  style={inputStyle}
                />
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                  {tfOptions.map((tf) => {
                    const active = (currentInputs.timeframes || []).includes(tf);
                    return (
                      <button
                        key={tf}
                        onClick={() => {
                          updateInput(
                            "timeframes",
                            active
                              ? (currentInputs.timeframes || []).filter((x) => x !== tf)
                              : [...(currentInputs.timeframes || []), tf]
                          );
                        }}
                        style={{
                          padding: "6px 10px",
                          borderRadius: 999,
                          border: "1px solid rgba(255,255,255,0.12)",
                          background: active ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.04)",
                          color: active ? "#22c55e" : "#e5e7eb",
                          cursor: "pointer",
                          fontSize: 12,
                        }}
                      >
                        {tf}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Col 2: Capital & Size */}
              <div style={inputColumn}>
                <div style={groupTitle}>Capital & Size</div>
                <label style={label}>Initial capital {infoDot("Vốn ban đầu cho backtest")}</label>
                <input
                  type="number"
                  value={currentInputs.properties?.initialCapital ?? ""}
                  onChange={(e) =>
                    updateInput("properties", {
                      ...(currentInputs.properties || {}),
                      initialCapital: Number(e.target.value),
                    })
                  }
                  style={inputStyle}
                />
                <label style={{ ...label, marginTop: 10 }}>Base currency {infoDot("Đơn vị base để tính PnL / size")}</label>
                <select
                  value={currentInputs.properties?.baseCurrency || "USDT"}
                  onChange={(e) =>
                    updateInput("properties", {
                      ...(currentInputs.properties || {}),
                      baseCurrency: e.target.value,
                    })
                  }
                  style={{ ...inputStyle, appearance: "none", cursor: "pointer" }}
                >
                  {["USDT"].map((c) => (
                    <option key={c} value={c} style={{ background: "#0f172a" }}>
                      {c}
                    </option>
                  ))}
                </select>
                <label style={{ ...label, marginTop: 10 }}>Default order size {infoDot("Position sizing mặc định")}</label>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 8 }}>
                  <input
                    type="number"
                    value={currentInputs.properties?.orderSize?.value ?? ""}
                    onChange={(e) =>
                      updateNested("properties", "orderSize", {
                        ...(currentInputs.properties?.orderSize || {}),
                        value: Number(e.target.value),
                      })
                    }
                    style={inputStyle}
                  />
                  <select
                    value={currentInputs.properties?.orderSize?.type || "percent"}
                    onChange={(e) =>
                      updateNested("properties", "orderSize", {
                        ...(currentInputs.properties?.orderSize || {}),
                        type: e.target.value,
                      })
                    }
                    style={{ ...inputStyle, appearance: "none", cursor: "pointer" }}
                  >
                    <option value="percent">% of equity</option>
                    <option value="fixed">USDT</option>
                  </select>
                </div>
              </div>

              {/* Col 3: Execution / Risk */}
              <div style={inputColumn}>
                <div style={groupTitle}>Execution / Risk</div>
                <label style={label}>Pyramiding {infoDot("Cho phép mở thêm vị thế chồng lệnh")}</label>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 8, alignItems: "center" }}>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={currentInputs.properties?.pyramiding ?? ""}
                    onChange={(e) =>
                      updateInput("properties", {
                        ...(currentInputs.properties || {}),
                        pyramiding: Number(e.target.value),
                      })
                    }
                    style={inputStyle}
                  />
                  <div style={{ fontSize: 12, opacity: 0.7 }}>orders</div>
                </div>

                <label style={{ ...label, marginTop: 10 }}>Commission {infoDot("Phí cho mỗi lệnh")}</label>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 8 }}>
                  <input
                    type="number"
                    step="0.01"
                    inputMode="decimal"
                    value={
                      currentInputs.properties?.commission?.value === 0
                        ? 0
                        : currentInputs.properties?.commission?.value ?? ""
                    }
                    onChange={(e) =>
                      updateNested("properties", "commission", {
                        ...(currentInputs.properties?.commission || {}),
                        value: e.target.value === "" ? "" : parseFloat(e.target.value),
                      })
                    }
                    style={inputStyle}
                  />
                  <select
                    value={currentInputs.properties?.commission?.type || "percent"}
                    onChange={(e) =>
                      updateNested("properties", "commission", {
                        ...(currentInputs.properties?.commission || {}),
                        type: e.target.value,
                      })
                    }
                    style={{ ...inputStyle, appearance: "none", cursor: "pointer" }}
                  >
                    <option value="percent">%</option>
                    <option value="fixed">Fixed</option>
                  </select>
                </div>

                <label style={{ ...label, marginTop: 10 }}>Slippage {infoDot("Tick slippage cho mỗi lệnh")}</label>
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 8, alignItems: "center" }}>
                  <input
                    type="number"
                    step="0.01"
                    inputMode="decimal"
                    value={
                      currentInputs.properties?.slippage === 0
                        ? 0
                        : currentInputs.properties?.slippage ?? ""
                    }
                    onChange={(e) =>
                      updateInput("properties", {
                        ...(currentInputs.properties || {}),
                        slippage: e.target.value === "" ? "" : parseFloat(e.target.value),
                      })
                    }
                    style={inputStyle}
                  />
                  <div style={{ fontSize: 12, opacity: 0.7 }}>ticks</div>
                </div>
              </div>
            </div>

            {/* Strategy Params full width */}
            <div style={{ marginTop: 14 }}>
              <div style={inputColumn}>
                <div style={groupTitle}>Strategy params</div>
                {currentStrategy?.id === "ema" && (
                  <>
                    <label style={label}>EMA Fast (start, end, step)</label>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 8 }}>
                      <input
                        type="number"
                        value={currentInputs.emaFastRange?.start || 0}
                        onChange={(e) =>
                          updateInput("emaFastRange", { ...currentInputs.emaFastRange, start: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%" }}
                      />
                      <input
                        type="number"
                        value={currentInputs.emaFastRange?.end || 0}
                        onChange={(e) =>
                          updateInput("emaFastRange", { ...currentInputs.emaFastRange, end: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%" }}
                      />
                      <input
                        type="number"
                        value={currentInputs.emaFastRange?.step || 1}
                        onChange={(e) =>
                          updateInput("emaFastRange", { ...currentInputs.emaFastRange, step: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%", opacity: 0.8 }}
                      />
                    </div>

                    <label style={{ ...label, marginTop: 10 }}>EMA Slow (start, end, step)</label>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 8 }}>
                      <input
                        type="number"
                        value={currentInputs.emaSlowRange?.start || 0}
                        onChange={(e) =>
                          updateInput("emaSlowRange", { ...currentInputs.emaSlowRange, start: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%" }}
                      />
                      <input
                        type="number"
                        value={currentInputs.emaSlowRange?.end || 0}
                        onChange={(e) =>
                          updateInput("emaSlowRange", { ...currentInputs.emaSlowRange, end: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%" }}
                      />
                      <input
                        type="number"
                        value={currentInputs.emaSlowRange?.step || 1}
                        onChange={(e) =>
                          updateInput("emaSlowRange", { ...currentInputs.emaSlowRange, step: Number(e.target.value) })
                        }
                        style={{ ...inputStyle, width: "100%", opacity: 0.8 }}
                      />
                    </div>
                  </>
                )}

                {currentStrategy?.id === "rf_st_rsi" && (
                  <div style={{ marginTop: 12 }}>
                    {/* Entry Settings Group */}
                    <div style={{ marginBottom: 12, border: "1px solid #2d2d44", borderRadius: 8, overflow: "hidden" }}>
                      <div
                        onClick={() => toggleParamGroup("Entry Settings")}
                        style={{
                          background: "#1a1a2e",
                          padding: "12px 16px",
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600, fontSize: 14 }}>Entry Settings</span>
                        <span style={{ color: "#9ca3af" }}>{expandedParamGroups["Entry Settings"] ? "▼" : "▶"}</span>
                      </div>
                      {expandedParamGroups["Entry Settings"] && (
                        <div style={{ padding: 16, background: "#0f0f1a" }}>
                          {/* Boolean options */}
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 16 }}>
                            <BoolInput label="Enable Dual Flip Entry" inputKey="showDualFlip" defaultVal={true} currentInputs={currentInputs} updateInput={updateInput} />
                            <BoolInput label="Enable RSI Divergence Entry" inputKey="showRSI" defaultVal={true} currentInputs={currentInputs} updateInput={updateInput} />
                            <BoolInput label="ST use ATR?" inputKey="st_useATR" defaultVal={true} currentInputs={currentInputs} updateInput={updateInput} />
                          </div>
                          {/* Range inputs */}
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
                            <RangeInput label="ST ATR Period" inputKey="st_atrPeriod" defaults={[8, 14, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="ST Multiplier" inputKey="st_mult" defaults={[1.5, 3.0, 0.5]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="RF Period" inputKey="rf_period" defaults={[80, 120, 20]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="RF Multiplier" inputKey="rf_mult" defaults={[2.5, 4.0, 0.5]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="RSI Length" inputKey="rsi_length" defaults={[10, 18, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="MA Length on RSI" inputKey="rsi_ma_length" defaults={[4, 8, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Stop Loss Group */}
                    <div style={{ marginBottom: 12, border: "1px solid #2d2d44", borderRadius: 8, overflow: "hidden" }}>
                      <div
                        onClick={() => toggleParamGroup("Stop Loss")}
                        style={{
                          background: "#1a1a2e",
                          padding: "12px 16px",
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600, fontSize: 14 }}>Stop Loss</span>
                        <span style={{ color: "#9ca3af" }}>{expandedParamGroups["Stop Loss"] ? "▼" : "▶"}</span>
                      </div>
                      {expandedParamGroups["Stop Loss"] && (
                        <div style={{ padding: 16, background: "#0f0f1a" }}>
                          {/* Boolean options */}
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 16 }}>
                            <BoolInput label="ST SL use ATR?" inputKey="sl_st_useATR" defaultVal={true} currentInputs={currentInputs} updateInput={updateInput} />
                          </div>
                          {/* Range inputs */}
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
                            <RangeInput label="ST SL ATR Period" inputKey="sl_st_atrPeriod" defaults={[8, 14, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="ST SL Mult" inputKey="sl_st_mult" defaults={[3.0, 5.0, 0.5]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="RF SL Period" inputKey="sl_rf_period" defaults={[80, 120, 20]} currentInputs={currentInputs} updateInput={updateInput} />
                            <RangeInput label="RF SL Multiplier" inputKey="sl_rf_mult" defaults={[5.0, 9.0, 1.0]} currentInputs={currentInputs} updateInput={updateInput} />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Take Profit - Dual Flip Group */}
                    <div style={{ marginBottom: 12, border: "1px solid #2d2d44", borderRadius: 8, overflow: "hidden" }}>
                      <div
                        onClick={() => toggleParamGroup("Take Profit - Dual Flip")}
                        style={{
                          background: "#1a1a2e",
                          padding: "12px 16px",
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600, fontSize: 14 }}>Take Profit - Dual Flip</span>
                        <span style={{ color: "#9ca3af" }}>{expandedParamGroups["Take Profit - Dual Flip"] ? "▼" : "▶"}</span>
                      </div>
                      {expandedParamGroups["Take Profit - Dual Flip"] && (
                        <div style={{ padding: 16, background: "#0f0f1a", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
                          <RangeInput label="ST TP ATR Period" inputKey="tp_dual_st_atrPeriod" defaults={[8, 14, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                          <RangeInput label="ST TP Mult" inputKey="tp_dual_st_mult" defaults={[1.5, 3.0, 0.5]} currentInputs={currentInputs} updateInput={updateInput} />
                          <RangeInput label="TP R:R Mult" inputKey="tp_dual_rr_mult" defaults={[1.0, 2.0, 0.3]} currentInputs={currentInputs} updateInput={updateInput} />
                        </div>
                      )}
                    </div>

                    {/* Take Profit - RSI Group */}
                    <div style={{ marginBottom: 12, border: "1px solid #2d2d44", borderRadius: 8, overflow: "hidden" }}>
                      <div
                        onClick={() => toggleParamGroup("Take Profit - RSI")}
                        style={{
                          background: "#1a1a2e",
                          padding: "12px 16px",
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600, fontSize: 14 }}>Take Profit - RSI</span>
                        <span style={{ color: "#9ca3af" }}>{expandedParamGroups["Take Profit - RSI"] ? "▼" : "▶"}</span>
                      </div>
                      {expandedParamGroups["Take Profit - RSI"] && (
                        <div style={{ padding: 16, background: "#0f0f1a", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
                          <RangeInput label="ST TP ATR Period" inputKey="tp_rsi_st_atrPeriod" defaults={[8, 14, 2]} currentInputs={currentInputs} updateInput={updateInput} />
                          <RangeInput label="ST TP Mult" inputKey="tp_rsi_st_mult" defaults={[1.5, 3.0, 0.5]} currentInputs={currentInputs} updateInput={updateInput} />
                          <RangeInput label="TP R:R Mult" inputKey="tp_rsi_rr_mult" defaults={[1.0, 2.0, 0.3]} currentInputs={currentInputs} updateInput={updateInput} />
                        </div>
                      )}
                    </div>

                    {/* Debug Group */}
                    <div style={{ marginBottom: 12, border: "1px solid #2d2d44", borderRadius: 8, overflow: "hidden" }}>
                      <div
                        onClick={() => toggleParamGroup("Debug")}
                        style={{
                          background: "#1a1a2e",
                          padding: "12px 16px",
                          cursor: "pointer",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <span style={{ fontWeight: 600, fontSize: 14 }}>Debug</span>
                        <span style={{ color: "#9ca3af" }}>{expandedParamGroups["Debug"] ? "▼" : "▶"}</span>
                      </div>
                      {expandedParamGroups["Debug"] && (
                        <div style={{ padding: 16, background: "#0f0f1a" }}>
                          <BoolInput label="Enable Debug Logging" inputKey="debug" defaultVal={false} currentInputs={currentInputs} updateInput={updateInput} />
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {!currentStrategy?.supported && (
                  <div style={{ marginTop: 12, color: "#f59e0b", fontSize: 12 }}>
                    Strategy này chưa được nối backend trong UI hiện tại. Bạn vẫn có thể chỉnh input và lưu preset, nhưng nút chạy sẽ bị khóa.
                  </div>
                )}
              </div>
            </div>

            <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
              <button
                onClick={handleRun}
                disabled={loading || !currentStrategy?.supported || rangeInvalid}
                style={{ ...primaryBtn, opacity: !currentStrategy?.supported ? 0.6 : 1 }}
              >
                {loading ? "Đang chạy..." : "Chạy AI Agent"}
              </button>
              {loading && (
                <div style={{ minWidth: 180, display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ fontSize: 12, opacity: 0.8, minWidth: 80 }}>
                    {progress.status === "running" ? `Running... ${progress.percent}%` : progress.status}
                  </div>
                  <div style={{ flex: 1, height: 6, borderRadius: 999, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${progress.percent || 0}%`,
                        height: "100%",
                        background: "linear-gradient(90deg, #22c55e, #3b82f6)",
                        transition: "width 0.3s ease",
                      }}
                    />
                  </div>
                  <button
                    onClick={handleCancel}
                    disabled={canceling}
                    style={{
                      ...chip,
                      padding: "8px 12px",
                      background: canceling ? "rgba(255,255,255,0.08)" : "rgba(239,68,68,0.15)",
                      border: "1px solid rgba(239,68,68,0.5)",
                      color: "#f87171",
                      cursor: canceling ? "not-allowed" : "pointer",
                    }}
                  >
                    {canceling ? "Canceling..." : "Cancel"}
                  </button>
                </div>
              )}
              {badge(`TF: ${(currentInputs.timeframes || []).join(", ")}`)}
              {currentStrategy?.id === "ema" && (
                <>
                  {badge(
                    `EmaFast ${currentInputs.emaFastRange?.start}-${currentInputs.emaFastRange?.end}/${currentInputs.emaFastRange?.step}`
                  )}
                  {badge(
                    `EmaSlow ${currentInputs.emaSlowRange?.start}-${currentInputs.emaSlowRange?.end}/${currentInputs.emaSlowRange?.step}`
                  )}
                </>
              )}
              {currentStrategy?.id === "rf_st_rsi" && (
                <>
                  {badge(
                    `ST-ATR ${Array.isArray(currentInputs.st_atrPeriod) ? currentInputs.st_atrPeriod.join("-") : "8-10-12"}`
                  )}
                  {badge(
                    `ST-Mult ${Array.isArray(currentInputs.st_mult) ? currentInputs.st_mult.join("-") : "1.8-2.0-2.2"}`
                  )}
                  {badge(
                    `RF-Perio ${Array.isArray(currentInputs.rf_period) ? currentInputs.rf_period.join("-") : "80-100-120"}`
                  )}
                  {badge(
                    `RF-Mult ${Array.isArray(currentInputs.rf_mult) ? currentInputs.rf_mult.join("-") : "2.8-3.0-3.2"}`
                  )}
                </>
              )}
            </div>

            {result && (
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  marginTop: 12,
                }}
              >
                <div style={badgeBox}>Runs: {result.stats?.totalRuns || 0}</div>
                <div style={badgeBox}>Passed: {result.stats?.passedRuns || 0}</div>
                <div style={{ ...badgeBox, flex: 1 }}>
                  <span style={{ opacity: 0.7 }}>Agent: </span>
                  {agentComment || "—"}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ================= RESULTS BELOW ================= */}
        <div style={{ marginTop: 24 }}>
          {result ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                <div style={statusCard}>
                  <div style={statusTitle}>Preset đang test</div>
                  <div style={statusValue}>{result?.preset || "Chưa chọn"}</div>
                  <div style={statusNote}>Bấm chạy để bắt đầu.</div>
                </div>
                <div style={statusCard}>
                  <div style={statusTitle}>Best hiện tại</div>
                  <div style={statusValue}>
                    {result.best ? `${result.best.symbol} ${result.best.timeframe}` : "Chưa có"}
                  </div>
                  <div style={statusNote}>
                    PnL {fmtPct(result.best?.summary?.netProfitPct)} · Win {fmtPct(result.best?.summary?.winrate)}
                  </div>
                </div>
                <div style={statusCard}>
                  <div style={statusTitle}>Log mới nhất</div>
                  <div style={statusNote}>{agentComment || "—"}</div>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16 }}>
                <Section title="Equity / PnL (overlay nhiều chiến lược)">
                  <div style={{ ...chartCard }}>
                    <div style={{ flex: 1, minHeight: "100%", width: "100%" }}>
                      {equityOverlay.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={equityOverlay}>
                            <XAxis dataKey="time" tick={false} axisLine={false} tickLine={false} />
                            <YAxis
                              domain={yDomain}
                              allowDataOverflow={true}
                              tickFormatter={tickFmt}
                              axisLine={false}
                              tickLine={false}
                              stroke="#334155"
                            />
                            <Tooltip
                              isAnimationActive={false}
                              content={({ label, payload }) => {
                                if (!payload || payload.length === 0) return null;
                                const items = payload
                                  .filter((p) => p && p.value !== undefined)
                                  .map((p) => ({
                                    id: p.dataKey,
                                    value: p.value,
                                    color: p.stroke || "#e5e7eb",
                                  }))
                                  .sort((a, b) => b.value - a.value);
                                const initCap = Number(currentInputs.properties?.initialCapital || 0);

                                return (
                                  <div
                                    style={{
                                      background: "#0b1021",
                                      border: "1px solid rgba(255,255,255,0.15)",
                                      borderRadius: 10,
                                      padding: 10,
                                      color: "#e5e7eb",
                                      maxHeight: 320,
                                      overflow: "auto",
                                      minWidth: 160,
                                    }}
                                  >
                                    <div style={{ marginBottom: 6, fontWeight: 700 }}>
                                      {label}
                                    </div>
                                    {items.map((item, idx) => (
                                      <div key={item.id} style={{ display: "flex", gap: 6, fontSize: 12 }}>
                                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: item.color, marginTop: 4 }} />
                                        <span style={{ color: "#94a3b8" }}>#{idx + 1}</span>
                                        <span style={{ color: item.color, fontWeight: 700 }}>{item.id}</span>
                                        <span>
                                          {(() => {
                                            const pct = item.value;
                                            const eq = initCap ? initCap * (1 + pct / 100) : 0;
                                            return `$${eq ? Number(eq).toLocaleString() : 0} (${pct.toFixed(2)}%)`;
                                          })()}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                );
                              }}
                            />
                            {sortedRuns.map((run, idx) => {
                              const key = run.strategyId || `s${idx + 1}`;
                              const isBestLine =
                                bestKey &&
                                run.symbol === bestKey.symbol &&
                                run.timeframe === bestKey.timeframe &&
                                JSON.stringify(run.params || {}) === JSON.stringify(bestKey.params || {});
                              const visible = visibleSeries[key];
                              return (
                                <Line
                                  key={key}
                                  type="monotone"
                                  dataKey={key}
                                  stroke={isBestLine ? "#22d3ee" : palette[idx % palette.length]}
                                  strokeWidth={isBestLine ? 2.6 : 1.2}
                                  opacity={visible ? (isBestLine ? 0.95 : 0.5) : 0.15}
                                  dot={false}
                                  hide={!visible}
                                />
                              );
                            })}
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <EmptyText text="Đang khởi tạo chart..." />
                      )}
                    </div>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    {sortedRuns.length === 1 ? (
                      <span style={legendChip}>Chỉ có 1 strategy – chưa có overlay</span>
                    ) : (
                      <>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 6 }}>
                          <span style={{ fontSize: 12, opacity: 0.7 }}>
                    Overlay {pagedOverlayRuns.length} / {overlayBase.length} strategies
                  </span>
                          {["5", "10", "20", "50", "all"].map((n) => (
                            <button
                              key={n}
                              onClick={() => {
                                setOverlayLimit(n);
                              }}
                              style={{
                                ...legendChip,
                                padding: "6px 10px",
                                background: overlayLimit === n ? "rgba(34,197,94,0.2)" : legendChip.background,
                                color: overlayLimit === n ? "#22c55e" : "#e5e7eb",
                              }}
                            >
                              {n === "all" ? "All" : `Top ${n}`}
                            </button>
                          ))}
                        </div>
                        <StrategyToggle
                          results={pagedOverlayRuns}
                          visible={visibleSeries}
                          setVisible={setVisibleSeries}
                          onSelect={(key) => setSelectedBotId(key)}
                        />
                    <Pagination
                          current={currentPageOverlay}
                          total={overlayBase.length}
                          pageSize={overlayPageSize}
                          onChange={(p) => setCurrentPageOverlay(p)}
                        />
                      </>
                    )}
                  </div>
                </Section>

                <Section title="Bots / chiến lược (từ backend)">
                  <div style={{ ...pill, height: "100%", display: "flex", flexDirection: "column", gap: 12 }}>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {["score", "pnl", "dd", "pf", "wr"].map((k) => (
                        <button
                          key={k}
                          onClick={() => setSortBy(k)}
                          style={{
                            ...legendChip,
                            padding: "6px 10px",
                            background: sortBy === k ? "rgba(34,197,94,0.15)" : legendChip.background,
                            color: sortBy === k ? "#22c55e" : "#e5e7eb",
                          }}
                        >
                          Sort: {k.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <StrategyTable
                      rows={sortedRuns}
                      pageSize={pageSize}
                      currentPage={currentPageBots}
                      bestKey={bestKey}
                      onSelectRow={(row) => {
                        const key = row.strategyId;
                        setVisibleSeries((v) => ({ ...v, [key]: !v[key] }));
                        setSelectedBotId(key);
                      }}
                    />
                    <Pagination
                      current={currentPageBots}
                      total={sortedRuns.length}
                      pageSize={pageSize}
                      onChange={(p) => setCurrentPageBots(p)}
                    />
                    {sortedRuns.length === 1 && (
                      <div style={{ fontSize: 12, opacity: 0.7 }}>
                        Chỉ có 1 kết quả – không đủ để so sánh.
                      </div>
                    )}
                  </div>
                </Section>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <Section title="Best Strategy">
                  {result.best ? (
                    <div
                      onClick={handleHighlightBest}
                      style={{
                        cursor: "pointer",
                        border: "1px solid rgba(56,189,248,0.4)",
                        borderRadius: 16,
                        padding: 16,
                        background: "linear-gradient(180deg, rgba(14,165,233,0.08), rgba(14,165,233,0.02))",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                        <span style={{ fontWeight: 700 }}>★ AI Best</span>
                        <span style={{ ...legendChip, padding: "6px 10px" }}>Click để highlight</span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontSize: 14 }}>
                        <div><b>Symbol:</b> {result.best.symbol}</div>
                        <div><b>TF:</b> {result.best.timeframe}</div>
                        <div><b>EMA Fast:</b> {result.best.params?.emaFast}</div>
                        <div><b>EMA Slow:</b> {result.best.params?.emaSlow}</div>
                      </div>
                      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
                        <Stat label="PF" value={fmt(result.best.summary?.profitFactor)} />
                        <Stat label="Winrate" value={fmtPct(result.best.summary?.winrate)} />
                        <Stat label="Max DD" value={fmtPct(result.best.summary?.maxDrawdownPct)} />
                        <Stat label="Score" value={fmt(result.best.summary?.score)} />
                      </div>
                    </div>
                  ) : (
                    <EmptyText text="Không có chiến lược đạt filter" />
                  )}
                </Section>

                <Section title="KPI nhanh">
                  <div style={pill}>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                      <Stat label="Net PnL" value={fmtPct(result.best?.summary?.netProfitPct)} />
                      <Stat label="Max DD" value={fmtPct(result.best?.summary?.maxDrawdownPct)} />
                      <Stat label="Winrate" value={fmtPct(result.best?.summary?.winrate)} />
                      <Stat label="PF" value={fmt(result.best?.summary?.profitFactor)} />
                    </div>
                  </div>
                </Section>

                <Section title="Equity (Properties)">
                  <div style={pill}>
                    {equityKPI ? (
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                        <Stat label="Final Equity" value={`$${Number(equityKPI.finalEq).toLocaleString()}`} />
                        <Stat label="Max Equity" value={`$${Number(equityKPI.maxEq).toLocaleString()}`} />
                        <Stat label="Min Equity" value={`$${Number(equityKPI.minEq).toLocaleString()}`} />
                        <Stat label="Equity Growth" value={`${equityKPI.growth.toFixed(2)}%`} />
                      </div>
                    ) : (
                      <EmptyText text="Chưa có equity curve" />
                    )}
                  </div>
                </Section>

                <Section title="KPI chi tiết">
                  <div style={{ ...pill, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                    <Stat label="Gross Profit" value={fmt(result.best?.summary?.grossProfit)} />
                    <Stat label="Gross Loss" value={fmt(result.best?.summary?.grossLoss)} />
                    <Stat label="Avg Trade" value={fmt(result.best?.summary?.avgTrade)} />
                    <Stat label="Avg Bars" value={fmt(result.best?.summary?.avgBars)} />
                    <Stat label="No. Trades" value={fmt(result.best?.summary?.totalTrades)} />
                    <Stat label="Sharpe (mock)" value={fmt(result.best?.summary?.sharpe || 0)} />
                    <Stat label="Sortino (mock)" value={fmt(result.best?.summary?.sortino || 0)} />
                    <Stat label="Exposure" value={fmtPct(result.best?.summary?.exposure || 0)} />
                  </div>
                </Section>
              </div>

              <Section title="Trades (lọc)">
                <>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 12,
                      flexWrap: "wrap",
                      marginBottom: 8,
                    }}
                  >
                    <div style={{ fontSize: 13, opacity: 0.8 }}>
                      Chọn bot để xem lịch sử trade
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                      }}
                    >
                      <span style={{ fontSize: 13, opacity: 0.7 }}>Bot</span>
                      <select
                        value={
                          tradesBotId ||
                          tradeBotOptions[0]?.value ||
                          selectedBot?.strategyId ||
                          ""
                        }
                        onChange={(e) => setTradesBotId(e.target.value)}
                        style={{
                          background: "rgba(255,255,255,0.05)",
                          color: "#e5e7eb",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: 8,
                          padding: "8px 10px",
                          minWidth: 200,
                        }}
                      >
                        {tradeBotOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <TradeFilterBar
                    filters={tradeFilters}
                    onChange={(f) => {
                      setTradeFilters(f);
                      setSelectedTrade(null);
                      setSelectedTradeIndex(null);
                    }}
                  />
                  <div style={{ height: 12 }} />
                  <StrategyTradesTable
                    trades={filteredTrades}
                    initialEquity={
                      tradesBot?.summary?.initialEquity ||
                      selectedBot?.summary?.initialEquity ||
                      initialCapital
                    }
                    selectedIndex={selectedTradeIndex}
                    onSelectTrade={(trade, idx) => {
                      setSelectedTrade(trade);
                      setSelectedTradeIndex(idx);
                    }}
                  />
                </>
              </Section>
            </div>
          ) : (
            <div style={{ marginTop: 12 }}>
              <EmptyText text="Nhập tham số và bấm chạy để xem kết quả từ AI Agent" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.12)",
  background: "rgba(255,255,255,0.06)",
  color: "#e5e7eb",
};

const primaryBtn = {
  padding: "12px 18px",
  borderRadius: 12,
  border: "none",
  background: "linear-gradient(110deg, #22c55e, #2dd4bf 60%, #3b82f6)",
  color: "#041023",
  fontWeight: 700,
  cursor: "pointer",
  boxShadow: "0 10px 30px rgba(34,197,94,0.2)",
};

const pill = {
  background: "linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))",
  borderRadius: 14,
  padding: 14,
  border: "1px solid rgba(255,255,255,0.06)",
};

const chartCard = {
  height: "60vh",
  minHeight: 600,
  borderRadius: 16,
  background: "linear-gradient(180deg, rgba(15,23,42,.92), rgba(2,6,23,.97))",
  padding: 12,
  border: "1px solid rgba(255,255,255,0.05)",
  boxShadow: "0 15px 40px rgba(0,0,0,0.35)",
  display: "flex",
  flexDirection: "column",
};

const palette = [
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
];

const panel = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
  background: "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))",
  borderRadius: 18,
  padding: 18,
  border: "1px solid rgba(255,255,255,0.08)",
  boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
};

const label = { fontSize: 13, opacity: 0.78, marginBottom: 4, display: "block" };

const controlBar = {
  display: "flex",
  justifyContent: "flex-start",
  alignItems: "center",
  gap: 12,
  padding: "12px 14px",
  borderRadius: 16,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.04)",
  margin: "12px 0 18px",
  flexWrap: "wrap",
};

const toggleLabel = { display: "flex", alignItems: "center", gap: 6, fontSize: 13, opacity: 0.85 };

const chip = {
  padding: "8px 12px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.12)",
  fontSize: 12,
};

const dateInput = {
  padding: "8px 10px",
  borderRadius: 10,
  border: "1px solid rgba(255,255,255,0.12)",
  background: "rgba(255,255,255,0.04)",
  color: "#e5e7eb",
};

const badgeBox = {
  padding: "8px 12px",
  borderRadius: 10,
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.08)",
  fontSize: 12,
};

const legendChip = {
  padding: "6px 12px",
  borderRadius: 999,
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.1)",
  fontSize: 12,
};

const inputColumn = {
  background: "rgba(255,255,255,0.02)",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 12,
  padding: 12,
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const groupTitle = {
  fontSize: 12,
  opacity: 0.7,
  marginBottom: 4,
  textTransform: "uppercase",
  letterSpacing: 0.6,
};

const statusCard = {
  padding: 12,
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03))",
};

const statusTitle = { fontSize: 12, opacity: 0.7 };
const statusValue = { fontSize: 16, fontWeight: 700 };
const statusNote = { fontSize: 12, opacity: 0.8 };

function EmptyText({ text = "Chưa có dữ liệu" }) {
  return (
    <div
      style={{
        height: "100%",
        minHeight: 160,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity: 0.6,
      }}
    >
      {text}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ padding: 10, borderRadius: 10, background: "rgba(255,255,255,0.02)" }}>
      <div style={{ fontSize: 12, opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600 }}>{value ?? "-"}</div>
    </div>
  );
}

function infoDot(text) {
  return (
    <span
      title={text}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 16,
        height: 16,
        borderRadius: "50%",
        border: "1px solid rgba(255,255,255,0.2)",
        fontSize: 10,
        marginLeft: 6,
        opacity: 0.8,
      }}
    >
      i
    </span>
  );
}

function Pagination({ current, total, pageSize, onChange }) {
  const totalPages = Math.max(1, Math.ceil((total || 0) / pageSize));
  const canPrev = current > 1;
  const canNext = current < totalPages;
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
      <button
        onClick={() => canPrev && onChange(current - 1)}
        disabled={!canPrev}
        style={{ ...legendChip, opacity: canPrev ? 1 : 0.5 }}
      >
        ⟨ Prev
      </button>
      <span style={{ fontSize: 12, opacity: 0.8 }}>
        Page {current} / {totalPages}
      </span>
      <button
        onClick={() => canNext && onChange(current + 1)}
        disabled={!canNext}
        style={{ ...legendChip, opacity: canNext ? 1 : 0.5 }}
      >
        Next ⟩
      </button>
    </div>
  );
}

function RangeInput({ label, inputKey, defaults, currentInputs, updateInput }) {
  const value = currentInputs[inputKey] || { start: defaults[0], end: defaults[1], step: defaults[2] };
  const rangeInputStyle = {
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.04)",
    color: "#e5e7eb",
    width: "100%",
    fontSize: 13,
  };
  return (
    <div>
      <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 6 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
        <input
          type="number"
          placeholder="start"
          step="any"
          value={value.start ?? defaults[0]}
          onChange={(e) => updateInput(inputKey, { ...value, start: Number(e.target.value) })}
          style={rangeInputStyle}
        />
        <input
          type="number"
          placeholder="end"
          step="any"
          value={value.end ?? defaults[1]}
          onChange={(e) => updateInput(inputKey, { ...value, end: Number(e.target.value) })}
          style={rangeInputStyle}
        />
        <input
          type="number"
          placeholder="step"
          step="any"
          value={value.step ?? defaults[2]}
          onChange={(e) => updateInput(inputKey, { ...value, step: Number(e.target.value) })}
          style={{ ...rangeInputStyle, opacity: 0.7 }}
        />
      </div>
    </div>
  );
}

function BoolInput({ label, inputKey, defaultVal, currentInputs, updateInput }) {
  const value = currentInputs[inputKey] ?? defaultVal;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input
        type="checkbox"
        checked={value === true}
        onChange={(e) => updateInput(inputKey, e.target.checked)}
        style={{ width: 18, height: 18, cursor: "pointer", accentColor: "#22c55e" }}
      />
      <label style={{ color: "#d1d5db", fontSize: 13 }}>{label}</label>
    </div>
  );
}

function fmt(v) {
  if (v === undefined || v === null) return "-";
  if (Number.isFinite(v)) return v.toFixed(2);
  return String(v);
}

function fmtPct(v) {
  if (v === undefined || v === null) return "-";
  return `${Number(v).toFixed(2)}%`;
}

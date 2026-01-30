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

// Color palette for timeframes (for multi-TF mode)
const TF_COLORS = {
  "5m": "#ef4444",   // red
  "15m": "#f59e0b",  // amber
  "30m": "#22c55e",  // green
  "1h": "#3b82f6",   // blue
  "4h": "#a855f7",   // purple
  "1d": "#ec4899",   // pink
};

// Sortable columns configuration
const SORTABLE_COLUMNS = {
  netProfit: { key: "netProfit", label: "Total PNL", defaultDir: "desc" },
  pf: { key: "pf", label: "PF", defaultDir: "desc" },
  winrate: { key: "winrate", label: "WR", defaultDir: "desc" },
  maxDD: { key: "maxDD", label: "DD", defaultDir: "asc" },  // Lower DD is better
  totalTrades: { key: "totalTrades", label: "Trades", defaultDir: "desc" },
  score: { key: "score", label: "Score", defaultDir: "desc" },
  source: { key: "source", label: "Source", defaultDir: "desc" },
};

export default function MemoryPage() {
  const [stats, setStats] = useState(null);
  const [genomes, setGenomes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedGenome, setSelectedGenome] = useState(null);
  const [sortBy, setSortBy] = useState("pf");  // Current sort column
  const [sortDir, setSortDir] = useState("desc");  // "asc" or "desc"
  const [overlayLimit, setOverlayLimit] = useState("5");
  const [visibleSeries, setVisibleSeries] = useState({});

  // Filters - declare these first so they can be used in selection state initialization
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("30m");
  const [selectedTimeframes, setSelectedTimeframes] = useState(["30m"]);  // Multi-TF mode
  const [multiTfMode, setMultiTfMode] = useState(false);  // Toggle for multi-TF view
  const [strategyType, setStrategyType] = useState("rf_st_rsi");  // Strategy type filter

  // Selection state for combine feature - persisted in localStorage (per strategy type)
  const [selectedGenomeIds, setSelectedGenomeIds] = useState(() => {
    // Load initial selection from localStorage - include strategyType in key
    try {
      const key = `quant_brain_selected_genomes_rf_st_rsi_BTCUSDT_30m`;  // Default values
      const saved = localStorage.getItem(key);
      if (saved) {
        const genomes = JSON.parse(saved);
        return new Set(genomes.map(g => g.genome_hash));
      }
    } catch (e) {
      console.warn("Failed to load initial selection:", e);
    }
    return new Set();
  });

  // Use ref to track previous genomes for rank comparison (avoid infinite loop)
  const prevGenomesRef = useRef([]);

  // Load previous rank history from localStorage
  const getRankHistoryKey = useCallback(() => {
    return `quant_brain_rank_history_${symbol}_${timeframe}`;
  }, [symbol, timeframe]);

  // Key for storing "new genomes" list (persists until newer genomes arrive)
  const getNewGenomesKey = useCallback(() => {
    return `quant_brain_new_genomes_${symbol}_${timeframe}`;
  }, [symbol, timeframe]);

  const loadRankHistory = useCallback(() => {
    try {
      const key = getRankHistoryKey();
      const saved = localStorage.getItem(key);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.warn("Failed to load rank history:", e);
    }
    return {};
  }, [getRankHistoryKey]);

  const saveRankHistory = useCallback((rankMap) => {
    try {
      const key = getRankHistoryKey();
      localStorage.setItem(key, JSON.stringify(rankMap));
    } catch (e) {
      console.warn("Failed to save rank history:", e);
    }
  }, [getRankHistoryKey]);

  // Load/save "new genomes" list
  const loadNewGenomesList = useCallback(() => {
    try {
      const key = getNewGenomesKey();
      const saved = localStorage.getItem(key);
      if (saved) {
        return new Set(JSON.parse(saved));
      }
    } catch (e) {
      console.warn("Failed to load new genomes list:", e);
    }
    return new Set();
  }, [getNewGenomesKey]);

  const saveNewGenomesList = useCallback((genomeSet) => {
    try {
      const key = getNewGenomesKey();
      localStorage.setItem(key, JSON.stringify([...genomeSet]));
    } catch (e) {
      console.warn("Failed to save new genomes list:", e);
    }
  }, [getNewGenomesKey]);

  // State to trigger refetch after reset
  const [resetTrigger, setResetTrigger] = useState(0);

  // Reset all tracking data - marks all current genomes as "NEW"
  const resetTracking = useCallback(() => {
    try {
      // Clear rank history
      localStorage.removeItem(getRankHistoryKey());
      // Clear new genomes list
      localStorage.removeItem(getNewGenomesKey());
      console.log("[Memory] Reset tracking data - all genomes will show as NEW");
      // Trigger refetch
      setResetTrigger((prev) => prev + 1);
    } catch (e) {
      console.warn("Failed to reset tracking:", e);
    }
  }, [getRankHistoryKey, getNewGenomesKey]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch stats first
      const statsRes = await getMemoryStats(strategyType);

      // Determine which timeframes to fetch
      const timeframesToFetch = multiTfMode ? selectedTimeframes : [timeframe];

      // Fetch genomes for all selected timeframes in parallel
      const genomesPromises = timeframesToFetch.map(tf =>
        getMemoryGenomes(symbol, tf, 100, strategyType)
      );
      const genomesResults = await Promise.all(genomesPromises);

      // Combine results from all timeframes
      let allGenomes = [];
      genomesResults.forEach((genomesRes, idx) => {
        if (genomesRes.success) {
          const tfGenomes = (genomesRes.genomes || []).map(g => ({
            ...g,
            timeframe: timeframesToFetch[idx],  // Ensure TF is set
          }));
          allGenomes = [...allGenomes, ...tfGenomes];
        }
      });

      if (statsRes.success) {
        setStats(statsRes);
      }
      if (allGenomes.length > 0 || !multiTfMode) {
        const newGenomes = allGenomes;

        // Sort new genomes by PF to get their new ranks
        const sortedNew = [...newGenomes].sort((a, b) => (b.pf || 0) - (a.pf || 0));

        // Load BASELINE rank history from localStorage
        // This stores the rank when each genome was FIRST seen
        // We compare current rank against this baseline to show movement
        const baselineRankHistory = loadRankHistory();

        // Load persisted "new genomes" list from localStorage
        // These genomes keep "NEW" badge until newer genomes arrive
        const persistedNewGenomesKey = `quant_brain_new_genomes_${symbol}_${timeframe}`;
        let persistedNewGenomes = new Set();
        try {
          const savedNewGenomes = localStorage.getItem(persistedNewGenomesKey);
          if (savedNewGenomes) {
            persistedNewGenomes = new Set(JSON.parse(savedNewGenomes));
          }
        } catch (e) {
          console.warn("Failed to load new genomes list:", e);
        }

        console.log(`[Memory] Loaded ${persistedNewGenomes.size} persisted NEW genomes from localStorage`);

        // ═══════════════════════════════════════════════════════
        // NEW LOGIC: Mark genomes as NEW based on Source (latest Run #)
        // Supports both simple sources (e.g., 21) and compound sources (e.g., "21.22")
        // Compound sources indicate a duplicate genome that was updated
        // ═══════════════════════════════════════════════════════

        // Helper function to extract the latest run number from source
        // Simple: 21 -> 21
        // Compound: "21.22" -> 22, "20.21.22" -> 22
        const getLatestRunFromSource = (source) => {
          if (source === null || source === undefined) return 0;
          const sourceStr = String(source);
          if (sourceStr.includes(".")) {
            const parts = sourceStr.split(".");
            return parseInt(parts[parts.length - 1], 10) || 0;
          }
          return parseInt(sourceStr, 10) || 0;
        };

        // Check if source is compound (has been updated as duplicate)
        const isCompoundSource = (source) => {
          if (source === null || source === undefined) return false;
          return String(source).includes(".");
        };

        // Find the highest run number across all sources
        const allLatestRuns = sortedNew.map(g => getLatestRunFromSource(g.source));
        const maxRunNumber = Math.max(...allLatestRuns, 0);
        console.log(`[Memory] Max run number: ${maxRunNumber}`);

        // Genomes are "NEW" if:
        // 1. Their latest run number = maxRunNumber (from latest run), OR
        // 2. They have a compound source (e.g., "21.22") indicating they were updated as duplicate
        const latestRunGenomeHashes = sortedNew
          .filter((g) => {
            if (!g.genome_hash) return false;
            const latestRun = getLatestRunFromSource(g.source);
            const isFromLatestRun = latestRun === maxRunNumber;
            const isDuplicateUpdated = isCompoundSource(g.source);
            return isFromLatestRun || isDuplicateUpdated;
          })
          .map((g) => g.genome_hash);

        // Log compound sources for debugging
        const compoundSources = sortedNew.filter(g => isCompoundSource(g.source));
        console.log(`[Memory] Found ${latestRunGenomeHashes.length} genomes from latest run or with compound source`);
        console.log(`[Memory] Compound sources (duplicates): ${compoundSources.length}`,
          compoundSources.map(g => ({ hash: g.genome_hash?.slice(0, 8), source: g.source })));

        // Also find TRULY NEW genomes (not in baseline history at all)
        const trulyNewGenomeHashes = sortedNew
          .filter((g) => g.genome_hash && baselineRankHistory[g.genome_hash] === undefined)
          .map((g) => g.genome_hash);
        const hasTrulyNewGenomes = trulyNewGenomeHashes.length > 0;

        console.log(`[Memory] Found ${trulyNewGenomeHashes.length} truly new genomes (not in baseline)`);

        // Update baseline for truly new genomes
        if (hasTrulyNewGenomes) {
          const updatedBaseline = { ...baselineRankHistory };
          sortedNew.forEach((g, idx) => {
            const currentRank = idx + 1;
            // Only add NEW genomes to baseline, don't update existing
            if (g.genome_hash && updatedBaseline[g.genome_hash] === undefined) {
              updatedBaseline[g.genome_hash] = currentRank;
            }
          });
          saveRankHistory(updatedBaseline);
        }

        // Determine which genomes should show NEW badge
        // Combine: genomes from latest run OR truly new genomes
        const currentNewGenomes = new Set([...latestRunGenomeHashes, ...trulyNewGenomeHashes]);

        // Save to localStorage for persistence
        localStorage.setItem(persistedNewGenomesKey, JSON.stringify([...currentNewGenomes]));
        console.log(`[Memory] Total NEW genomes: ${currentNewGenomes.size} (${latestRunGenomeHashes.length} from Run #${maxRunNumber}, ${trulyNewGenomeHashes.length} truly new)`);

        // Use updated baseline
        const finalBaseline = hasTrulyNewGenomes ? loadRankHistory() : baselineRankHistory;

        // Add rank info to genomes
        // A genome is "NEW" if it's from the latest run OR truly new
        const genomesWithRank = sortedNew.map((g, idx) => {
          const currentRank = idx + 1;
          const baselineRank = finalBaseline[g.genome_hash];
          const isNew = currentNewGenomes.has(g.genome_hash);

          return {
            ...g,
            currentRank: currentRank,
            // Always use baseline rank for comparison (even for NEW genomes)
            previousRank: baselineRank !== undefined ? baselineRank : null,
            isNew: isNew,
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
  }, [symbol, timeframe, multiTfMode, selectedTimeframes, strategyType, loadRankHistory, saveRankHistory, loadNewGenomesList, saveNewGenomesList]);

  useEffect(() => {
    fetchData();
  }, [fetchData, resetTrigger]);

  // Reload selection from localStorage when symbol/timeframe/strategyType changes
  useEffect(() => {
    try {
      const key = `quant_brain_selected_genomes_${strategyType}_${symbol}_${timeframe}`;
      const saved = localStorage.getItem(key);
      if (saved) {
        const genomes = JSON.parse(saved);
        setSelectedGenomeIds(new Set(genomes.map(g => g.genome_hash)));
        console.log(`Loaded ${genomes.length} selected genomes for ${strategyType}/${symbol}/${timeframe}`);
      } else {
        setSelectedGenomeIds(new Set());
      }
    } catch (e) {
      console.warn("Failed to reload selection:", e);
      setSelectedGenomeIds(new Set());
    }
  }, [symbol, timeframe, strategyType]);

  // Handle column sort click
  const handleSort = useCallback((columnKey) => {
    if (sortBy === columnKey) {
      // Toggle direction if same column
      setSortDir(prev => prev === "desc" ? "asc" : "desc");
    } else {
      // New column - use default direction
      setSortBy(columnKey);
      setSortDir(SORTABLE_COLUMNS[columnKey]?.defaultDir || "desc");
    }
  }, [sortBy]);

  // Sort genomes by selected column and direction
  const sortedGenomes = useMemo(() => {
    const arr = [...genomes];

    // Helper to extract the latest run number from source (handles compound sources)
    const getLatestRunFromSource = (source) => {
      if (source === null || source === undefined) return 0;
      const sourceStr = String(source);
      if (sourceStr.includes(".")) {
        const parts = sourceStr.split(".");
        return parseFloat(parts.join(".")) || 0;  // Parse as float to preserve order: 21.22 > 21.21 > 21
      }
      return parseFloat(sourceStr) || 0;
    };

    arr.sort((a, b) => {
      let aVal = a[sortBy] ?? 0;
      let bVal = b[sortBy] ?? 0;

      // Handle special cases
      if (sortBy === "netProfit") {
        aVal = a.netProfit ?? 0;
        bVal = b.netProfit ?? 0;
      } else if (sortBy === "source") {
        // Special handling for source column - convert to comparable numbers
        aVal = getLatestRunFromSource(a.source);
        bVal = getLatestRunFromSource(b.source);
      }

      // Compare
      if (sortDir === "desc") {
        return bVal - aVal;
      } else {
        return aVal - bVal;
      }
    });

    return arr.map((g, idx) => ({ ...g, displayRank: idx + 1 }));
  }, [genomes, sortBy, sortDir]);

  // Key for storing selected genomes in localStorage (for Dashboard to read)
  // Include strategyType to separate selections between Combined and Long Only
  const getSelectedGenomesKey = useCallback(() => {
    return `quant_brain_selected_genomes_${strategyType}_${symbol}_${timeframe}`;
  }, [strategyType, symbol, timeframe]);

  // Save selected genomes to localStorage whenever selection changes
  const saveSelectedGenomesToStorage = useCallback((selectedIds) => {
    try {
      const key = getSelectedGenomesKey();
      const selectedGenomesList = sortedGenomes
        .filter(g => selectedIds.has(g.genome_hash))
        .map(g => ({
          genome_hash: g.genome_hash,
          genome: g.genome,
          results: {
            pf: g.pf,
            winrate: g.winrate,
            max_dd: g.maxDD,
            net_profit: g.netProfit,
            net_profit_pct: g.netProfitPct,
            total_trades: g.totalTrades,
            score: g.score,
          },
          equity_curve: g.equityCurve || [],
          symbol: g.symbol || symbol,
          timeframe: g.timeframe || timeframe,
          source: g.source || 0,  // Include source for tracking
        }));
      localStorage.setItem(key, JSON.stringify(selectedGenomesList));
      console.log(`Saved ${selectedGenomesList.length} selected genomes to localStorage`);
    } catch (e) {
      console.warn("Failed to save selected genomes:", e);
    }
  }, [getSelectedGenomesKey, sortedGenomes, symbol, timeframe]);

  // Selection handlers
  const toggleGenomeSelection = useCallback((genomeHash) => {
    setSelectedGenomeIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(genomeHash)) {
        newSet.delete(genomeHash);
      } else {
        newSet.add(genomeHash);
      }
      // Save to localStorage for Dashboard to read
      saveSelectedGenomesToStorage(newSet);
      return newSet;
    });
  }, [saveSelectedGenomesToStorage]);

  const selectAllGenomes = useCallback(() => {
    const allHashes = sortedGenomes.map(g => g.genome_hash).filter(Boolean);
    const newSet = new Set(allHashes);
    setSelectedGenomeIds(newSet);
    saveSelectedGenomesToStorage(newSet);
  }, [sortedGenomes, saveSelectedGenomesToStorage]);

  const clearSelection = useCallback(() => {
    setSelectedGenomeIds(new Set());
    // Clear from localStorage too
    try {
      const key = getSelectedGenomesKey();
      localStorage.removeItem(key);
    } catch (e) {
      console.warn("Failed to clear selected genomes:", e);
    }
  }, [getSelectedGenomesKey]);

  const isAllSelected = useMemo(() => {
    const validGenomes = sortedGenomes.filter(g => g.genome_hash);
    return validGenomes.length > 0 && selectedGenomeIds.size === validGenomes.length;
  }, [sortedGenomes, selectedGenomeIds]);

  const isSomeSelected = useMemo(() => {
    return selectedGenomeIds.size > 0 && !isAllSelected;
  }, [selectedGenomeIds, isAllSelected]);

  // Get selected genomes for combine
  const selectedGenomesForCombine = useMemo(() => {
    return sortedGenomes.filter(g => selectedGenomeIds.has(g.genome_hash));
  }, [sortedGenomes, selectedGenomeIds]);

  // Handle combine selected genomes
  const handleCombineSelected = useCallback(() => {
    if (selectedGenomesForCombine.length < 2) {
      alert("Cần chọn ít nhất 2 genomes để combine!");
      return;
    }
    // TODO: Call combine API here
    console.log("Combining genomes:", selectedGenomesForCombine.map(g => g.genome_hash));
    alert(`Combine ${selectedGenomesForCombine.length} genomes (chức năng đang phát triển)`);
  }, [selectedGenomesForCombine]);

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

  // Rank change indicator component
  const RankChangeIndicator = ({ genome }) => {
    const hasBaseline = genome.previousRank !== null;
    const change = hasBaseline ? genome.previousRank - genome.currentRank : 0;

    // NEW genome - show NEW badge + rank change if available
    if (genome.isNew) {
      // NEW without baseline - just show NEW
      if (!hasBaseline || change === 0) {
        return <span style={newBadgeStyle}>NEW</span>;
      }
      // NEW with rank change - show both
      return (
        <div style={rankChangeContainer}>
          <span style={{ ...newBadgeStyle, marginRight: 4 }}>NEW</span>
          {change > 0 ? (
            <>
              <span style={{ ...arrowStyle, color: "#00C853" }}>▲</span>
              <span style={{ ...changeValueStyle, color: "#00C853" }}>{change}</span>
            </>
          ) : (
            <>
              <span style={{ ...arrowStyle, color: "#FF5252" }}>▼</span>
              <span style={{ ...changeValueStyle, color: "#FF5252" }}>{Math.abs(change)}</span>
            </>
          )}
        </div>
      );
    }

    // Not NEW, no baseline - show dash
    if (!hasBaseline) {
      return <span style={noChangeStyle}>—</span>;
    }

    // No change
    if (change === 0) {
      return <span style={noChangeStyle}>—</span>;
    }

    // Rank improved (moved UP in leaderboard)
    if (change > 0) {
      return (
        <div style={rankChangeContainer}>
          <span style={{ ...arrowStyle, color: "#00C853" }}>▲</span>
          <span style={{ ...changeValueStyle, color: "#00C853" }}>{change}</span>
        </div>
      );
    }

    // Rank dropped (moved DOWN in leaderboard)
    return (
      <div style={rankChangeContainer}>
        <span style={{ ...arrowStyle, color: "#FF5252" }}>▼</span>
        <span style={{ ...changeValueStyle, color: "#FF5252" }}>{Math.abs(change)}</span>
      </div>
    );
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

          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            {/* Strategy Type filter */}
            <select
              value={strategyType}
              onChange={(e) => setStrategyType(e.target.value)}
              style={{ ...selectStyle, minWidth: 160 }}
            >
              <option value="rf_st_rsi">RF+ST+RSI (Long Only)</option>
              <option value="rf_st_rsi_combined">RF+ST+RSI Combined</option>
            </select>

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

            {/* Multi-TF Toggle */}
            <button
              onClick={() => {
                setMultiTfMode(!multiTfMode);
                if (!multiTfMode && !selectedTimeframes.includes(timeframe)) {
                  setSelectedTimeframes([timeframe]);
                }
              }}
              style={{
                ...chipBtn,
                background: multiTfMode ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.06)",
                borderColor: multiTfMode ? "#3b82f6" : "rgba(255,255,255,0.1)",
                color: multiTfMode ? "#3b82f6" : "#e5e7eb",
              }}
            >
              {multiTfMode ? "📊 Multi-TF ON" : "Multi-TF"}
            </button>

            {/* Single Timeframe filter (when multi-TF is OFF) */}
            {!multiTfMode && (
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                style={selectStyle}
              >
                {(stats?.timeframes ? Object.keys(stats.timeframes) : ["30m"]).map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            )}

            {/* Multi-Timeframe checkboxes (when multi-TF is ON) */}
            {multiTfMode && (
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {(stats?.timeframes ? Object.keys(stats.timeframes) : ["5m", "15m", "30m", "1h", "4h"]).map((tf) => {
                  const isSelected = selectedTimeframes.includes(tf);
                  return (
                    <button
                      key={tf}
                      onClick={() => {
                        if (isSelected) {
                          // Don't allow deselecting all
                          if (selectedTimeframes.length > 1) {
                            setSelectedTimeframes(selectedTimeframes.filter(t => t !== tf));
                          }
                        } else {
                          setSelectedTimeframes([...selectedTimeframes, tf]);
                        }
                      }}
                      style={{
                        ...tfChipBtn,
                        background: isSelected ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.04)",
                        borderColor: isSelected ? "#22c55e" : "rgba(255,255,255,0.1)",
                        color: isSelected ? "#22c55e" : "#94a3b8",
                      }}
                    >
                      {isSelected && "✓ "}{tf}
                    </button>
                  );
                })}
              </div>
            )}

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

        {/* Main content - Stacked layout (Chart on top, Table below) */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* TOP: Equity Chart - Full width */}
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

            <div style={chartContainerFull}>
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

          {/* BOTTOM: Leaderboard - Full width */}
          <div style={panel}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 16 }}>
                Bảng xếp hạng Genomes
                {multiTfMode && selectedTimeframes.length > 1 && (
                  <span style={{ fontSize: 12, fontWeight: 400, opacity: 0.7, marginLeft: 8 }}>
                    ({selectedTimeframes.join(", ")})
                  </span>
                )}
              </h2>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 12, opacity: 0.6 }}>
                  Sorted by <span style={{ color: "#22c55e", fontWeight: 600 }}>{SORTABLE_COLUMNS[sortBy]?.label || sortBy}</span>
                  {sortDir === "desc" ? " ▼" : " ▲"}
                </span>
                <span style={{ fontSize: 11, opacity: 0.5 }}>
                  {sortedGenomes.length} genomes
                </span>
              </div>
            </div>

            {/* Selection Action Bar */}
            {selectedGenomeIds.size > 0 && (
              <div style={selectionBar}>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <span style={{ fontWeight: 600, color: "#22c55e" }}>
                    Selected: {selectedGenomeIds.size}
                  </span>
                  <button
                    onClick={handleCombineSelected}
                    disabled={selectedGenomeIds.size < 2}
                    style={{
                      ...combineBtn,
                      opacity: selectedGenomeIds.size < 2 ? 0.5 : 1,
                      cursor: selectedGenomeIds.size < 2 ? "not-allowed" : "pointer",
                    }}
                  >
                    🧬 Combine Selected
                  </button>
                  <button onClick={clearSelection} style={clearBtn}>
                    ✕ Clear
                  </button>
                </div>
                <span style={{ fontSize: 11, opacity: 0.6 }}>
                  {selectedGenomeIds.size < 2 ? "Chọn ít nhất 2 genomes để combine" : "Sẵn sàng combine"}
                </span>
              </div>
            )}

            {/* Table */}
            <div style={tableContainerFull}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "rgba(255,255,255,0.05)" }}>
                    {/* Checkbox column */}
                    <th style={{ ...thStyle, width: 40, textAlign: "center" }}>
                      <label style={checkboxLabel}>
                        <input
                          type="checkbox"
                          checked={isAllSelected}
                          ref={el => {
                            if (el) el.indeterminate = isSomeSelected;
                          }}
                          onChange={(e) => {
                            if (e.target.checked) {
                              selectAllGenomes();
                            } else {
                              clearSelection();
                            }
                          }}
                          style={checkboxInput}
                        />
                        <span style={{
                          ...checkboxCustom,
                          background: isAllSelected ? "#22c55e" : isSomeSelected ? "#3b82f6" : "rgba(255,255,255,0.1)",
                          borderColor: isAllSelected || isSomeSelected ? "transparent" : "rgba(255,255,255,0.3)",
                        }}>
                          {isAllSelected && "✓"}
                          {isSomeSelected && "−"}
                        </span>
                      </label>
                    </th>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>▲▼</th>
                    <th style={thStyle}>TF</th>
                    <SortableHeader
                      columnKey="netProfit"
                      label="Total PNL"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="pf"
                      label="PF"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="winrate"
                      label="WR"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="maxDD"
                      label="DD"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="totalTrades"
                      label="Trades"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="score"
                      label="Score"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      columnKey="source"
                      label="Source"
                      sortBy={sortBy}
                      sortDir={sortDir}
                      onSort={handleSort}
                    />
                    <th style={thStyle}>Range</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedGenomes.map((g, idx) => {
                    const isDetailSelected = selectedGenome?.genome_hash === g.genome_hash;
                    const isChecked = selectedGenomeIds.has(g.genome_hash);
                    const isTop1 = idx === 0;

                    return (
                      <tr
                        key={g.genome_hash || idx}
                        style={{
                          cursor: "pointer",
                          background: isChecked
                            ? "rgba(34,197,94,0.15)"
                            : isDetailSelected
                            ? "rgba(59,130,246,0.2)"
                            : isTop1
                            ? "rgba(255,215,0,0.08)"
                            : idx % 2 === 0
                            ? "transparent"
                            : "rgba(255,255,255,0.02)",
                          borderBottom: "1px solid rgba(255,255,255,0.05)",
                          borderLeft: isChecked
                            ? "3px solid #22c55e"
                            : isTop1
                            ? "3px solid #fbbf24"
                            : "3px solid transparent",
                        }}
                      >
                        {/* Row checkbox */}
                        <td style={{ ...tdStyle, textAlign: "center", width: 40 }}>
                          <label
                            style={checkboxLabel}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => toggleGenomeSelection(g.genome_hash)}
                              style={checkboxInput}
                            />
                            <span style={{
                              ...checkboxCustom,
                              background: isChecked ? "#22c55e" : "rgba(255,255,255,0.1)",
                              borderColor: isChecked ? "transparent" : "rgba(255,255,255,0.3)",
                            }}>
                              {isChecked && "✓"}
                            </span>
                          </label>
                        </td>
                        <td
                          style={tdStyle}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {isTop1 && <span style={{ marginRight: 4 }}>⭐</span>}
                          #{idx + 1}
                        </td>
                        <td
                          style={{ ...tdStyle, textAlign: "center" }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          <RankChangeIndicator genome={g} />
                        </td>
                        <td
                          style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11 }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          <span style={{
                            color: TF_COLORS[g.timeframe] || "#e5e7eb",
                            fontWeight: multiTfMode ? 600 : 400,
                            padding: multiTfMode ? "2px 6px" : 0,
                            background: multiTfMode ? `${TF_COLORS[g.timeframe]}15` : "transparent",
                            borderRadius: 4,
                          }}>
                            {g.timeframe || timeframe}
                          </span>
                        </td>
                        <td
                          style={tdStyle}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          <div style={{ color: (g.netProfitPct || 0) >= 0 ? "#22c55e" : "#ef4444" }}>
                            {formatMoney(g.netProfit)}
                          </div>
                          <div style={{ fontSize: 10, opacity: 0.7 }}>
                            ({formatPct(g.netProfitPct)})
                          </div>
                        </td>
                        <td
                          style={{ ...tdStyle, color: (g.pf || 0) >= 1.5 ? "#22c55e" : "#e5e7eb" }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {g.pf?.toFixed(2) || "-"}
                        </td>
                        <td
                          style={tdStyle}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {formatPct(g.winrate)}
                        </td>
                        <td
                          style={{ ...tdStyle, color: "#f87171" }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {formatPct(g.maxDD)}
                        </td>
                        <td
                          style={tdStyle}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {g.totalTrades || "-"}
                        </td>
                        <td
                          style={{ ...tdStyle, fontWeight: 600, color: "#22c55e" }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          {g.score?.toFixed(2) || "-"}
                        </td>
                        <td
                          style={{ ...tdStyle, textAlign: "center" }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
                          <span style={{
                            ...sourceStyle,
                            // Highlight compound sources (duplicates) with different color
                            background: String(g.source || "").includes(".")
                              ? "rgba(251,191,36,0.2)"
                              : "rgba(59,130,246,0.15)",
                            color: String(g.source || "").includes(".")
                              ? "#fbbf24"
                              : "#60a5fa",
                          }}>
                            Run #{g.source || 1}
                          </span>
                        </td>
                        <td
                          style={{ ...tdStyle, fontSize: 10, opacity: 0.8 }}
                          onClick={() => setSelectedGenome(isDetailSelected ? null : g)}
                        >
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

// Helper component - Sortable Header
function SortableHeader({ columnKey, label, sortBy, sortDir, onSort }) {
  const isActive = sortBy === columnKey;

  return (
    <th
      style={{
        ...thStyle,
        cursor: "pointer",
        userSelect: "none",
        transition: "background 0.15s",
      }}
      onClick={() => onSort(columnKey)}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = "rgba(255,255,255,0.08)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between" }}>
        <span>{label}</span>
        <div style={{
          display: "flex",
          flexDirection: "column",
          fontSize: 11,
          lineHeight: 1,
          gap: 1,
          padding: "2px 4px",
          borderRadius: 4,
          background: isActive ? "rgba(34,197,94,0.15)" : "transparent",
        }}>
          <span style={{
            color: isActive && sortDir === "asc" ? "#22c55e" : "#64748b",
            fontWeight: isActive && sortDir === "asc" ? 700 : 400,
            fontSize: 10,
          }}>▲</span>
          <span style={{
            color: isActive && sortDir === "desc" ? "#22c55e" : "#64748b",
            fontWeight: isActive && sortDir === "desc" ? 700 : 400,
            fontSize: 10,
          }}>▼</span>
        </div>
      </div>
    </th>
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

const resetBtn = {
  padding: "10px 16px",
  borderRadius: 8,
  border: "1px solid rgba(251,191,36,0.3)",
  background: "rgba(251,191,36,0.1)",
  color: "#fbbf24",
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 13,
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

const tfChipBtn = {
  padding: "5px 10px",
  borderRadius: 6,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.1)",
  color: "#94a3b8",
  cursor: "pointer",
  fontSize: 11,
  fontWeight: 500,
  transition: "all 0.15s ease",
};

const chartContainer = {
  height: 400,
  borderRadius: 12,
  background: "rgba(0,0,0,0.2)",
  padding: 12,
};

// Full width chart container (for new layout)
const chartContainerFull = {
  height: 350,
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

// Full width table container (for new layout)
const tableContainerFull = {
  maxHeight: 450,
  overflowY: "auto",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 8,
};

// Source column style
const sourceStyle = {
  display: "inline-block",
  padding: "3px 8px",
  borderRadius: 4,
  background: "rgba(59,130,246,0.15)",
  color: "#60a5fa",
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: "0.3px",
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

// Rank Change Indicator Styles
const rankChangeContainer = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  gap: 2,
  minWidth: 28,
};

const arrowStyle = {
  fontSize: 14,
  fontWeight: 700,
  lineHeight: 1,
};

const changeValueStyle = {
  fontSize: 11,
  fontWeight: 600,
  lineHeight: 1,
};

const newBadgeStyle = {
  background: "linear-gradient(135deg, #fbbf24, #f59e0b)",
  color: "#000",
  padding: "3px 8px",
  borderRadius: 4,
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: "0.5px",
};

const noChangeStyle = {
  color: "#64748b",
  fontSize: 14,
  fontWeight: 500,
};

// Selection bar styles
const selectionBar = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "12px 16px",
  marginBottom: 12,
  background: "rgba(34,197,94,0.1)",
  border: "1px solid rgba(34,197,94,0.3)",
  borderRadius: 8,
};

const combineBtn = {
  padding: "8px 16px",
  borderRadius: 6,
  border: "1px solid #22c55e",
  background: "linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1))",
  color: "#22c55e",
  fontWeight: 600,
  fontSize: 13,
};

const clearBtn = {
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid rgba(255,255,255,0.2)",
  background: "rgba(255,255,255,0.05)",
  color: "#94a3b8",
  cursor: "pointer",
  fontSize: 12,
};

// Checkbox styles
const checkboxLabel = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
};

const checkboxInput = {
  position: "absolute",
  opacity: 0,
  width: 0,
  height: 0,
};

const checkboxCustom = {
  width: 18,
  height: 18,
  borderRadius: 4,
  border: "2px solid rgba(255,255,255,0.3)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 12,
  fontWeight: 700,
  color: "#fff",
  transition: "all 0.15s ease",
};

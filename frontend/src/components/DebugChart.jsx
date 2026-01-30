import { useEffect, useRef, useState } from "react";
import { createChart, CandlestickSeries, LineSeries } from "lightweight-charts";

// Custom Markers Primitive for Lightweight Charts v5
class MarkersPrimitive {
  constructor(markers) {
    this._markers = markers || [];
    this._chart = null;
    this._series = null;
  }

  attached({ chart, series }) {
    this._chart = chart;
    this._series = series;
  }

  detached() {
    this._chart = null;
    this._series = null;
  }

  updateMarkers(markers) {
    this._markers = markers || [];
    if (this._chart) {
      this._chart.applyOptions({});
    }
  }

  paneViews() {
    return [new MarkersPaneView(this._markers, this._series, this._chart)];
  }
}

class MarkersPaneView {
  constructor(markers, series, chart) {
    this._markers = markers;
    this._series = series;
    this._chart = chart;
  }

  renderer() {
    return new MarkersRenderer(this._markers, this._series, this._chart);
  }

  zOrder() {
    return "top";
  }
}

class MarkersRenderer {
  constructor(markers, series, chart) {
    this._markers = markers;
    this._series = series;
    this._chart = chart;
  }

  draw(target) {
    target.useBitmapCoordinateSpace((scope) => {
      const ctx = scope.context;
      const timeScale = this._chart.timeScale();
      const ratio = scope.horizontalPixelRatio;

      // Group markers by time+position to handle overlapping markers
      // Key: "time_position", value: array of markers at that time/position
      const markerGroups = {};
      for (const marker of this._markers) {
        const key = `${marker.time}_${marker.position}`;
        if (!markerGroups[key]) {
          markerGroups[key] = [];
        }
        markerGroups[key].push(marker);
      }

      // Draw each group with horizontal offset for multiple markers
      for (const key of Object.keys(markerGroups)) {
        const group = markerGroups[key];
        const groupSize = group.length;

        for (let idx = 0; idx < groupSize; idx++) {
          const marker = group[idx];
          const x = timeScale.timeToCoordinate(marker.time);
          if (x === null) continue;

          // Get price coordinate for the marker
          const priceY = this._series.priceToCoordinate(marker.price || 0);
          if (priceY === null) continue;

          // Calculate horizontal offset for multiple markers at same time
          // Spread them out horizontally: -20, 0, +20 pixels for 2 markers
          const spreadWidth = 25 * ratio;
          let horizontalOffset = 0;
          if (groupSize > 1) {
            // Center the group: for 2 markers, offsets are -12.5 and +12.5
            horizontalOffset = (idx - (groupSize - 1) / 2) * spreadWidth;
          }

          const pixelX = x * ratio + horizontalOffset;
          const pixelY = priceY * scope.verticalPixelRatio;

          ctx.save();

          const color = marker.color || "#2962FF";
          // Trade markers (Entry/TP/SL) are 2x larger than debug markers (RF/ST)
          const isTradeMarker = marker.isTradeMarker === true;
          const arrowSize = isTradeMarker ? 12 * ratio : 6 * ratio;
          const offset = isTradeMarker ? 15 * ratio : 10 * ratio;
          const fontSize = isTradeMarker ? 12 * ratio : 10 * ratio;

          // Add vertical offset for stacked markers (second marker goes further out)
          const stackOffset = idx * 20 * ratio;

          if (marker.position === "belowBar") {
            // Arrow pointing up (Entry) - below the candle
            const arrowY = pixelY + offset + stackOffset;

            // Draw arrow
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(pixelX, arrowY);
            ctx.lineTo(pixelX - arrowSize, arrowY + arrowSize * 1.5);
            ctx.lineTo(pixelX + arrowSize, arrowY + arrowSize * 1.5);
            ctx.closePath();
            ctx.fill();

            // Draw text label
            if (marker.text) {
              ctx.font = `bold ${fontSize}px Arial`;
              ctx.textAlign = "center";
              ctx.fillStyle = color;
              ctx.fillText(marker.text, pixelX, arrowY + arrowSize * 1.5 + 11 * ratio);
            }
          } else {
            // Arrow pointing down (Exit) - above the candle
            const arrowY = pixelY - offset - stackOffset;

            // Draw arrow
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(pixelX, arrowY);
            ctx.lineTo(pixelX - arrowSize, arrowY - arrowSize * 1.5);
            ctx.lineTo(pixelX + arrowSize, arrowY - arrowSize * 1.5);
            ctx.closePath();
            ctx.fill();

            // Draw text label
            if (marker.text) {
              ctx.font = `bold ${fontSize}px Arial`;
              ctx.textAlign = "center";
              ctx.fillStyle = color;
              ctx.fillText(marker.text, pixelX, arrowY - arrowSize * 1.5 - 5 * ratio);
            }
          }

          ctx.restore();
        }
      }
    });
  }
}

export default function DebugChart({ candles, indicators, markers, trades, debugMarkers, visibility }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef({});
  const [error, setError] = useState(null);
  const [ohlc, setOhlc] = useState(null);

  // Default visibility - all hidden, user clicks to show
  const vis = visibility || {
    rangeFilter: false,
    superTrend: false,
    rfSlShort: false,
    stSlShort: false,
    stTpRsiShort: false,
    long: false,
    short: false,
    tp: false,
    exitSl: false,
  };

  useEffect(() => {
    if (!containerRef.current || !candles?.length) return;

    setError(null);

    try {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }

      const chart = createChart(containerRef.current, {
        layout: {
          background: { type: "solid", color: "#1a1a2e" },
          textColor: "#d1d5db",
        },
        grid: {
          vertLines: { color: "#2d2d44" },
          horzLines: { color: "#2d2d44" },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: "#2d2d44",
        },
        timeScale: {
          borderColor: "#2d2d44",
          timeVisible: true,
          secondsVisible: false,
          fixLeftEdge: true,
          fixRightEdge: true,
          barSpacing: 6,  // Consistent bar spacing
        },
        width: containerRef.current.clientWidth,
        height: 1000,
      });

      chartRef.current = chart;

      // Candlestick series
      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderDownColor: "#ef4444",
        borderUpColor: "#22c55e",
        wickDownColor: "#ef4444",
        wickUpColor: "#22c55e",
      });

      candleSeries.setData(candles);

      // Build candle lookup for price at time
      const candleMap = {};
      candles.forEach((c) => {
        candleMap[c.time] = c;
      });

      // Process trade markers with price data (Entry/TP/SL)
      // These are larger and use distinct colors from RF/ST markers
      if (markers?.length) {
        const markersWithPrice = markers
          .map((m) => {
            const candle = candleMap[m.time];
            let price = candle?.close || candle?.low || 0;
            if (m.position === "belowBar") {
              price = candle?.low || price;
            } else {
              price = candle?.high || price;
            }

            // Determine color and category based on marker type
            let color = m.color;
            let category = "other";
            const text = m.text?.toLowerCase() || "";
            if (text.includes("entry") || text.includes("long") && m.position === "belowBar") {
              color = "#00d4ff"; // Cyan for Long Entry
              category = "long";
            } else if (text.includes("short") && m.position === "aboveBar" && !text.includes("tp") && !text.includes("sl")) {
              color = "#ff00ff"; // Magenta for Short Entry
              category = "short";
            } else if (text.includes("tp") || text.includes("take")) {
              color = "#ffd700"; // Gold for Take Profit
              category = "tp";
            } else if (text.includes("sl") || text.includes("stop") || text.includes("exit")) {
              color = "#dc143c"; // Crimson for Stop Loss
              category = "exitSl";
            }

            return { ...m, price, color, isTradeMarker: true, category };
          })
          // Filter based on visibility
          .filter((m) => {
            if (m.category === "long" && !vis.long) return false;
            if (m.category === "short" && !vis.short) return false;
            if (m.category === "tp" && !vis.tp) return false;
            if (m.category === "exitSl" && !vis.exitSl) return false;
            return true;
          });

        const markersPrimitive = new MarkersPrimitive(markersWithPrice);
        candleSeries.attachPrimitive(markersPrimitive);
      }

      // Process debug markers (Buy/Sell flip signals) - like TradingView labels
      // Separate RF and ST markers with different colors and text labels
      if (debugMarkers?.length) {
        const debugMarkersFormatted = debugMarkers
          .map((m) => {
            const candle = candleMap[m.time];
            // BUY markers go below the candle (at low), SELL markers go above (at high)
            let price = m.price;
            if (!price && candle) {
              price = m.type === "BUY" ? candle.low : candle.high;
            }

            // Use source (RF or ST) to differentiate markers visually
            const source = m.source || "RF";
            const isRF = source === "RF";
            const isBuy = m.type === "BUY";

            // Different colors for RF vs ST
            // RF: Green/Red (original)
            // ST: Blue/Orange (distinct)
            let color;
            if (isRF) {
              color = isBuy ? "#22c55e" : "#ef4444"; // Green/Red for RF
            } else {
              color = isBuy ? "#3b82f6" : "#f97316"; // Blue/Orange for ST
            }

            // Add source tag to text so both are visible when overlapping
            const text = isBuy ? `Buy ${source}` : `Sell ${source}`;

            return {
              time: m.time,
              price: price || 0,
              position: isBuy ? "belowBar" : "aboveBar",
              color: color,
              text: text,
              shape: isBuy ? "arrowUp" : "arrowDown",
              source: source, // Pass through for potential further use
            };
          })
          // Filter based on visibility - RF and ST flip markers
          .filter((m) => {
            if (m.source === "RF" && !vis.rangeFilter) return false;
            if (m.source === "ST" && !vis.superTrend) return false;
            return true;
          });

        const debugMarkersPrimitive = new MarkersPrimitive(debugMarkersFormatted);
        candleSeries.attachPrimitive(debugMarkersPrimitive);
      }

      // EMA Fast line - without grid
      if (indicators?.emaFast?.length) {
        const emaFastSeries = chart.addSeries(LineSeries, {
          color: "#f59e0b",
          lineWidth: 2,
          title: "EMA Fast",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const emaFastData = indicators.emaFast
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        emaFastSeries.setData(emaFastData);
      }

      // EMA Slow line - without grid
      if (indicators?.emaSlow?.length) {
        const emaSlowSeries = chart.addSeries(LineSeries, {
          color: "#3b82f6",
          lineWidth: 2,
          title: "EMA Slow",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const emaSlowData = indicators.emaSlow
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        emaSlowSeries.setData(emaSlowData);
      }

      // ===== Range Filter (RF) lines =====
      const hasRfMid = indicators?.rf_f?.length;
      const hasLegacyRange = indicators?.rangeFilter?.length;
      if (vis.rangeFilter && (hasRfMid || hasLegacyRange)) {
        const rfMidSeries = chart.addSeries(LineSeries, {
          color: "#26A69A",
          lineWidth: 2,
          title: "RF Mid",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfMidData = (hasRfMid ? indicators.rf_f : indicators.rangeFilter)
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfMidSeries.setData(rfMidData);
        seriesRef.current.rfMid = rfMidSeries;
      }

      if (vis.rangeFilter && indicators?.rf_hb?.length) {
        const rfHbSeries = chart.addSeries(LineSeries, {
          color: "#10b981",
          lineWidth: 1,
          lineStyle: 1, // dashed
          title: "RF High Band",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfHbData = indicators.rf_hb
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfHbSeries.setData(rfHbData);
        seriesRef.current.rfHb = rfHbSeries;
      }

      if (vis.rangeFilter && indicators?.rf_lb?.length) {
        const rfLbSeries = chart.addSeries(LineSeries, {
          color: "#ef4444",
          lineWidth: 1,
          lineStyle: 1, // dashed
          title: "RF Low Band",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfLbData = indicators.rf_lb
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfLbSeries.setData(rfLbData);
        seriesRef.current.rfLb = rfLbSeries;
      }

      // ===== SuperTrend up / down lines =====
      const hasStUp = indicators?.st_up?.length;
      const hasLegacySt = indicators?.supertrend?.length;
      if (vis.superTrend && (hasStUp || hasLegacySt)) {
        const stUpSeries = chart.addSeries(LineSeries, {
          color: "#22c55e",
          lineWidth: 2,
          title: "ST Up",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stUpData = (hasStUp ? indicators.st_up : indicators.supertrend)
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stUpSeries.setData(stUpData);
        seriesRef.current.stUp = stUpSeries;
      }

      if (vis.superTrend && (indicators?.st_dn?.length || (hasLegacySt && !indicators?.st_dn?.length))) {
        const stDnSeries = chart.addSeries(LineSeries, {
          color: "#f97316",
          lineWidth: 2,
          title: "ST Down",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stDnSource = indicators?.st_dn?.length ? indicators.st_dn : indicators.supertrend;
        const stDnData = stDnSource
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stDnSeries.setData(stDnData);
        seriesRef.current.stDn = stDnSeries;
      }

      // ===== RF SL Short lines =====
      if (vis.rfSlShort && indicators?.rf_sl_S_filt?.length) {
        const rfSlSFiltSeries = chart.addSeries(LineSeries, {
          color: "#f97316",
          lineWidth: 1,
          title: "RF SL S Mid",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfSlSFiltData = indicators.rf_sl_S_filt
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfSlSFiltSeries.setData(rfSlSFiltData);
        seriesRef.current.rfSlSFilt = rfSlSFiltSeries;
      }
      if (vis.rfSlShort && indicators?.rf_sl_S_hband?.length) {
        const rfSlSHbSeries = chart.addSeries(LineSeries, {
          color: "#fdba74",
          lineWidth: 1,
          lineStyle: 1,
          title: "RF SL S High",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfSlSHbData = indicators.rf_sl_S_hband
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfSlSHbSeries.setData(rfSlSHbData);
        seriesRef.current.rfSlSHb = rfSlSHbSeries;
      }
      if (vis.rfSlShort && indicators?.rf_sl_S_lband?.length) {
        const rfSlSLbSeries = chart.addSeries(LineSeries, {
          color: "#fdba74",
          lineWidth: 1,
          lineStyle: 1,
          title: "RF SL S Low",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const rfSlSLbData = indicators.rf_sl_S_lband
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        rfSlSLbSeries.setData(rfSlSLbData);
        seriesRef.current.rfSlSLb = rfSlSLbSeries;
      }

      // ===== ST SL Short lines =====
      if (vis.stSlShort && indicators?.st_sl_S_up?.length) {
        const stSlSUpSeries = chart.addSeries(LineSeries, {
          color: "#ea580c",
          lineWidth: 2,
          title: "ST SL S Up",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stSlSUpData = indicators.st_sl_S_up
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stSlSUpSeries.setData(stSlSUpData);
        seriesRef.current.stSlSUp = stSlSUpSeries;
      }
      if (vis.stSlShort && indicators?.st_sl_S_dn?.length) {
        const stSlSDnSeries = chart.addSeries(LineSeries, {
          color: "#c2410c",
          lineWidth: 2,
          title: "ST SL S Dn",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stSlSDnData = indicators.st_sl_S_dn
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stSlSDnSeries.setData(stSlSDnData);
        seriesRef.current.stSlSDn = stSlSDnSeries;
      }

      // ===== ST TP RSI Short lines =====
      if (vis.stTpRsiShort && indicators?.st_tp_rsi_S_up?.length) {
        const stTpRsiSUpSeries = chart.addSeries(LineSeries, {
          color: "#0ea5e9",
          lineWidth: 2,
          title: "ST TP RSI S Up",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stTpRsiSUpData = indicators.st_tp_rsi_S_up
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stTpRsiSUpSeries.setData(stTpRsiSUpData);
        seriesRef.current.stTpRsiSUp = stTpRsiSUpSeries;
      }
      if (vis.stTpRsiShort && indicators?.st_tp_rsi_S_dn?.length) {
        const stTpRsiSDnSeries = chart.addSeries(LineSeries, {
          color: "#0284c7",
          lineWidth: 2,
          title: "ST TP RSI S Dn",
          lastValueVisible: false,
          priceLineVisible: false,
        });
        const stTpRsiSDnData = indicators.st_tp_rsi_S_dn
          .filter((d) => d.value !== null)
          .map((d) => ({ time: d.time, value: d.value }));
        stTpRsiSDnSeries.setData(stTpRsiSDnData);
        seriesRef.current.stTpRsiSDn = stTpRsiSDnSeries;
      }

      // Handle crosshair/hover to show OHLC
      chart.subscribeCrosshairMove((param) => {
        if (param.point === undefined) {
          setOhlc(null);
          return;
        }

        const time = param.time;
        if (time && candleMap[time]) {
          const candle = candleMap[time];
          setOhlc({
            open: candle.open.toFixed(2),
            high: candle.high.toFixed(2),
            low: candle.low.toFixed(2),
            close: candle.close.toFixed(2),
          });
        }
      });

      chart.timeScale().fitContent();

      const handleResize = () => {
        if (containerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: containerRef.current.clientWidth,
          });
        }
      };
      window.addEventListener("resize", handleResize);

      return () => {
        window.removeEventListener("resize", handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }
      };
    } catch (err) {
      console.error("DebugChart error:", err);
      setError(err.message);
    }
  }, [candles, indicators, markers, trades, debugMarkers, visibility]);

  if (error) {
    return (
      <div
        style={{
          width: "100%",
          height: "1000px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#1a1a2e",
          borderRadius: "8px",
          color: "#ef4444",
        }}
      >
        Chart Error: {error}
      </div>
    );
  }

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "1000px",
      }}
    >
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "8px",
          overflow: "hidden",
        }}
      />

      {/* OHLC Info Box */}
      {ohlc && (
        <div
          style={{
            position: "absolute",
            top: "12px",
            left: "12px",
            background: "rgba(26, 26, 46, 0.95)",
            border: "1px solid #2d2d44",
            borderRadius: "6px",
            padding: "8px 12px",
            fontSize: "12px",
            fontFamily: "monospace",
            color: "#d1d5db",
            zIndex: 10,
            minWidth: "140px",
          }}
        >
          <div style={{ marginBottom: "4px" }}>
            <span style={{ color: "#f59e0b" }}>O:</span> {ohlc.open}
          </div>
          <div style={{ marginBottom: "4px" }}>
            <span style={{ color: "#22c55e" }}>H:</span> {ohlc.high}
          </div>
          <div style={{ marginBottom: "4px" }}>
            <span style={{ color: "#ef4444" }}>L:</span> {ohlc.low}
          </div>
          <div>
            <span style={{ color: "#3b82f6" }}>C:</span> {ohlc.close}
          </div>
        </div>
      )}
    </div>
  );
}

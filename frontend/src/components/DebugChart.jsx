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

      for (const marker of this._markers) {
        const x = timeScale.timeToCoordinate(marker.time);
        if (x === null) continue;

        // Get price coordinate for the marker
        const priceY = this._series.priceToCoordinate(marker.price || 0);
        if (priceY === null) continue;

        const pixelX = x * ratio;
        const pixelY = priceY * scope.verticalPixelRatio;

        ctx.save();

        const color = marker.color || "#2962FF";
        const arrowSize = 6 * ratio;
        const offset = 10 * ratio;

        if (marker.position === "belowBar") {
          // Arrow pointing up (Entry) - below the candle
          const arrowY = pixelY + offset;

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
            ctx.font = `bold ${11 * ratio}px Arial`;
            ctx.textAlign = "center";
            ctx.fillStyle = color;
            ctx.fillText(marker.text, pixelX, arrowY + arrowSize * 1.5 + 12 * ratio);
          }
        } else {
          // Arrow pointing down (Exit) - above the candle
          const arrowY = pixelY - offset;

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
            ctx.font = `bold ${11 * ratio}px Arial`;
            ctx.textAlign = "center";
            ctx.fillStyle = color;
            ctx.fillText(marker.text, pixelX, arrowY - arrowSize * 1.5 - 5 * ratio);
          }
        }

        ctx.restore();
      }
    });
  }
}

export default function DebugChart({ candles, indicators, markers, trades }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const [error, setError] = useState(null);
  const [ohlc, setOhlc] = useState(null);

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
        },
        width: containerRef.current.clientWidth,
        height: 500,
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

      // Process markers with price data
      if (markers?.length) {
        const markersWithPrice = markers.map((m) => {
          const candle = candleMap[m.time];
          let price = candle?.close || candle?.low || 0;
          if (m.position === "belowBar") {
            price = candle?.low || price;
          } else {
            price = candle?.high || price;
          }
          return { ...m, price };
        });

        const markersPrimitive = new MarkersPrimitive(markersWithPrice);
        candleSeries.attachPrimitive(markersPrimitive);
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
      if (hasRfMid || hasLegacyRange) {
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
      }

      if (indicators?.rf_hb?.length) {
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
      }

      if (indicators?.rf_lb?.length) {
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
      }

      // ===== SuperTrend up / down lines =====
      const hasStUp = indicators?.st_up?.length;
      const hasLegacySt = indicators?.supertrend?.length;
      if (hasStUp || hasLegacySt) {
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
      }

      if (indicators?.st_dn?.length || (hasLegacySt && !indicators?.st_dn?.length)) {
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
  }, [candles, indicators, markers, trades]);

  if (error) {
    return (
      <div
        style={{
          width: "100%",
          height: "500px",
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
        height: "500px",
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

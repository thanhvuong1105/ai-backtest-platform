export default function StrategyCard({ data }) {
  const s = data.summary;

  return (
    <div style={{
      background: "#0f172a",
      padding: 20,
      borderRadius: 12
    }}>
      <p><b>Symbol:</b> {data.symbol}</p>
      <p><b>Timeframe:</b> {data.timeframe}</p>
      <p><b>Profit Factor:</b> {s.profitFactor}</p>
      <p><b>Winrate:</b> {s.winrate}%</p>
      <p><b>Max DD:</b> {s.maxDrawdownPct}%</p>
      <p><b>Score:</b> {s.score}</p>
    </div>
  );
}

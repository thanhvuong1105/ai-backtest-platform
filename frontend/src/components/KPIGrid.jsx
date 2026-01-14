const Card = ({ label, value }) => (
  <div style={{
    background: "#111827",
    padding: 16,
    borderRadius: 12,
    minWidth: 160
  }}>
    <div style={{ fontSize: 12, opacity: 0.7 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 600 }}>{value}</div>
  </div>
);

export default function KPIGrid({ stats }) {
  if (!stats) return null;

  return (
    <div style={{ display: "flex", gap: 16 }}>
      <Card label="Total Runs" value={stats.totalRuns} />
      <Card label="Passed" value={stats.passedRuns} />
      <Card label="Rejected" value={stats.rejectedRuns} />
    </div>
  );
}

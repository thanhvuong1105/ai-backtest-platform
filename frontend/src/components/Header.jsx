export default function Header({ onRun, loading, title = "Optimizer Dashboard" }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 24,
        padding: "16px 0",
      }}
    >
      <h1 style={{ margin: 0, fontSize: "24px", color: "#fff" }}>{title}</h1>

      {onRun && (
        <button
          onClick={onRun}
          disabled={loading}
          style={{
            padding: "10px 20px",
            borderRadius: 8,
            border: "none",
            background: loading ? "#4b5563" : "#8b5cf6",
            color: "#fff",
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Running..." : "Run Optimize"}
        </button>
      )}
    </div>
  );
}

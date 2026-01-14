export default function TradeFilterBar({
  filters,
  onChange
}) {
  const toggle = (key) => {
    onChange({
      ...filters,
      [key]: !filters[key]
    });
  };

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        flexWrap: "wrap"
      }}
    >
      <FilterBtn
        label="Long"
        active={filters.long}
        onClick={() => toggle("long")}
        color="#22c55e"
      />
      <FilterBtn
        label="Short"
        active={filters.short}
        onClick={() => toggle("short")}
        color="#ef4444"
      />
      <FilterBtn
        label="Win"
        active={filters.win}
        onClick={() => toggle("win")}
        color="#22c55e"
      />
      <FilterBtn
        label="Loss"
        active={filters.loss}
        onClick={() => toggle("loss")}
        color="#ef4444"
      />
    </div>
  );
}

function FilterBtn({ label, active, onClick, color }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 14px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.15)",
        background: active ? color : "transparent",
        color: active ? "#020617" : "#e5e7eb",
        fontWeight: 600,
        cursor: "pointer",
        opacity: active ? 1 : 0.6
      }}
    >
      {label}
    </button>
  );
}

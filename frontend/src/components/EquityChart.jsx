import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";

const COLORS = ["#22c55e", "#38bdf8", "#f59e0b", "#a78bfa"];

export default function EquityChart({ data, visible }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
        <XAxis dataKey="time" hide />
        <YAxis
          tickFormatter={v => `${v}%`}
          stroke="#94a3b8"
        />
        <Tooltip
          formatter={v => `${v}%`}
          contentStyle={{
            background: "#020617",
            border: "1px solid #1e293b"
          }}
        />
        <Legend />

        {Object.keys(visible).map((key, i) =>
          visible[key] ? (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
            />
          ) : null
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

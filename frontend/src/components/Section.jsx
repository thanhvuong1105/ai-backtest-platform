export default function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h2 style={{ marginBottom: 12 }}>{title}</h2>
      {children}
    </div>
  );
}

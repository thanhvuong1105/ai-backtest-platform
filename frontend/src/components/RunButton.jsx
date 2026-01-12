export default function RunButton({ onRun, loading }) {
  return (
    <button
      onClick={onRun}
      disabled={loading}
      className="bg-cyan-500 px-4 py-2 rounded text-black"
    >
      {loading ? "Running..." : "Run Optimize"}
    </button>
  );
}

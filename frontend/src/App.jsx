import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import MemoryPage from "./pages/MemoryPage";
import DebugView from "./pages/DebugView";

const navLinkStyle = ({ isActive }) => ({
  color: isActive ? "#22c55e" : "#9ca3af",
  textDecoration: "none",
  padding: "8px 16px",
  borderRadius: "6px",
  background: isActive ? "rgba(34,197,94,0.15)" : "transparent",
  fontWeight: isActive ? "600" : "400",
  transition: "all 0.2s ease",
});

function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: "100vh", background: "#0f0f1a" }}>
        {/* Navigation */}
        <nav
          style={{
            background: "#1a1a2e",
            padding: "12px 24px",
            borderBottom: "1px solid #2d2d44",
            display: "flex",
            gap: "24px",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: "18px", fontWeight: "600", color: "#fff" }}>
            AI Backtest Platform
          </span>
          <div style={{ display: "flex", gap: "12px" }}>
            <NavLink to="/" style={navLinkStyle}>
              Optimizer
            </NavLink>
            <NavLink to="/memory" style={navLinkStyle}>
              Memory
            </NavLink>
            <NavLink to="/debug" style={navLinkStyle}>
              Debug View
            </NavLink>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/debug" element={<DebugView />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;

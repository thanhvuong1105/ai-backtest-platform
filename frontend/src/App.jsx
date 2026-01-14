import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import DebugView from "./pages/DebugView";

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
          <div style={{ display: "flex", gap: "16px" }}>
            <NavLink
              to="/"
              style={({ isActive }) => ({
                color: isActive ? "#8b5cf6" : "#9ca3af",
                textDecoration: "none",
                padding: "8px 16px",
                borderRadius: "6px",
                background: isActive ? "#8b5cf620" : "transparent",
                fontWeight: isActive ? "600" : "400",
              })}
            >
              Optimizer
            </NavLink>
            <NavLink
              to="/debug"
              style={({ isActive }) => ({
                color: isActive ? "#8b5cf6" : "#9ca3af",
                textDecoration: "none",
                padding: "8px 16px",
                borderRadius: "6px",
                background: isActive ? "#8b5cf620" : "transparent",
                fontWeight: isActive ? "600" : "400",
              })}
            >
              Debug View
            </NavLink>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/debug" element={<DebugView />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;

import React, { useState } from "react";
import LiveDashboard from "./components/LiveDashboard";
import ModelComparison from "./components/ModelComparison";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("live");

  return (
    <div className="app-shell">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="navbar-brand">
          <span className="brand-icon">⚙️</span>
          <div>
            <h1 className="brand-title">Industrial Anomaly Detection</h1>
            <span className="brand-sub">Real-Time Sensor Intelligence &amp; Predictive Maintenance</span>
          </div>
        </div>

        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === "live" ? "active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            📡 Live Monitor
          </button>
          <button
            className={`tab-btn ${activeTab === "models" ? "active" : ""}`}
            onClick={() => setActiveTab("models")}
          >
            📊 Model Comparison
          </button>
        </nav>
      </header>

      {/* Main View Container */}
      <main className="main-content">
        {activeTab === "live" && <LiveDashboard />}
        {activeTab === "models" && <ModelComparison />}
      </main>

      <footer className="footer">
        FastAPI Backend (Port 8000) &nbsp;|&nbsp; MongoDB Storage &nbsp;|&nbsp; Isolation Forest &amp; PyTorch Autoencoder
      </footer>
    </div>
  );
}

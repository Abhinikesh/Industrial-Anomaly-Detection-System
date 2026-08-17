import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [status, setStatus] = useState(null);
  const [error, setError]   = useState(null);

  useEffect(() => {
    axios.get(`${API}/`)
      .then(res => setStatus(res.data.status))
      .catch(() => setError("Could not reach backend — is FastAPI running on :8000?"));
  }, []);

  return (
    <div className="app">
      <header className="hero">
        <h1>Industrial Anomaly Detection</h1>
        <p className="subtitle">
          Real-time sensor monitoring &amp; anomaly detection dashboard
        </p>
      </header>

      <section className="status-card">
        <h2>Backend Connection</h2>
        {status && (
          <div className="badge ok">
            ✓ {status}
          </div>
        )}
        {error && (
          <div className="badge err">
            ✗ {error}
          </div>
        )}
        {!status && !error && (
          <div className="badge loading">Connecting...</div>
        )}
      </section>

      <section className="info-grid">
        <div className="info-card">
          <span className="icon">🌡️</span>
          <strong>Temperature</strong>
          <span>60–80 °C</span>
        </div>
        <div className="info-card">
          <span className="icon">📳</span>
          <strong>Vibration</strong>
          <span>0.1–0.5 g</span>
        </div>
        <div className="info-card">
          <span className="icon">💨</span>
          <strong>Pressure</strong>
          <span>100–120 bar</span>
        </div>
        <div className="info-card">
          <span className="icon">🔩</span>
          <strong>Torque</strong>
          <span>40–60 Nm</span>
        </div>
        <div className="info-card">
          <span className="icon">🔧</span>
          <strong>Tool Wear</strong>
          <span>0–200 min</span>
        </div>
      </section>

      <footer className="footer">
        API: <code>{API}</code> &nbsp;|&nbsp; Models: Isolation Forest + Autoencoder
      </footer>
    </div>
  );
}

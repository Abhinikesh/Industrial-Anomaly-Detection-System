import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const API_BASE = "http://localhost:8000";

const AnomalyDot = (props) => {
  const { cx, cy, payload } = props;
  if (payload && payload.is_anomaly) {
    return (
      <svg key={`anomaly-${payload.timestamp}-${cx}-${cy}`}>
        <circle cx={cx} cy={cy} r={6} fill="#ef4444" stroke="#ffffff" strokeWidth={2} />
        <circle cx={cx} cy={cy} r={9} fill="none" stroke="#ef4444" strokeWidth={1.5} opacity={0.6} className="pulse-ring" />
      </svg>
    );
  }
  return null;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const isAnomaly = payload[0]?.payload?.is_anomaly;
    return (
      <div className="custom-tooltip">
        <div className="tooltip-header">
          <span>Time: {label}</span>
          {isAnomaly && <span className="tooltip-alert-tag">⚠ ANOMALY</span>}
        </div>
        <div className="tooltip-body">
          {payload.map((entry, index) => (
            <div key={`item-${index}`} className="tooltip-row" style={{ color: entry.color }}>
              <span>{entry.name}:</span>
              <strong>{typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}</strong>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

export default function LiveDashboard() {
  const [readings, setReadings] = useState([]);
  const [stats, setStats] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [lastReading, setLastReading] = useState(null);
  const [isLive, setIsLive] = useState(true);
  const [fetchError, setFetchError] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [recentRes, statsRes, anomaliesRes] = await Promise.all([
        axios.get(`${API_BASE}/readings/recent?limit=50`),
        axios.get(`${API_BASE}/readings/stats`),
        axios.get(`${API_BASE}/readings/anomalies?limit=50`),
      ]);

      const chronological = [...recentRes.data].reverse().map((item, idx) => ({
        ...item,
        displayTime: item.timestamp ? item.timestamp.substring(11, 19) : `#${idx}`,
      }));

      setReadings(chronological);
      if (recentRes.data.length > 0) {
        setLastReading(recentRes.data[0]); 
      }
      setStats(statsRes.data);
      setAnomalies(anomaliesRes.data);
      setFetchError(null);
    } catch (err) {
      setFetchError("Connection lost to backend on :8000. Retrying live stream in background...");
    } finally {
      setInitialLoading(false);
    }
  };


  useEffect(() => {
    fetchData();
    if (!isLive) return;

    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [isLive]);

  if (initialLoading) {
    return (
      <div className="dashboard-loading-state">
        <div className="spinner"></div>
        <h3>Connecting to Live Sensor Stream...</h3>
        <p className="dim">Polling telemetry data and model inference pipelines from FastAPI (:8000)</p>
      </div>
    );
  }

  const isEmptyDatabase = readings.length === 0 && !fetchError;

  return (
    <div className="dashboard-container">
      {/* Top Status & System Banner */}
      <div className="status-banner">
        <div className="status-indicator-box">
          <span className="section-label">Live Telemetry State</span>
          {lastReading ? (
            <div
              className={`status-pill ${
                lastReading.is_anomaly ? "status-alert" : "status-ok"
              }`}
            >
              <span className="pulse-dot"></span>
              <strong>
                {lastReading.is_anomaly ? "CRITICAL: ANOMALY DETECTED" : "NOMINAL: SYSTEM NORMAL"}
              </strong>
            </div>
          ) : (
            <div className="status-pill status-loading">Waiting for data stream...</div>
          )}
        </div>

        <div className="status-details">
          {lastReading && (
            <>
              <div className="detail-item">
                <span className="dim">Machine Node:</span> <code>{lastReading.machine_id}</code>
              </div>
              <div className="detail-item">
                <span className="dim">Last Reading:</span>{" "}
                <span className="font-mono">{lastReading.timestamp ? lastReading.timestamp.substring(11, 19) : "--"}</span>
              </div>
            </>
          )}
          <button
            className={`btn-toggle ${isLive ? "btn-active" : ""}`}
            onClick={() => setIsLive(!isLive)}
          >
            {isLive ? "● Streaming Active (2s)" : "❚❚ Stream Paused"}
          </button>
        </div>
      </div>

      {fetchError && (
        <div className="alert-box error">
          <span>⚠️ {fetchError}</span>
        </div>
      )}

      {isEmptyDatabase && (
        <div className="alert-box info">
          <span>ℹ️ <strong>Fresh Database Detected:</strong> No sensor readings ingested yet. Run <code>python ml/simulator.py</code> to start streaming data.</span>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card card-total">
          <div className="kpi-icon-wrap">📊</div>
          <div className="kpi-content">
            <span className="kpi-label">Total Readings</span>
            <span className="kpi-value">{stats ? stats.total_readings.toLocaleString() : "--"}</span>
            <span className="kpi-sub">Ingested into MongoDB</span>
          </div>
        </div>

        <div className="kpi-card card-anomalies">
          <div className="kpi-icon-wrap icon-warn">⚠️</div>
          <div className="kpi-content">
            <span className="kpi-label">Anomalies Detected</span>
            <span className={`kpi-value ${stats && stats.total_anomalies > 0 ? "text-warn" : ""}`}>
              {stats ? stats.total_anomalies.toLocaleString() : "--"}
            </span>
            <span className="kpi-sub">
              Incident Rate: <strong>{stats ? `${stats.anomaly_rate}%` : "--"}</strong>
            </span>
          </div>
        </div>

        <div className="kpi-card card-if">
          <div className="kpi-icon-wrap icon-blue">🌲</div>
          <div className="kpi-content">
            <span className="kpi-label">Isolation Forest</span>
            <span className="kpi-value text-blue">
              {stats ? stats.flagged_by.isolation_forest_only + stats.flagged_by.both_models : "--"}
            </span>
            <span className="kpi-sub">
              Exclusive: {stats ? stats.flagged_by.isolation_forest_only : 0} | Shared: {stats ? stats.flagged_by.both_models : 0}
            </span>
          </div>
        </div>

        <div className="kpi-card card-ae">
          <div className="kpi-icon-wrap icon-purple">🧠</div>
          <div className="kpi-content">
            <span className="kpi-label">Autoencoder (MSE)</span>
            <span className="kpi-value text-purple">
              {stats ? stats.flagged_by.autoencoder_only + stats.flagged_by.both_models : "--"}
            </span>
            <span className="kpi-sub">
              Exclusive: {stats ? stats.flagged_by.autoencoder_only : 0} | Shared: {stats ? stats.flagged_by.both_models : 0}
            </span>
          </div>
        </div>
      </div>

      {/* 3 Real-time Line Charts */}
      <div className="charts-grid">
        {/* Chart 1: Temperature */}
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h3>Thermal Profile (Air vs Process)</h3>
              <span className="chart-tag">Kelvin (K) &bull; Threshold Monitoring</span>
            </div>
            <span className="badge-legend">Dual Sensor</span>
          </div>
          <div className="chart-wrapper">
            {isEmptyDatabase ? (
              <div className="chart-empty-state">Waiting for sensor stream...</div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={readings} margin={{ top: 15, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#253248" />
                  <XAxis dataKey="displayTime" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={["auto", "auto"]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                  <Line
                    type="monotone"
                    dataKey="air_temp"
                    name="Air Temp (K)"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={<AnomalyDot />}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="process_temp"
                    name="Process Temp (K)"
                    stroke="#f97316"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Chart 2: Speed & Torque */}
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <h3>Kinetic &amp; Mechanical Load</h3>
              <span className="chart-tag">Spindle Speed (RPM) vs Torque (Nm)</span>
            </div>
            <span className="badge-legend">Dual Axis</span>
          </div>
          <div className="chart-wrapper">
            {isEmptyDatabase ? (
              <div className="chart-empty-state">Waiting for sensor stream...</div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={readings} margin={{ top: 15, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#253248" />
                  <XAxis dataKey="displayTime" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="rpm" domain={["auto", "auto"]} stroke="#a855f7" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="torque" orientation="right" domain={["auto", "auto"]} stroke="#34d399" tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                  <Line
                    yAxisId="rpm"
                    type="monotone"
                    dataKey="rpm"
                    name="Speed (RPM)"
                    stroke="#a855f7"
                    strokeWidth={2.5}
                    dot={<AnomalyDot />}
                    isAnimationActive={false}
                  />
                  <Line
                    yAxisId="torque"
                    type="monotone"
                    dataKey="torque"
                    name="Torque (Nm)"
                    stroke="#34d399"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Chart 3: Tool Wear */}
        <div className="chart-card full-width">
          <div className="chart-header">
            <div>
              <h3>Tool Wear Degradation Timeline</h3>
              <span className="chart-tag">Cumulative Tool Usage (Minutes) &bull; Monotonic Progression</span>
            </div>
            <span className="badge-legend">Wear Rate</span>
          </div>
          <div className="chart-wrapper">
            {isEmptyDatabase ? (
              <div className="chart-empty-state">Waiting for sensor stream...</div>
            ) : (
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={readings} margin={{ top: 15, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#253248" />
                  <XAxis dataKey="displayTime" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={["auto", "auto"]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                  <Line
                    type="monotone"
                    dataKey="tool_wear"
                    name="Tool Wear (min)"
                    stroke="#fbbf24"
                    strokeWidth={2.5}
                    dot={<AnomalyDot />}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Anomalies Table */}
      <div className="table-card">
        <div className="table-header">
          <div>
            <h3>Live Anomalies Incident Log</h3>
            <span className="dim">Real-time flagged events by Isolation Forest and/or Autoencoder</span>
          </div>
          <span className="badge-count">{anomalies.length} total events recorded</span>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Machine ID</th>
                <th>Thermal (Air / Proc)</th>
                <th>Speed (RPM)</th>
                <th>Torque</th>
                <th>Tool Wear</th>
                <th>Model Trigger</th>
                <th>Model Scores</th>
                <th>Ground Truth</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.length === 0 ? (
                <tr>
                  <td colSpan="9" className="text-center dim py-5">
                    {isEmptyDatabase ? "No sensor readings recorded yet." : "No anomalies recorded. System operating in nominal state."}
                  </td>
                </tr>
              ) : (
                anomalies.slice(0, 15).map((row, idx) => {
                  let modelLabel = "None";
                  let tagClass = "tag-none";
                  if (row.iso_flag && row.ae_flag) {
                    modelLabel = "⚡ Both Models";
                    tagClass = "tag-both";
                  } else if (row.iso_flag) {
                    modelLabel = "🌲 Isolation Forest";
                    tagClass = "tag-if";
                  } else if (row.ae_flag) {
                    modelLabel = "🧠 Autoencoder";
                    tagClass = "tag-ae";
                  }

                  return (
                    <tr key={idx} className="anomaly-row">
                      <td><code className="font-mono text-cyan">{row.timestamp ? row.timestamp.substring(11, 19) : "--"}</code></td>
                      <td><span className="machine-badge">{row.machine_id}</span></td>
                      <td className="font-mono">{row.air_temp?.toFixed(1)}K / {row.process_temp?.toFixed(1)}K</td>
                      <td className="font-mono">{row.rpm?.toFixed(0)}</td>
                      <td className="font-mono">{row.torque?.toFixed(1)} Nm</td>
                      <td className="font-mono">{row.tool_wear?.toFixed(0)} min</td>
                      <td>
                        <span className={`tag-model ${tagClass}`}>
                          {modelLabel}
                        </span>
                      </td>
                      <td>
                        <div className="scores-cell font-mono">
                          <span>IF: {row.iso_score?.toFixed(3)}</span>
                          <span>AE: {row.ae_score?.toFixed(3)}</span>
                        </div>
                      </td>
                      <td>
                        {row.true_failure ? (
                          <span className="badge-failure">Actual Failure</span>
                        ) : (
                          <span className="badge-normal">True Normal</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

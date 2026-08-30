import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const API_BASE = "http://localhost:8000";

// ── Machine type config ────────────────────────────────────────────────────────
// Each machine type declares: label, icon, API key, sensor charts, and table columns.

const MACHINE_CONFIGS = {
  milling_machine: {
    label: "Milling Machine",
    icon: "⚙️",
    color: "#38bdf8",
    description: "AI4I 2020 — CNC milling spindle telemetry",
    charts: [
      {
        title: "Thermal Profile (Air vs Process)",
        tag: "Kelvin (K) · Threshold Monitoring",
        badge: "Dual Sensor",
        height: 240,
        lines: [
          { key: "air_temp",     name: "Air Temp (K)",     color: "#38bdf8", yAxisId: "left", dot: true },
          { key: "process_temp", name: "Process Temp (K)", color: "#f97316", yAxisId: "left", dot: false },
        ],
        dualAxis: false,
      },
      {
        title: "Kinetic & Mechanical Load",
        tag: "Spindle Speed (RPM) vs Torque (Nm)",
        badge: "Dual Axis",
        height: 240,
        lines: [
          { key: "rpm",    name: "Speed (RPM)",  color: "#a855f7", yAxisId: "left",  dot: true  },
          { key: "torque", name: "Torque (Nm)",  color: "#34d399", yAxisId: "right", dot: false },
        ],
        dualAxis: true,
      },
      {
        title: "Tool Wear Degradation Timeline",
        tag: "Cumulative Tool Usage (Minutes) · Monotonic Progression",
        badge: "Wear Rate",
        height: 230,
        fullWidth: true,
        lines: [
          { key: "tool_wear", name: "Tool Wear (min)", color: "#fbbf24", yAxisId: "left", dot: true },
        ],
        dualAxis: false,
      },
    ],
    tableColumns: [
      { header: "Thermal (Air / Proc)", render: r => `${r.air_temp?.toFixed(1) ?? "--"}K / ${r.process_temp?.toFixed(1) ?? "--"}K` },
      { header: "Speed (RPM)",          render: r => r.rpm?.toFixed(0)          ?? "--" },
      { header: "Torque",               render: r => r.torque != null ? `${r.torque.toFixed(1)} Nm` : "--" },
      { header: "Tool Wear",            render: r => r.tool_wear != null ? `${r.tool_wear.toFixed(0)} min` : "--" },
    ],
  },

  fleet_machine: {
    label: "Azure Fleet",
    icon: "☁️",
    color: "#06b6d4",
    description: "Microsoft Azure PdM — 100-machine rotating equipment fleet",
    charts: [
      {
        title: "Electrical Profile (Voltage)",
        tag: "Volts · Power Supply Monitoring",
        badge: "Single Sensor",
        height: 240,
        lines: [
          { key: "voltage", name: "Voltage (V)", color: "#06b6d4", yAxisId: "left", dot: true },
        ],
        dualAxis: false,
      },
      {
        title: "Mechanical Load (Rotation vs Pressure)",
        tag: "RPM vs PSI",
        badge: "Dual Sensor",
        height: 240,
        lines: [
          { key: "rotation", name: "Rotation (RPM)", color: "#a78bfa", yAxisId: "left",  dot: true  },
          { key: "pressure", name: "Pressure (PSI)", color: "#f59e0b", yAxisId: "right", dot: false },
        ],
        dualAxis: true,
      },
      {
        title: "Vibration Level",
        tag: "mm/s · Bearing & Imbalance Indicator",
        badge: "Vibration",
        height: 230,
        fullWidth: true,
        lines: [
          { key: "vibration", name: "Vibration (mm/s)", color: "#f87171", yAxisId: "left", dot: true },
        ],
        dualAxis: false,
      },
    ],
    tableColumns: [
      { header: "Voltage",   render: r => r.voltage   != null ? `${r.voltage.toFixed(1)} V`   : "--" },
      { header: "Rotation",  render: r => r.rotation  != null ? `${r.rotation.toFixed(0)} RPM` : "--" },
      { header: "Pressure",  render: r => r.pressure  != null ? `${r.pressure.toFixed(1)} PSI` : "--" },
      { header: "Vibration", render: r => r.vibration != null ? `${r.vibration.toFixed(3)} mm/s` : "--" },
    ],
  },

  water_pump: {
    label: "Water Pump",
    icon: "💧",
    color: "#10b981",
    description: "Kaggle Pump Sensor — Industrial water pump degradation tracking",
    // Show the top 4 correlated sensors: sensor_04, sensor_00, sensor_10, sensor_06
    charts: [
      {
        title: "Primary Sensors (sensor_04 · sensor_00)",
        tag: "Top 2 failure-correlated sensors (|r| > 0.87)",
        badge: "High Corr",
        height: 240,
        lines: [
          { key: "sensor_04", name: "sensor_04", color: "#10b981", yAxisId: "left",  dot: true  },
          { key: "sensor_00", name: "sensor_00", color: "#34d399", yAxisId: "right", dot: false },
        ],
        dualAxis: true,
      },
      {
        title: "Secondary Sensors (sensor_10 · sensor_06)",
        tag: "Sensors 3-4 by failure correlation (|r| > 0.85)",
        badge: "Secondary",
        height: 240,
        lines: [
          { key: "sensor_10", name: "sensor_10", color: "#6ee7b7", yAxisId: "left",  dot: true  },
          { key: "sensor_06", name: "sensor_06", color: "#fbbf24", yAxisId: "right", dot: false },
        ],
        dualAxis: true,
      },
      {
        title: "Vibration / Stress (sensor_11 · sensor_07)",
        tag: "Sensors 5-6 by failure correlation (|r| > 0.78)",
        badge: "Vibration",
        height: 230,
        fullWidth: true,
        lines: [
          { key: "sensor_11", name: "sensor_11", color: "#a78bfa", yAxisId: "left",  dot: true  },
          { key: "sensor_07", name: "sensor_07", color: "#f87171", yAxisId: "right", dot: false },
        ],
        dualAxis: true,
      },
    ],
    tableColumns: [
      { header: "sensor_04", render: r => r.sensor_04?.toFixed(4) ?? "--" },
      { header: "sensor_00", render: r => r.sensor_00?.toFixed(4) ?? "--" },
      { header: "sensor_10", render: r => r.sensor_10?.toFixed(4) ?? "--" },
      { header: "sensor_06", render: r => r.sensor_06?.toFixed(4) ?? "--" },
    ],
  },
};

const MACHINE_TYPES = ["all", "milling_machine", "fleet_machine", "water_pump"];

const TYPE_LABELS = {
  all:             { label: "All Fleets", icon: "🏭" },
  milling_machine: MACHINE_CONFIGS.milling_machine,
  fleet_machine:   MACHINE_CONFIGS.fleet_machine,
  water_pump:      MACHINE_CONFIGS.water_pump,
};

// ── Sub-components ─────────────────────────────────────────────────────────────

const AnomalyDot = (props) => {
  const { cx, cy, payload } = props;
  if (payload?.is_anomaly) {
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
  if (active && payload?.length) {
    const isAnomaly = payload[0]?.payload?.is_anomaly;
    return (
      <div className="custom-tooltip">
        <div className="tooltip-header">
          <span>Time: {label}</span>
          {isAnomaly && <span className="tooltip-alert-tag">⚠ ANOMALY</span>}
        </div>
        <div className="tooltip-body">
          {payload.map((entry, i) => (
            <div key={i} className="tooltip-row" style={{ color: entry.color }}>
              <span>{entry.name}:</span>
              <strong>{typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}</strong>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

// Fleet overview card shown in "All" mode
const FleetOverviewGrid = ({ fleet, onSelect }) => {
  if (!fleet || fleet.length === 0) {
    return (
      <div className="alert-box info">
        <span>ℹ️ No machine data yet. Run a simulator to populate the fleet.</span>
      </div>
    );
  }
  return (
    <div className="fleet-overview-grid">
      {fleet.map((m) => {
        const cfg = MACHINE_CONFIGS[m.machine_type] || {};
        return (
          <div
            key={m.machine_type}
            className={`fleet-card ${m.is_anomaly_now ? "fleet-card-alert" : "fleet-card-ok"}`}
            style={{ "--fleet-color": cfg.color || "#64748b" }}
          >
            <div className="fleet-card-top">
              <span className="fleet-icon">{cfg.icon || "🔧"}</span>
              <div>
                <div className="fleet-machine-label">{cfg.label || m.machine_type}</div>
                <div className="fleet-machine-desc dim text-xs">{cfg.description || ""}</div>
              </div>
              <span className={`fleet-status-pill ${m.is_anomaly_now ? "pill-alert" : "pill-ok"}`}>
                {m.is_anomaly_now ? "⚠ ANOMALY" : "✓ NORMAL"}
              </span>
            </div>
            <div className="fleet-stats-row">
              <div className="fleet-stat">
                <span className="dim text-xs">Total Readings</span>
                <strong>{m.total_readings.toLocaleString()}</strong>
              </div>
              <div className="fleet-stat">
                <span className="dim text-xs">Anomalies</span>
                <strong className={m.anomaly_count > 0 ? "text-warn" : ""}>{m.anomaly_count.toLocaleString()}</strong>
              </div>
              <div className="fleet-stat">
                <span className="dim text-xs">Rate</span>
                <strong className={m.anomaly_rate > 5 ? "text-warn" : "text-emerald"}>{m.anomaly_rate}%</strong>
              </div>
            </div>
            <button
              className="btn-view-detail"
              onClick={() => onSelect(m.machine_type)}
            >
              View Details →
            </button>
          </div>
        );
      })}
    </div>
  );
};

// Sensor chart panel for a specific machine type
const SensorCharts = ({ readings, config }) => {
  const isEmpty = readings.length === 0;
  return (
    <div className="charts-grid">
      {config.charts.map((chart, ci) => (
        <div key={ci} className={`chart-card ${chart.fullWidth ? "full-width" : ""}`}>
          <div className="chart-header">
            <div>
              <h3>{chart.title}</h3>
              <span className="chart-tag">{chart.tag}</span>
            </div>
            <span className="badge-legend">{chart.badge}</span>
          </div>
          <div className="chart-wrapper">
            {isEmpty ? (
              <div className="chart-empty-state">Waiting for sensor stream...</div>
            ) : (
              <ResponsiveContainer width="100%" height={chart.height}>
                <LineChart data={readings} margin={{ top: 15, right: 20, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#253248" />
                  <XAxis dataKey="displayTime" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="left" domain={["auto", "auto"]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  {chart.dualAxis && (
                    <YAxis yAxisId="right" orientation="right" domain={["auto", "auto"]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  )}
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                  {chart.lines.map((line, li) => (
                    <Line
                      key={li}
                      yAxisId={line.yAxisId}
                      type="monotone"
                      dataKey={line.key}
                      name={line.name}
                      stroke={line.color}
                      strokeWidth={2.5}
                      dot={line.dot ? <AnomalyDot /> : false}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

// Anomaly table — adapts columns per machine type
const AnomalyTable = ({ anomalies, config, isEmpty }) => (
  <div className="table-card">
    <div className="table-header">
      <div>
        <h3>Live Anomalies Incident Log</h3>
        <span className="dim">Real-time flagged events by Isolation Forest and/or Autoencoder</span>
      </div>
      <span className="badge-count">{anomalies.length} events recorded</span>
    </div>
    <div className="table-responsive">
      <table className="data-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Machine ID</th>
            {config.tableColumns.map(c => <th key={c.header}>{c.header}</th>)}
            <th>Model Trigger</th>
            <th>Model Scores</th>
            <th>Ground Truth</th>
          </tr>
        </thead>
        <tbody>
          {anomalies.length === 0 ? (
            <tr>
              <td colSpan={5 + config.tableColumns.length} className="text-center dim py-5">
                {isEmpty ? "No sensor readings recorded yet." : "No anomalies recorded. System operating in nominal state."}
              </td>
            </tr>
          ) : (
            anomalies.slice(0, 15).map((row, idx) => {
              let modelLabel = "None", tagClass = "tag-none";
              if (row.iso_flag && row.ae_flag)   { modelLabel = "⚡ Both Models";       tagClass = "tag-both"; }
              else if (row.iso_flag)             { modelLabel = "🌲 Isolation Forest"; tagClass = "tag-if";   }
              else if (row.ae_flag)              { modelLabel = "🧠 Autoencoder";       tagClass = "tag-ae";   }
              return (
                <tr key={idx} className="anomaly-row">
                  <td><code className="font-mono text-cyan">{row.timestamp?.substring(11, 19) ?? "--"}</code></td>
                  <td><span className="machine-badge">{row.machine_id}</span></td>
                  {config.tableColumns.map(c => (
                    <td key={c.header} className="font-mono">{c.render(row)}</td>
                  ))}
                  <td><span className={`tag-model ${tagClass}`}>{modelLabel}</span></td>
                  <td>
                    <div className="scores-cell font-mono">
                      <span>IF: {row.iso_score?.toFixed(3) ?? "--"}</span>
                      <span>AE: {row.ae_score?.toFixed(3) ?? "--"}</span>
                    </div>
                  </td>
                  <td>
                    {row.true_failure
                      ? <span className="badge-failure">Actual Failure</span>
                      : <span className="badge-normal">True Normal</span>}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────────

export default function LiveDashboard() {
  const [machineType, setMachineType] = useState("all");
  const [readings,    setReadings]    = useState([]);
  const [stats,       setStats]       = useState(null);
  const [anomalies,   setAnomalies]   = useState([]);
  const [fleet,       setFleet]       = useState([]);
  const [lastReading, setLastReading] = useState(null);
  const [isLive,      setIsLive]      = useState(true);
  const [fetchError,  setFetchError]  = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const typeParam = machineType !== "all" ? `?machine_type=${machineType}` : "";
      const [recentRes, statsRes, anomaliesRes, fleetRes] = await Promise.all([
        axios.get(`${API_BASE}/readings/recent?limit=50${machineType !== "all" ? `&machine_type=${machineType}` : ""}`),
        axios.get(`${API_BASE}/readings/stats${typeParam}`),
        axios.get(`${API_BASE}/readings/anomalies?limit=50${machineType !== "all" ? `&machine_type=${machineType}` : ""}`),
        axios.get(`${API_BASE}/readings/fleet-overview`),
      ]);

      const chronological = [...recentRes.data].reverse().map((item, idx) => ({
        ...item,
        // Flatten sensor_values for recharts: { sensor_04: 1.23, ... }
        ...(item.sensor_values || {}),
        displayTime: item.timestamp ? item.timestamp.substring(11, 19) : `#${idx}`,
      }));

      setReadings(chronological);
      setLastReading(recentRes.data[0] ?? null);
      setStats(statsRes.data);
      setAnomalies(anomaliesRes.data);
      setFleet(fleetRes.data);
      setFetchError(null);
    } catch {
      setFetchError("Connection lost to backend on :8000. Retrying live stream in background...");
    } finally {
      setInitialLoading(false);
    }
  }, [machineType]);

  useEffect(() => {
    setInitialLoading(true);
    fetchData();
    if (!isLive) return;
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData, isLive]);

  if (initialLoading) {
    return (
      <div className="dashboard-loading-state">
        <div className="spinner"></div>
        <h3>Connecting to Live Sensor Stream...</h3>
        <p className="dim">Polling telemetry data and model inference pipelines from FastAPI (:8000)</p>
      </div>
    );
  }

  const cfg = machineType !== "all" ? MACHINE_CONFIGS[machineType] : null;
  const isEmptyDatabase = readings.length === 0 && !fetchError;

  return (
    <div className="dashboard-container">

      {/* Machine type selector */}
      <div className="machine-type-selector">
        <span className="selector-label">Fleet View</span>
        <div className="machine-type-tabs">
          {MACHINE_TYPES.map(t => {
            const info = TYPE_LABELS[t];
            return (
              <button
                key={t}
                className={`machine-tab ${machineType === t ? "machine-tab-active" : ""}`}
                style={machineType === t && info.color ? { "--tab-color": info.color } : {}}
                onClick={() => setMachineType(t)}
              >
                <span>{info.icon}</span>
                <span>{info.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Top Status Banner */}
      <div className="status-banner">
        <div className="status-indicator-box">
          <span className="section-label">
            {machineType === "all" ? "Fleet-Wide Telemetry" : `${cfg?.label} Telemetry`}
          </span>
          {lastReading ? (
            <div className={`status-pill ${lastReading.is_anomaly ? "status-alert" : "status-ok"}`}>
              <span className="pulse-dot"></span>
              <strong>{lastReading.is_anomaly ? "CRITICAL: ANOMALY DETECTED" : "NOMINAL: SYSTEM NORMAL"}</strong>
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
              {lastReading.machine_type && (
                <div className="detail-item">
                  <span className="dim">Type:</span>{" "}
                  <code>{lastReading.machine_type}</code>
                </div>
              )}
              <div className="detail-item">
                <span className="dim">Last Reading:</span>{" "}
                <span className="font-mono">{lastReading.timestamp?.substring(11, 19) ?? "--"}</span>
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

      {fetchError && <div className="alert-box error"><span>⚠️ {fetchError}</span></div>}
      {isEmptyDatabase && (
        <div className="alert-box info">
          <span>ℹ️ <strong>Fresh Database Detected:</strong> No readings yet. Run{" "}
            <code>python ml/simulator.py</code>, <code>ml/simulator_azure.py</code>, or{" "}
            <code>ml/simulator_pump.py</code> to start streaming data.</span>
        </div>
      )}

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card card-total">
          <div className="kpi-icon-wrap">📊</div>
          <div className="kpi-content">
            <span className="kpi-label">Total Readings</span>
            <span className="kpi-value">{stats ? stats.total_readings.toLocaleString() : "--"}</span>
            <span className="kpi-sub">
              {machineType === "all" ? "All fleet types" : `${cfg?.label} only`}
            </span>
          </div>
        </div>

        <div className="kpi-card card-anomalies">
          <div className="kpi-icon-wrap icon-warn">⚠️</div>
          <div className="kpi-content">
            <span className="kpi-label">Anomalies Detected</span>
            <span className={`kpi-value ${stats?.total_anomalies > 0 ? "text-warn" : ""}`}>
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
              {stats ? (stats.flagged_by.isolation_forest_only + stats.flagged_by.both_models) : "--"}
            </span>
            <span className="kpi-sub">
              Exclusive: {stats?.flagged_by.isolation_forest_only ?? 0} | Shared: {stats?.flagged_by.both_models ?? 0}
            </span>
          </div>
        </div>

        <div className="kpi-card card-ae">
          <div className="kpi-icon-wrap icon-purple">🧠</div>
          <div className="kpi-content">
            <span className="kpi-label">Autoencoder (MSE)</span>
            <span className="kpi-value text-purple">
              {stats ? (stats.flagged_by.autoencoder_only + stats.flagged_by.both_models) : "--"}
            </span>
            <span className="kpi-sub">
              Exclusive: {stats?.flagged_by.autoencoder_only ?? 0} | Shared: {stats?.flagged_by.both_models ?? 0}
            </span>
          </div>
        </div>
      </div>

      {/* "All" fleet overview */}
      {machineType === "all" && (
        <>
          <div className="section-title-row">
            <h3>🏭 Live Fleet Overview</h3>
            <span className="dim text-sm">Select a machine type to drill into its sensor detail view</span>
          </div>
          <FleetOverviewGrid fleet={fleet} onSelect={setMachineType} />
        </>
      )}

      {/* Detail view for a specific machine type */}
      {machineType !== "all" && cfg && (
        <>
          <div className="section-title-row">
            <h3>{cfg.icon} {cfg.label} — Real-Time Sensor Charts</h3>
            <span className="dim text-sm">{cfg.description}</span>
          </div>
          <SensorCharts readings={readings} config={cfg} />
          <AnomalyTable anomalies={anomalies} config={cfg} isEmpty={isEmptyDatabase} />
        </>
      )}

      {/* Combined anomaly table for "All" view */}
      {machineType === "all" && !isEmptyDatabase && (
        <AnomalyTable anomalies={anomalies} config={MACHINE_CONFIGS.milling_machine} isEmpty={isEmptyDatabase} />
      )}
    </div>
  );
}

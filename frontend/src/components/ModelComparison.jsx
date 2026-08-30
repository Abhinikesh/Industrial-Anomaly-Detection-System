import React, { useState, useEffect } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

const MACHINE_TYPES = [
  { key: "all",             label: "All Combined",    icon: "🏭", color: "#94a3b8" },
  { key: "milling_machine", label: "Milling Machine", icon: "⚙️", color: "#38bdf8" },
  { key: "fleet_machine",   label: "Azure Fleet",     icon: "☁️", color: "#06b6d4" },
  { key: "water_pump",      label: "Water Pump",      icon: "💧", color: "#10b981" },
];

// Per-machine-type architecture notes shown in the model cards
const ARCH_NOTES = {
  milling_machine: {
    if: "100 isolation trees · 5 features (air_temp, process_temp, rpm, torque, tool_wear) · 3.4% contamination",
    ae: "5 → 3 → 2 → 3 → 5 bottleneck · trained on normal-only data · threshold = mean + 2σ",
  },
  fleet_machine: {
    if: "150 isolation trees · 4 features (voltage, rotation, pressure, vibration) · 2.04% contamination",
    ae: "4 → 3 → 2 → 3 → 4 bottleneck · tightly coupled sensor channels benefit encoder · threshold = mean + 2σ",
  },
  water_pump: {
    if: "150 isolation trees · 15 top-correlated sensors · 6.57% contamination",
    ae: "15 → 10 → 5 → 10 → 15 bottleneck · captures 5 dominant normal operating modes · BatchNorm + Dropout(0.1)",
  },
  all: {
    if: "Combined across all fleet types (milling machine, Azure fleet, water pump)",
    ae: "Combined across all fleet types — note: metrics mix different model pairs and sensor spaces",
  },
};

function MetricBar({ value, max = 1 }) {
  const pct = Math.min((value ?? 0) / max, 1) * 100;
  return (
    <div className="metric-bar-track">
      <div className="metric-bar-fill" style={{ width: `${pct}%` }} />
      <span className="metric-bar-value">{(value ?? 0).toFixed(3)}</span>
    </div>
  );
}

function ModelCard({ title, icon, badgeLabel, badgeClass, flagged, flaggedPct, data, archNote, colorClass }) {
  return (
    <div className={`model-card ${colorClass}`}>
      <div className="model-card-header">
        <div>
          <h3>{icon} {title}</h3>
          <span className={`badge-model-type ${badgeClass}`}>{badgeLabel}</span>
        </div>
        <span className={`flagged-badge ${colorClass === "card-ae-theme" ? "text-purple" : "text-blue"}`}>
          {flagged ?? 0} flagged ({flaggedPct ?? 0}%)
        </span>
      </div>

      <div className="metrics-list">
        <div className="metric-row">
          <span className="metric-name">Precision:</span>
          <MetricBar value={data?.precision} />
        </div>
        <div className="metric-row highlight-metric">
          <span className="metric-name">Recall:</span>
          <MetricBar value={data?.recall} />
        </div>
        <div className="metric-row highlight-metric">
          <span className="metric-name">F1-Score:</span>
          <MetricBar value={data?.f1} />
        </div>

        <div className="matrix-subgrid">
          <div className="matrix-cell">
            <span className="dim text-xs">True Pos (TP)</span>
            <strong className="font-mono">{data?.tp ?? 0}</strong>
          </div>
          <div className="matrix-cell">
            <span className="dim text-xs">False Pos (FP)</span>
            <strong className="font-mono text-warn">{data?.fp ?? 0}</strong>
          </div>
          <div className="matrix-cell">
            <span className="dim text-xs">False Neg (FN)</span>
            <strong className="font-mono text-warn">{data?.fn ?? 0}</strong>
          </div>
          <div className="matrix-cell">
            <span className="dim text-xs">True Neg (TN)</span>
            <strong className="font-mono text-emerald">{data?.tn ?? 0}</strong>
          </div>
        </div>
      </div>

      <div className="model-notes">
        <strong>Architecture:</strong>
        <p>{archNote}</p>
      </div>
    </div>
  );
}

// Compact per-type summary card for the overview grid
function TypeSummaryCard({ mt, data, onSelect }) {
  if (!data) return null;
  const winner = (data.autoencoder?.f1 ?? 0) >= (data.isolation_forest?.f1 ?? 0) ? "AE" : "IF";
  const winnerF1 = Math.max(data.autoencoder?.f1 ?? 0, data.isolation_forest?.f1 ?? 0);
  return (
    <div
      className="type-summary-card"
      style={{ "--type-color": mt.color }}
      onClick={() => onSelect(mt.key)}
    >
      <div className="type-summary-top">
        <span className="fleet-icon">{mt.icon}</span>
        <div>
          <div className="fleet-machine-label">{mt.label}</div>
          <div className="dim text-xs">{data.total_readings.toLocaleString()} readings · {data.true_failures} failures</div>
        </div>
      </div>
      <div className="type-summary-metrics">
        <div>
          <span className="dim text-xs">Best F1</span>
          <strong className="font-mono text-emerald">{winnerF1.toFixed(3)}</strong>
          <span className="dim text-xs"> ({winner} wins)</span>
        </div>
        <div>
          <span className="dim text-xs">Failure rate</span>
          <strong className="font-mono">
            {data.total_readings > 0 ? ((data.true_failures / data.total_readings) * 100).toFixed(1) : "0.0"}%
          </strong>
        </div>
      </div>
      <button className="btn-view-detail">Details →</button>
    </div>
  );
}

export default function ModelComparison() {
  const [selectedType, setSelectedType] = useState("all");
  const [dataByType, setDataByType]     = useState({});
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [lastRefreshed, setLastRefreshed] = useState(null);

  const fetchAll = async () => {
    setLoading(true);
    try {
      // Fetch comparison stats for all types in parallel
      const results = await Promise.all(
        MACHINE_TYPES.map(mt =>
          axios.get(`${API_BASE}/readings/model-comparison${mt.key !== "all" ? `?machine_type=${mt.key}` : ""}`)
            .then(r => [mt.key, r.data])
            .catch(() => [mt.key, null])
        )
      );
      const byType = Object.fromEntries(results);
      setDataByType(byType);
      setLastRefreshed(new Date().toLocaleTimeString());
      setError(null);
    } catch {
      setError("Connection lost to backend on :8000.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  if (loading && Object.keys(dataByType).length === 0) {
    return (
      <div className="dashboard-loading-state">
        <div className="spinner"></div>
        <h3>Computing Model Benchmark Telemetry...</h3>
        <p className="dim">Aggregating precision, recall, and agreement metrics per fleet type</p>
      </div>
    );
  }

  const data    = dataByType[selectedType];
  const ifData  = data?.isolation_forest || {};
  const aeData  = data?.autoencoder || {};
  const agree   = data?.agreement || {};
  const isEmpty = data?.total_readings === 0;
  const notes   = ARCH_NOTES[selectedType] || ARCH_NOTES.all;
  const activeMT = MACHINE_TYPES.find(m => m.key === selectedType);

  const nonAllTypes = MACHINE_TYPES.filter(m => m.key !== "all");

  return (
    <div className="model-comparison-container">
      {/* Header */}
      <div className="section-header-row">
        <div>
          <h2>Model Benchmark &amp; Live Evaluation</h2>
          <p className="dim">Real-time streaming evaluation calculated against true failure labels in MongoDB.</p>
        </div>
        <div className="header-actions">
          {lastRefreshed && <span className="dim text-sm">Last Synced: <span className="font-mono">{lastRefreshed}</span></span>}
          <button className="btn-refresh" onClick={fetchAll} disabled={loading}>
            {loading ? "Refreshing..." : "🔄 Refresh Metrics"}
          </button>
        </div>
      </div>

      {error && <div className="alert-box error"><span>⚠️ {error}</span></div>}

      {/* Machine type selector */}
      <div className="machine-type-selector">
        <span className="selector-label">Filter by Fleet</span>
        <div className="machine-type-tabs">
          {MACHINE_TYPES.map(mt => (
            <button
              key={mt.key}
              className={`machine-tab ${selectedType === mt.key ? "machine-tab-active" : ""}`}
              style={selectedType === mt.key ? { "--tab-color": mt.color } : {}}
              onClick={() => setSelectedType(mt.key)}
            >
              <span>{mt.icon}</span>
              <span>{mt.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Per-type summary overview (shown only in "all" mode) */}
      {selectedType === "all" && (
        <>
          <div className="section-title-row">
            <h3>📊 Per-Fleet Model Performance Overview</h3>
            <span className="dim text-sm">Click a card to drill into that fleet's detailed metrics</span>
          </div>
          <div className="fleet-overview-grid">
            {nonAllTypes.map(mt => (
              <TypeSummaryCard
                key={mt.key}
                mt={mt}
                data={dataByType[mt.key]}
                onSelect={setSelectedType}
              />
            ))}
          </div>
          <div className="alert-box info">
            <span>ℹ️ Combined "All" metrics below mix different model pairs and sensor spaces — per-fleet numbers are more meaningful for evaluation.</span>
          </div>
        </>
      )}

      {/* Overview KPI row */}
      <div className="benchmark-overview-grid">
        <div className="overview-metric-card card-total">
          <span className="kpi-label">Sample Pool</span>
          <span className="kpi-value">{data ? data.total_readings.toLocaleString() : "--"}</span>
          <span className="kpi-sub">
            {data?.total_readings > 0
              ? `${data.true_failures} failures (${((data.true_failures / data.total_readings) * 100).toFixed(1)}% rate)`
              : "No data yet"}
          </span>
        </div>

        <div className="overview-metric-card card-agree">
          <span className="kpi-label">Model Consensus</span>
          <span className="kpi-value text-emerald">{data ? `${agree.agreement_pct}%` : "--"}</span>
          <span className="kpi-sub">{agree.agreed_count?.toLocaleString()} of {data?.total_readings?.toLocaleString()} readings</span>
        </div>

        <div className="overview-metric-card card-ae">
          <span className="kpi-label">Best F1</span>
          <span className="kpi-value text-purple font-mono">
            {Math.max(aeData.f1 ?? 0, ifData.f1 ?? 0).toFixed(3)}
          </span>
          <span className="kpi-sub">
            {(aeData.f1 ?? 0) >= (ifData.f1 ?? 0) ? "Autoencoder leads" : "Isolation Forest leads"}
          </span>
        </div>

        <div className="overview-metric-card card-advantage">
          <span className="kpi-label">Recall Advantage</span>
          <span className="kpi-value text-purple font-mono">
            {aeData.recall != null && ifData.recall != null
              ? `${aeData.recall >= ifData.recall ? "+" : ""}${((aeData.recall - ifData.recall) * 100).toFixed(1)}% AE`
              : "--"}
          </span>
          <span className="kpi-sub">AE vs IF on {activeMT?.label}</span>
        </div>
      </div>

      {isEmpty && (
        <div className="alert-box info">
          <span>ℹ️ <strong>No data for {activeMT?.label}:</strong> Start the simulator to accumulate records.</span>
        </div>
      )}

      {/* Side-by-side model cards */}
      <div className="comparison-cards-grid">
        <ModelCard
          title="Isolation Forest"
          icon="🌲"
          badgeLabel="Tree-based Random Partitioning"
          badgeClass=""
          colorClass="card-if-theme"
          flagged={ifData.flagged}
          flaggedPct={ifData.flagged_pct}
          data={ifData}
          archNote={notes.if}
        />
        <ModelCard
          title="Deep Autoencoder"
          icon="🧠"
          badgeLabel="PyTorch (MSE Reconstruction)"
          badgeClass="badge-purple"
          colorClass="card-ae-theme featured"
          flagged={aeData.flagged}
          flaggedPct={aeData.flagged_pct}
          data={aeData}
          archNote={notes.ae}
        />
      </div>

      {/* Agreement breakdown */}
      <div className="agreement-card">
        <div className="agreement-header">
          <div>
            <h3>Model Agreement &amp; Divergence Analysis</h3>
            <p className="dim">How frequently both models reach consensus vs where they diverge:</p>
          </div>
          <span className="agreement-stat-pill text-emerald">{agree.agreement_pct ?? "--"}% Agreement</span>
        </div>

        <div className="agreement-progress-bar">
          <div className="bar-segment bar-both"
            style={{ width: `${(agree.both_flagged / (data?.total_readings || 1)) * 100}%` }}
            title={`Both Flagged: ${agree.both_flagged}`} />
          <div className="bar-segment bar-ae"
            style={{ width: `${(agree.ae_only / (data?.total_readings || 1)) * 100}%` }}
            title={`AE Only: ${agree.ae_only}`} />
          <div className="bar-segment bar-if"
            style={{ width: `${(agree.iso_only / (data?.total_readings || 1)) * 100}%` }}
            title={`IF Only: ${agree.iso_only}`} />
          <div className="bar-segment bar-normal"
            style={{ width: `${(agree.both_normal / (data?.total_readings || 1)) * 100}%` }}
            title={`Both Normal: ${agree.both_normal}`} />
        </div>

        <div className="agreement-legend-grid">
          <div className="legend-item"><span className="dot dot-both"></span><div><strong>Both Flagged:</strong> <span className="font-mono">{agree.both_flagged ?? 0}</span></div></div>
          <div className="legend-item"><span className="dot dot-ae"></span><div><strong>Autoencoder Only:</strong> <span className="font-mono">{agree.ae_only ?? 0}</span></div></div>
          <div className="legend-item"><span className="dot dot-if"></span><div><strong>Isolation Forest Only:</strong> <span className="font-mono">{agree.iso_only ?? 0}</span></div></div>
          <div className="legend-item"><span className="dot dot-normal"></span><div><strong>Both Normal:</strong> <span className="font-mono">{agree.both_normal ?? 0}</span></div></div>
        </div>
      </div>

      {/* Evaluation artifacts */}
      <div className="artifacts-gallery-section">
        <div className="gallery-header">
          <h3>Training &amp; Evaluation Artifacts</h3>
          <p className="dim">Offline visual evaluations generated during model development</p>
        </div>
        <div className="gallery-grid">
          {[
            { src: `${API_BASE}/results/autoencoder/confusion_matrix.png`,           title: "AE · Confusion Matrix (Milling)",  tag: "Offline Eval",  caption: "Evaluated with mean + 2σ reconstruction threshold" },
            { src: `${API_BASE}/results/autoencoder/reconstruction_error.png`,       title: "AE · Error Distribution (Milling)",tag: "MSE Error",     caption: "Normal vs anomalous separation along error scale" },
            { src: `${API_BASE}/results/isolation_forest/confusion_matrix.png`,      title: "IF · Confusion Matrix (Milling)",  tag: "Offline Eval",  caption: "Performance at 3.4% estimated contamination rate" },
            { src: `${API_BASE}/results/isolation_forest/score_distribution.png`,    title: "IF · Score Distribution (Milling)",tag: "Path Length",   caption: "Average anomaly score distribution across 100 trees" },
            { src: `${API_BASE}/results/azure_pdm/ae_confusion_matrix.png`,          title: "AE · Confusion Matrix (Azure)",    tag: "Azure PdM",     caption: "4-sensor fleet telemetry autoencoder evaluation" },
            { src: `${API_BASE}/results/azure_pdm/if_score_distribution.png`,        title: "IF · Score Distribution (Azure)",  tag: "Azure PdM",     caption: "Azure fleet anomaly score separation" },
            { src: `${API_BASE}/results/pump_sensor/ae_confusion_matrix.png`,        title: "AE · Confusion Matrix (Pump)",     tag: "Pump Sensor",   caption: "15-sensor water pump autoencoder evaluation" },
            { src: `${API_BASE}/results/pump_sensor/ae_reconstruction_error.png`,    title: "AE · Error Distribution (Pump)",   tag: "Pump Sensor",   caption: "Normal vs RECOVERING/BROKEN reconstruction error" },
            { src: `${API_BASE}/results/eda/correlation_heatmap.png`,                title: "EDA · Sensor Correlation Heatmap", tag: "Exploratory",   caption: "Cross-sensor correlations including failure ground truth" },
            { src: `${API_BASE}/results/eda/sensor_distributions.png`,               title: "EDA · Sensor Histograms",          tag: "Distribution",  caption: "Normal vs anomalous distributions (milling machine)" },
          ].map(({ src, title, tag, caption }) => (
            <div key={src} className="artifact-image-card">
              <div className="artifact-card-header">
                <h4>{title}</h4>
                <span className="badge-tag">{tag}</span>
              </div>
              <div className="artifact-img-wrap">
                <img
                  src={src}
                  alt={title}
                  className="artifact-img"
                  onError={e => { e.target.style.display = "none"; }}
                />
              </div>
              <span className="artifact-caption">{caption}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

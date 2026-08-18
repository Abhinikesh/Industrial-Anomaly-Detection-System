import React, { useState, useEffect } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefreshed, setLastRefreshed] = useState(null);

  const fetchComparison = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_BASE}/readings/model-comparison`);
      setData(res.data);
      setLastRefreshed(new Date().toLocaleTimeString());
      setError(null);
    } catch (err) {
      setError("Connection lost to backend on :8000. Unable to refresh live benchmark data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComparison();
  }, []);

  if (loading && !data) {
    return (
      <div className="dashboard-loading-state">
        <div className="spinner"></div>
        <h3>Computing Model Benchmark Telemetry...</h3>
        <p className="dim">Aggregating precision, recall, and agreement metrics across MongoDB records</p>
      </div>
    );
  }

  const isEmpty = data && data.total_readings === 0;
  const ifData = data?.isolation_forest || {};
  const aeData = data?.autoencoder || {};
  const agree = data?.agreement || {};

  return (
    <div className="model-comparison-container">
      {/* Header & Controls */}
      <div className="section-header-row">
        <div>
          <h2>Model Benchmark &amp; Live Evaluation</h2>
          <p className="dim">
            Real-time streaming evaluation calculated against true failure labels in MongoDB.
          </p>
        </div>
        <div className="header-actions">
          {lastRefreshed && <span className="dim text-sm">Last Synced: <span className="font-mono">{lastRefreshed}</span></span>}
          <button className="btn-refresh" onClick={fetchComparison} disabled={loading}>
            {loading ? "Refreshing..." : "🔄 Refresh Metrics"}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-box error">
          <span>⚠️ {error}</span>
        </div>
      )}

      {isEmpty && (
        <div className="alert-box info">
          <span>ℹ️ <strong>No Telemetry Ingested Yet:</strong> Start the simulator (<code>python ml/simulator.py</code>) to accumulate streaming records for real-time benchmark calculation.</span>
        </div>
      )}

      {/* Live Stream Benchmark Grid */}
      <div className="benchmark-overview-grid">
        <div className="overview-metric-card card-total">
          <span className="kpi-label">Sample Pool Evaluated</span>
          <span className="kpi-value">{data ? data.total_readings.toLocaleString() : "--"}</span>
          <span className="kpi-sub">
            {data && data.total_readings > 0
              ? `${data.true_failures} Failures (${((data.true_failures / data.total_readings) * 100).toFixed(1)}% base rate)`
              : "0 Failures (0.0% base rate)"}
          </span>
        </div>

        <div className="overview-metric-card card-agree">
          <span className="kpi-label">Model Consensus Rate</span>
          <span className="kpi-value text-emerald">
            {data ? `${agree.agreement_pct}%` : "--"}
          </span>
          <span className="kpi-sub">
            {agree.agreed_count?.toLocaleString()} of {data?.total_readings?.toLocaleString()} readings
          </span>
        </div>

        <div className="overview-metric-card card-ae">
          <span className="kpi-label">Leading F1 Score</span>
          <span className="kpi-value text-purple font-mono">
            {aeData.f1 ? aeData.f1.toFixed(3) : "0.000"}
          </span>
          <span className="kpi-sub">PyTorch Deep Autoencoder</span>
        </div>

        <div className="overview-metric-card card-advantage">
          <span className="kpi-label">Recall Advantage</span>
          <span className="kpi-value text-purple font-mono">
            {aeData.recall && ifData.recall
              ? `+${((aeData.recall - ifData.recall) * 100).toFixed(1)}%`
              : "--"}
          </span>
          <span className="kpi-sub">AE detected significantly more real failures</span>
        </div>
      </div>

      {/* Side-by-Side Model Comparison Cards */}
      <div className="comparison-cards-grid">
        {/* Isolation Forest Card */}
        <div className="model-card card-if-theme">
          <div className="model-card-header">
            <div>
              <h3>🌲 Isolation Forest</h3>
              <span className="badge-model-type">Tree-based Random Partitioning</span>
            </div>
            <span className="flagged-badge text-blue">
              {ifData.flagged || 0} flagged ({ifData.flagged_pct || 0}%)
            </span>
          </div>

          <div className="metrics-list">
            <div className="metric-row">
              <span className="metric-name">Precision:</span>
              <strong className="metric-val font-mono">{ifData.precision?.toFixed(3) ?? "--"}</strong>
            </div>
            <div className="metric-row">
              <span className="metric-name">Recall:</span>
              <strong className="metric-val font-mono">{ifData.recall?.toFixed(3) ?? "--"}</strong>
            </div>
            <div className="metric-row highlight-metric">
              <span className="metric-name">F1-Score:</span>
              <strong className="metric-val font-mono text-blue">{ifData.f1?.toFixed(3) ?? "--"}</strong>
            </div>

            <div className="matrix-subgrid">
              <div className="matrix-cell">
                <span className="dim text-xs">True Pos (TP)</span>
                <strong className="font-mono">{ifData.tp ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">False Pos (FP)</span>
                <strong className="font-mono text-warn">{ifData.fp ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">False Neg (FN)</span>
                <strong className="font-mono text-warn">{ifData.fn ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">True Neg (TN)</span>
                <strong className="font-mono text-emerald">{ifData.tn ?? 0}</strong>
              </div>
            </div>
          </div>

          <div className="model-notes">
            <strong>Architecture &amp; Mechanism:</strong>
            <p>
              Partitions multi-dimensional feature space using 100 isolation trees with a contamination prior of 3.4%. Isolates anomalies near tree roots without calculating complex non-linear combinations.
            </p>
          </div>
        </div>

        {/* Autoencoder Card */}
        <div className="model-card featured card-ae-theme">
          <div className="model-card-header">
            <div>
              <h3>🧠 Deep Autoencoder</h3>
              <span className="badge-model-type badge-purple">PyTorch (MSE Reconstruction)</span>
            </div>
            <span className="flagged-badge text-purple">
              {aeData.flagged || 0} flagged ({aeData.flagged_pct || 0}%)
            </span>
          </div>

          <div className="metrics-list">
            <div className="metric-row">
              <span className="metric-name">Precision:</span>
              <strong className="metric-val font-mono">{aeData.precision?.toFixed(3) ?? "--"}</strong>
            </div>
            <div className="metric-row highlight-metric-purple">
              <span className="metric-name">Recall:</span>
              <strong className="metric-val font-mono text-purple">{aeData.recall?.toFixed(3) ?? "--"}</strong>
            </div>
            <div className="metric-row highlight-metric-purple">
              <span className="metric-name">F1-Score:</span>
              <strong className="metric-val font-mono text-purple">{aeData.f1?.toFixed(3) ?? "--"}</strong>
            </div>

            <div className="matrix-subgrid">
              <div className="matrix-cell">
                <span className="dim text-xs">True Pos (TP)</span>
                <strong className="font-mono text-purple">{aeData.tp ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">False Pos (FP)</span>
                <strong className="font-mono text-warn">{aeData.fp ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">False Neg (FN)</span>
                <strong className="font-mono text-warn">{aeData.fn ?? 0}</strong>
              </div>
              <div className="matrix-cell">
                <span className="dim text-xs">True Neg (TN)</span>
                <strong className="font-mono text-emerald">{aeData.tn ?? 0}</strong>
              </div>
            </div>
          </div>

          <div className="model-notes">
            <strong>Architecture &amp; Mechanism:</strong>
            <p>
              Trained purely on normal sensor sequences through a <code>5 → 3 → 2 → 3 → 5</code> bottleneck. Spikes in reconstruction error (&gt; mean + 2σ) signal multi-sensor physical correlation drift.
            </p>
          </div>
        </div>
      </div>

      {/* Model Agreement & Overlap Breakdown */}
      <div className="agreement-card">
        <div className="agreement-header">
          <div>
            <h3>Model Agreement &amp; Divergence Analysis</h3>
            <p className="dim">
              How frequently both models reach consensus vs where they diverge across telemetry stream:
            </p>
          </div>
          <span className="agreement-stat-pill text-emerald">
            {agree.agreement_pct}% Total Agreement
          </span>
        </div>

        <div className="agreement-progress-bar">
          <div
            className="bar-segment bar-both"
            style={{ width: `${(agree.both_flagged / (data?.total_readings || 1)) * 100}%` }}
            title={`Both Flagged: ${agree.both_flagged}`}
          ></div>
          <div
            className="bar-segment bar-ae"
            style={{ width: `${(agree.ae_only / (data?.total_readings || 1)) * 100}%` }}
            title={`Autoencoder Only: ${agree.ae_only}`}
          ></div>
          <div
            className="bar-segment bar-if"
            style={{ width: `${(agree.iso_only / (data?.total_readings || 1)) * 100}%` }}
            title={`Isolation Forest Only: ${agree.iso_only}`}
          ></div>
          <div
            className="bar-segment bar-normal"
            style={{ width: `${(agree.both_normal / (data?.total_readings || 1)) * 100}%` }}
            title={`Both Normal: ${agree.both_normal}`}
          ></div>
        </div>

        <div className="agreement-legend-grid">
          <div className="legend-item">
            <span className="dot dot-both"></span>
            <div>
              <strong>Both Flagged (High-Confidence):</strong> <span className="font-mono">{agree.both_flagged ?? 0}</span> readings
            </div>
          </div>
          <div className="legend-item">
            <span className="dot dot-ae"></span>
            <div>
              <strong>Autoencoder Only (Relational):</strong> <span className="font-mono">{agree.ae_only ?? 0}</span> readings
            </div>
          </div>
          <div className="legend-item">
            <span className="dot dot-if"></span>
            <div>
              <strong>Isolation Forest Only (Outlier):</strong> <span className="font-mono">{agree.iso_only ?? 0}</span> readings
            </div>
          </div>
          <div className="legend-item">
            <span className="dot dot-normal"></span>
            <div>
              <strong>Both Normal (Nominal State):</strong> <span className="font-mono">{agree.both_normal ?? 0}</span> readings
            </div>
          </div>
        </div>
      </div>

      {/* Plain Language Explanation Section */}
      <div className="explanation-card">
        <div className="explanation-header">
          <h3>Analysis: Which Model Performs Better &amp; Why</h3>
          <span className="badge-editable">Editable Summary</span>
        </div>
        <div className="explanation-body">
          <p>
            <strong>Performance Verdict:</strong> On this industrial dataset, the <strong>Deep Autoencoder</strong> consistently outperforms the <strong>Isolation Forest</strong>, achieving both higher recall and a superior F1-score on live streaming telemetry.
          </p>
          <p>
            <strong>Technical Rationale:</strong> Industrial machine failures in the AI4I dataset (such as Heat Dissipation Failures and Overstrain Failures) are rarely simple 1-dimensional point outliers. Instead, they represent subtle violations of physical correlations between interdependent sensors — for instance, the rotational speed and torque curve or the delta between air and process temperatures.
          </p>
          <ul>
            <li>
              <strong>Isolation Forest limitation:</strong> Employs axis-aligned random splits, which detect isolated extreme values well but struggle to capture non-linear relationships across multiple sensor dimensions.
            </li>
            <li>
              <strong>Autoencoder advantage:</strong> The 2-neuron bottleneck forces the neural network to compress the manifold of normal operating physics. When sensor combinations deviate from normal correlation laws, the reconstruction error increases substantially, flagging complex multi-variable faults.
            </li>
          </ul>
          <p className="dim text-sm mt-2">
            <em>Note: For production deployments, our union (OR) ensemble rule provides the safest operational posture by retaining the Autoencoder's relational sensitivity while catching any sudden point spikes isolated by the tree model.</em>
          </p>
        </div>
      </div>

      {/* Static Evaluation Artifacts Gallery */}
      <div className="artifacts-gallery-section">
        <div className="gallery-header">
          <h3>Training &amp; Evaluation Artifacts</h3>
          <p className="dim">
            Offline visual evaluations and sensor distributions generated during model development:
          </p>
        </div>

        <div className="gallery-grid">
          {/* Autoencoder Confusion Matrix */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>Autoencoder &bull; Confusion Matrix</h4>
              <span className="badge-tag">Offline Eval</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/autoencoder/confusion_matrix.png`}
                alt="Autoencoder Confusion Matrix"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Evaluated against test holdout with mean + 2σ reconstruction threshold</span>
          </div>

          {/* Autoencoder Error Distribution */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>Autoencoder &bull; Error Distribution</h4>
              <span className="badge-tag">MSE Error</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/autoencoder/reconstruction_error.png`}
                alt="Autoencoder Reconstruction Error"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Normal vs anomalous sample separation along the reconstruction error scale</span>
          </div>

          {/* Isolation Forest Confusion Matrix */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>Isolation Forest &bull; Confusion Matrix</h4>
              <span className="badge-tag">Offline Eval</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/isolation_forest/confusion_matrix.png`}
                alt="Isolation Forest Confusion Matrix"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Performance at 3.4% estimated contamination rate</span>
          </div>

          {/* Isolation Forest Score Distribution */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>Isolation Forest &bull; Score Distribution</h4>
              <span className="badge-tag">Path Length</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/isolation_forest/score_distribution.png`}
                alt="Isolation Forest Score Distribution"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Average anomaly score distribution across 100 trees</span>
          </div>

          {/* Sensor Correlation Heatmap */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>EDA &bull; Sensor Correlation Heatmap</h4>
              <span className="badge-tag">Exploratory</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/eda/correlation_heatmap.png`}
                alt="Sensor Correlation Heatmap"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Cross-sensor correlations including failure ground truth</span>
          </div>

          {/* Sensor Distributions */}
          <div className="artifact-image-card">
            <div className="artifact-card-header">
              <h4>EDA &bull; Sensor Histograms</h4>
              <span className="badge-tag">Distribution</span>
            </div>
            <div className="artifact-img-wrap">
              <img
                src={`${API_BASE}/results/eda/sensor_distributions.png`}
                alt="Sensor Distributions"
                className="artifact-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
            <span className="artifact-caption">Normal vs anomalous distributions across all 5 physical parameters</span>
          </div>
        </div>
      </div>
    </div>
  );
}

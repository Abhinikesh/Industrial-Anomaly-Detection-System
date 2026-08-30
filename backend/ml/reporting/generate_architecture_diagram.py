import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../../docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_PATH = os.path.join(OUTPUT_DIR, "architecture_diagram.png")

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0b0f19')

# Define boxes
boxes = [
    {"title": "1. Telemetry Simulator", "sub": "backend/ml/simulator.py\n• AI4I 2020 Dataset Replay\n• 5 Sensor Feeds (Temp, RPM, etc.)\n• 1s Stream Cadence (HTTP POST)", "xy": (0.05, 0.55), "wh": (0.24, 0.35), "color": "#0284c7"},
    {"title": "2. FastAPI Ingest Engine", "sub": "backend/app/routes/ingest.py\n• POST /ingest Endpoint\n• CORS & Pydantic Validation\n• Async Telemetry Ingestion", "xy": (0.37, 0.55), "wh": (0.26, 0.35), "color": "#2563eb"},
    {"title": "3. Dual ML Scoring Pipeline", "sub": "backend/app/services/anomaly_service.py\n• StandardScaler Normalization\n• Isolation Forest (100 Trees)\n• PyTorch Autoencoder (5→3→2→3→5)\n• Union (OR) Decision Logic", "xy": (0.71, 0.55), "wh": (0.25, 0.35), "color": "#7c3aed"},
    {"title": "4. MongoDB Document Store", "sub": "Database: anomaly_detection\nCollection: sensor_readings\n• Indexed by timestamp (-1)\n• Compound index on is_anomaly\n• Full Telemetry & Model Scores", "xy": (0.71, 0.08), "wh": (0.25, 0.35), "color": "#059669"},
    {"title": "5. React Mission Control", "sub": "frontend/src/App.jsx (Port 3000)\n• Live Monitor: 3 Recharts Lines\n• Real-Time Anomaly Markers\n• Model Comparison Benchmark Tab\n• Polling /readings/* Endpoints", "xy": (0.20, 0.08), "wh": (0.43, 0.35), "color": "#d97706"},
]

# Draw boxes
for b in boxes:
    rect = patches.FancyBboxPatch(
        b["xy"], b["wh"][0], b["wh"][1],
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#151d2e",
        edgecolor=b["color"],
        linewidth=2,
        transform=ax.transAxes,
        zorder=2
    )
    ax.add_patch(rect)
    ax.text(
        b["xy"][0] + b["wh"][0]/2, b["xy"][1] + b["wh"][1] - 0.06,
        b["title"],
        color='#f8fafc', fontsize=11, fontweight='bold', ha='center', va='center',
        transform=ax.transAxes, zorder=3
    )
    ax.text(
        b["xy"][0] + b["wh"][0]/2, b["xy"][1] + b["wh"][1]/2 - 0.03,
        b["sub"],
        color='#cbd5e1', fontsize=8.5, ha='center', va='center',
        transform=ax.transAxes, zorder=3, multialignment='center'
    )

# Draw arrows connecting components
arrows = [
    # 1 -> 2
    ((0.29, 0.72), (0.37, 0.72), "HTTP POST (JSON)"),
    # 2 -> 3
    ((0.63, 0.72), (0.71, 0.72), "score_reading()"),
    # 3 -> 4
    ((0.835, 0.55), (0.835, 0.43), "save_reading()"),
    # 4 -> 5
    ((0.71, 0.25), (0.63, 0.25), "GET /readings/*"),
    # 2 -> 5 (routes serve data)
    ((0.50, 0.55), (0.50, 0.43), "REST API Queries"),
]

for start, end, label in arrows:
    ax.annotate(
        '', xy=end, xytext=start, xycoords='axes fraction',
        arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2, headwidth=8, headlength=8, shrink=0.02),
        zorder=4
    )
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2 + 0.03
    ax.text(mid_x, mid_y, label, color='#94a3b8', fontsize=7.5, ha='center', va='center', transform=ax.transAxes, zorder=5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.title("Industrial Anomaly Detection System — End-to-End System Architecture", color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()
print(f"Architecture diagram saved to {IMG_PATH}")

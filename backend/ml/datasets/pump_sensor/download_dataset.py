"""
Downloads the Pump Sensor Data from Kaggle.

Dataset : nphantawee/pump-sensor-data
File    : sensor.csv  (~124 MB, 220,320 rows, 52 sensors + machine_status)
Output  : data/raw/pump_sensor/sensor.csv

If you already downloaded sensor.csv manually and placed it at
data/raw/sensor.csv  OR  data/raw/pump_sensor/sensor.csv,
this script will find it and skip the download automatically.

Usage (from backend/ with venv active):
  python ml/datasets/pump_sensor/download_dataset.py
"""

import os
import sys
import shutil

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
# backend/ml/datasets/pump_sensor/ → four levels up → project root
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../"))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "pump_sensor")
OUTPUT_PATH   = os.path.join(RAW_DIR, "sensor.csv")
FALLBACK_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sensor.csv")

DATASET_SLUG = "nphantawee/pump-sensor-data"
KAGGLE_URL   = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"


def check_already_present() -> str | None:
    """Return the path if sensor.csv already exists in either expected location."""
    for path in [OUTPUT_PATH, FALLBACK_PATH]:
        if os.path.exists(path):
            return path
    return None


def download_via_kaggle_api() -> bool:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  ⚠  kaggle package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle>=1.6", "-q"])

    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()

    try:
        api.authenticate()
    except Exception as e:
        print(f"\n  ✗ Kaggle authentication failed: {e}")
        print(
            "\n  Set up credentials:\n"
            "    1. https://www.kaggle.com/settings → API → 'Create New Token'\n"
            "    2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "    3. chmod 600 ~/.kaggle/kaggle.json\n"
        )
        return False

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"\n  Downloading '{DATASET_SLUG}' from Kaggle...")
    try:
        api.dataset_download_files(DATASET_SLUG, path=RAW_DIR, unzip=True, quiet=False)
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False
    return True


def main():
    print("=" * 62)
    print("Pump Sensor Dataset — Downloader")
    print("=" * 62)

    existing = check_already_present()
    if existing:
        size_mb = os.path.getsize(existing) / (1024 ** 2)
        print(f"\n✓ sensor.csv already present at:\n  {existing}  ({size_mb:.0f} MB)")

        # If it's at the root fallback location, copy it to the canonical dir
        if existing == FALLBACK_PATH and not os.path.exists(OUTPUT_PATH):
            os.makedirs(RAW_DIR, exist_ok=True)
            shutil.copy2(FALLBACK_PATH, OUTPUT_PATH)
            print(f"\n  Copied to canonical location:\n  {OUTPUT_PATH}")

        print("\nSkipping download. Run preprocess.py next:\n"
              "  python ml/datasets/pump_sensor/preprocess.py")
        return

    success = download_via_kaggle_api()
    if not success:
        print(
            f"\n  Manual fallback:\n"
            f"    1. Visit: {KAGGLE_URL}\n"
            f"    2. Download sensor.csv\n"
            f"    3. Place it at: {OUTPUT_PATH}\n"
            f"    4. Run: python ml/datasets/pump_sensor/preprocess.py"
        )
        sys.exit(1)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
        print(f"\n✓ Download complete — {OUTPUT_PATH}  ({size_mb:.0f} MB)")
        print("\nNext step:\n  python ml/datasets/pump_sensor/preprocess.py")
    else:
        print("\n⚠  sensor.csv not found after download. Check Kaggle output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

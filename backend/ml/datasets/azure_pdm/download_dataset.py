"""
Downloads the Microsoft Azure Predictive Maintenance dataset from Kaggle.

Dataset : arnabbiswas1/microsoft-azure-predictive-maintenance
Files   : PdM_telemetry.csv, PdM_errors.csv, PdM_maint.csv,
          PdM_failures.csv, PdM_machines.csv
Output  : data/raw/azure_pdm/

Usage (from backend/ with venv active):
  python ml/datasets/azure_pdm/download_dataset.py
"""

import os
import sys
import shutil

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Navigate from backend/ml/datasets/azure_pdm/ → project root → data/raw/azure_pdm
RAW_DIR     = os.path.join(SCRIPT_DIR, "../../../../data/raw/azure_pdm")

DATASET_SLUG  = "arnabbiswas1/microsoft-azure-predictive-maintenance"
KAGGLE_URL    = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"

EXPECTED_FILES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
]


def all_files_present(directory: str) -> bool:
    return all(
        os.path.exists(os.path.join(directory, f))
        for f in EXPECTED_FILES
    )


def download_via_kaggle_api() -> bool:
    """
    Use the kaggle Python package to download and unzip the dataset.
    Returns True on success, False on failure.

    The kaggle package is a thin wrapper around the Kaggle REST API.
    It reads credentials from ~/.kaggle/kaggle.json automatically.
    """
    try:
        import kaggle  # noqa: F401 — just confirm it's importable
    except ImportError:
        print("  ⚠  kaggle package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle>=1.6", "-q"])
        print("  ✓ kaggle installed.")

    # Import after potential install
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
    api = KaggleApiExtended()

    try:
        api.authenticate()
    except Exception as e:
        print(f"\n  ✗ Kaggle authentication failed: {e}")
        print(
            "\n  ──────────────────────────────────────────────────────────────\n"
            "  To fix this, set up your Kaggle API credentials:\n"
            "    1. Go to  https://www.kaggle.com/settings → API\n"
            "    2. Click  'Create New Token'  → downloads kaggle.json\n"
            "    3. Move it to  ~/.kaggle/kaggle.json\n"
            "    4. Run:  chmod 600 ~/.kaggle/kaggle.json\n"
            "    5. Re-run this script.\n"
            "  ──────────────────────────────────────────────────────────────"
        )
        return False

    # Download + unzip directly into RAW_DIR
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"\n  Downloading dataset '{DATASET_SLUG}' from Kaggle...")
    print(f"  Target directory: {RAW_DIR}\n")

    try:
        api.dataset_download_files(
            DATASET_SLUG,
            path=RAW_DIR,
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False

    return True


def verify_and_report(directory: str) -> bool:
    """Check that all expected CSVs are present and print their sizes."""
    print("\n  Verifying downloaded files:")
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = os.path.join(directory, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 ** 2)
            print(f"    ✓  {fname:<28}  ({size_mb:.1f} MB)")
        else:
            print(f"    ✗  {fname}  — MISSING")
            all_ok = False
    return all_ok


def print_manual_fallback():
    print(
        "\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  MANUAL DOWNLOAD FALLBACK\n"
        "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  1. Visit: {KAGGLE_URL}\n"
        "  2. Click 'Download' (zip file, ~12 MB)\n"
        "  3. Unzip and copy these 5 CSVs to:\n"
        f"       {RAW_DIR}/\n"
        "         PdM_telemetry.csv\n"
        "         PdM_errors.csv\n"
        "         PdM_maint.csv\n"
        "         PdM_failures.csv\n"
        "         PdM_machines.csv\n"
        "  4. Then run:  python ml/datasets/azure_pdm/preprocess.py\n"
        "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def main():
    print("=" * 66)
    print("Azure Predictive Maintenance Dataset — Downloader")
    print("=" * 66)

    # Fast path: already downloaded
    if all_files_present(RAW_DIR):
        print(f"\nAll files already present in {RAW_DIR}")
        verify_and_report(RAW_DIR)
        print("\nSkipping download. Run preprocess.py next.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    success = download_via_kaggle_api()

    if success:
        ok = verify_and_report(RAW_DIR)
        if ok:
            print(
                f"\n✓  Download complete. All 5 CSVs saved to:\n"
                f"   {RAW_DIR}\n"
                "\nNext step:\n"
                "  python ml/datasets/azure_pdm/preprocess.py"
            )
        else:
            print("\n⚠  Some files missing after download. Check Kaggle output above.")
            print_manual_fallback()
            sys.exit(1)
    else:
        print_manual_fallback()
        sys.exit(1)


if __name__ == "__main__":
    main()

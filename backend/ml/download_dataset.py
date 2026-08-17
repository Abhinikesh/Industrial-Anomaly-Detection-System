"""
Downloads the AI4I 2020 Predictive Maintenance Dataset from the UCI ML Repository
and saves it as data/raw/ai4i2020.csv.

UCI hosts this dataset as a zip at:
  https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

No API key needed — UCI datasets are freely downloadable without authentication.
The zip contains a few files; we extract the CSV named 'ai4i2020.csv'.
"""

import os
import io
import zipfile
import urllib.request

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# resolve relative to this script so it works from any working directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(SCRIPT_DIR, "../../data/raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "ai4i2020.csv")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"Already downloaded: {OUTPUT_PATH}")
        return

    print(f"Downloading from UCI ML Repository...")
    print(f"  {DATASET_URL}")

    try:
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "Mozilla/5.0"}  # UCI sometimes blocks bare Python UA
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual fallback:")
        print("  1. Visit https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset")
        print("  2. Click 'Download' to get the zip")
        print(f"  3. Extract 'ai4i2020.csv' to {RAW_DIR}")
        return

    # unzip in memory and grab the CSV
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in zip. Contents: {zf.namelist()}")

        # the file is named 'ai4i2020.csv' inside the zip
        target = next((n for n in csv_names if "ai4i" in n.lower()), csv_names[0])
        print(f"Extracting '{target}' ...")
        with zf.open(target) as src, open(OUTPUT_PATH, "wb") as dst:
            dst.write(src.read())

    print(f"Saved to: {OUTPUT_PATH}")
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    download()

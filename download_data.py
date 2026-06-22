"""
Dataset bootstrap utility.

Source dataset: UCI Phishing Websites Data Set (30 structural URL features + a
binary `Result` label). The CSV is expected at ``data/phisingData.csv``.

If the file is already present, this script is a no-op. Otherwise it will fetch
the dataset from a URL provided via the ``DATA_URL`` environment variable, so no
source location is hard-coded into the project.
"""

import os
import urllib.request

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "phisingData.csv")
DATA_URL = os.getenv("DATA_URL")  # optional override; not hard-coded


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE)
        print(f"Dataset already present: {OUTPUT_FILE} ({size} bytes). Nothing to do.")
        return

    if not DATA_URL:
        print(
            "Dataset not found and no DATA_URL is configured.\n"
            f"Place the phishing CSV at '{OUTPUT_FILE}', or set the DATA_URL "
            "environment variable to a downloadable source and re-run."
        )
        return

    print("Downloading phishing dataset from DATA_URL ...")
    try:
        urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)
        print(f"Downloaded dataset to: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)} bytes)")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    main()

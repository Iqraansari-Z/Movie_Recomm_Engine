"""
Phase 1 - Data Acquisition
Downloads and extracts the MovieLens 1M dataset into data/raw/.

Usage:
    python src/download_data.py
"""
import zipfile
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ml-1m.zip"


def download_file(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def flatten_ml1m_folder(dest_dir: Path) -> None:
    nested = dest_dir / "ml-1m"
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        nested.rmdir()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "ratings.dat").exists():
        print(f"Dataset already present at {RAW_DIR}. Skipping download.")
        return

    print(f"Downloading MovieLens 1M from {ML_1M_URL} ...")
    download_file(ML_1M_URL, ZIP_PATH)

    print("Extracting...")
    extract_zip(ZIP_PATH, RAW_DIR)
    flatten_ml1m_folder(RAW_DIR)

    ZIP_PATH.unlink(missing_ok=True)

    print("Done. Files in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

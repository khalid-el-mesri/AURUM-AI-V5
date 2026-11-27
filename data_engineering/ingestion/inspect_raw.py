
from pathlib import Path
import hashlib

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)

EXPECTED_SHA256 = (
    "09C4C83FBDD3E92A0A1C21DA6D5FB80F"
    "DCF7579EC348066BCB94DE2DC876231A"
)


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def inspect_dataset() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    actual_hash = calculate_sha256(DATASET_PATH)

    if actual_hash != EXPECTED_SHA256:
        raise ValueError("RAW dataset SHA-256 mismatch. Inspection stopped.")

    print("\n=== AURUM-AI | RAW DATASET INSPECTION ===")
    print(f"File: {DATASET_PATH.name}")
    print(f"Size (bytes): {DATASET_PATH.stat().st_size}")
    print(f"SHA-256 verified: {actual_hash}")

    df = pd.read_csv(DATASET_PATH)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nFIRST 5 ROWS")
    print(df.head())

    print("\nLAST 5 ROWS")
    print(df.tail())

    print("\nMEMORY USAGE (MB)")
    print(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2))

    print("\nMISSING VALUES BY COLUMN")
    print(df.isna().sum())


if __name__ == "__main__":
    inspect_dataset()

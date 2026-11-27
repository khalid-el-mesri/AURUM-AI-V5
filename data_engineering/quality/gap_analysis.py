
from pathlib import Path

import pandas as pd


# Chemin du dataset original
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "xomususd-m5-bid-2017-01-26-2026-10-01.csv"
)


def analyze_gaps():
    # Lire uniquement la colonne temporelle
    df = pd.read_csv(DATASET_PATH, usecols=["timestamp"])

    # Convertir les millisecondes Unix en dates UTC
    timestamps = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Calculer le temps écoulé entre deux lignes
    intervals = timestamps.diff()

    gaps = pd.DataFrame({
        "start": timestamps.shift(1),
        "end": timestamps,
        "duration": intervals,
    })

    # Repérer les interruptions supérieures à 3 jours
    long_gaps = gaps[
        gaps["duration"] > pd.Timedelta(days=3)
    ].copy()

    # Associer chaque interruption à son année de début
    long_gaps["year"] = long_gaps["start"].dt.year

    print("\n=== AURUM-AI | GAP ANALYSIS ===")

    print(f"\nTotal long gaps (> 3 days): {len(long_gaps):,}")

    print("\nLONG GAPS BY YEAR")
    print(
        long_gaps.groupby("year")
        .size()
        .to_string()
    )

    print("\nTOP 20 LONGEST GAPS")
    print(
        long_gaps.nlargest(20, "duration")
        .to_string(index=False)
    )

    print("\nOBSERVATIONS BY YEAR")
    print(
        timestamps.dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    analyze_gaps()

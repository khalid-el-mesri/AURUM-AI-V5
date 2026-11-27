
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "02_clean"
    / "xom_regular_session.csv"
)


def audit_exchange_calendar():
    # 1. Load our filtered dataset
    df = pd.read_csv(DATASET_PATH)

    df["datetime_utc"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["date_ny"] = (
        df["datetime_utc"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    # 2. Load the NYSE trading calendar
    nyse = mcal.get_calendar("NYSE")

    start_date = df["date_ny"].min()
    end_date = df["date_ny"].max()

    schedule = nyse.schedule(
        start_date=start_date,
        end_date=end_date,
    )

    # 3. Group our real observations by trading date
    actual_by_date = {
        date: pd.DatetimeIndex(group["datetime_utc"])
        for date, group in df.groupby("date_ny")
    }

    # 4. Compare data against expected NYSE sessions
    results = []

    for date, session in schedule.iterrows():
        trading_date = date.date()

        # Expected 5-minute candle opening timestamps
        expected = pd.date_range(
            start=session["market_open"],
            end=session["market_close"],
            freq="5min",
            inclusive="left",
        )

        actual = actual_by_date.get(
            trading_date,
            pd.DatetimeIndex([], tz="UTC"),
        )

        missing = expected.difference(actual)
        extra = actual.difference(expected)

        results.append({
            "date": trading_date,
            "year": trading_date.year,
            "expected": len(expected),
            "observed": len(actual),
            "missing_bars": len(missing),
            "extra_bars": len(extra),
            "absent": len(actual) == 0,
        })

    report = pd.DataFrame(results)

    # 5. Detect dates present but not in NYSE calendar
    expected_dates = set(schedule.index.date)
    unexpected_dates = sorted(
        set(actual_by_date) - expected_dates
    )

    # 6. Display results
    print("\n=== AURUM-AI | NYSE CALENDAR AUDIT ===")

    print(f"\nDataset dates: {len(actual_by_date):,}")
    print(f"Expected NYSE sessions: {len(report):,}")
    print(f"Completely absent sessions: {report['absent'].sum():,}")

    print(
        "Missing candles on present sessions:",
        report.loc[
            ~report["absent"], "missing_bars"
        ].sum(),
    )

    print("Unexpected candle timestamps:",
          report["extra_bars"].sum())

    print("\nEXPECTED VS OBSERVED SESSIONS BY YEAR")

    yearly = report.groupby("year").agg(
        expected_sessions=("date", "size"),
        present_sessions=("absent", lambda x: (~x).sum()),
        absent_sessions=("absent", "sum"),
    )

    print(yearly.to_string())

    print("\nFIRST 20 ABSENT NYSE SESSIONS")
    print(
        report.loc[
            report["absent"],
            ["date", "expected"],
        ].head(20).to_string(index=False)
    )

    print("\nPRESENT SESSIONS WITH CANDLE MISMATCHES")

    mismatches = report[
        (report["observed"] > 0)
        & (
            (report["missing_bars"] > 0)
            | (report["extra_bars"] > 0)
        )
    ]

    print(
        mismatches[
            ["date", "expected", "observed",
             "missing_bars", "extra_bars"]
        ].head(30).to_string(index=False)
    )

    print("\nDATES OUTSIDE NYSE CALENDAR")
    print(unexpected_dates[:20])
    print(f"Total: {len(unexpected_dates):,}")

    print("\nAUDIT COMPLETED")
    print("No dataset modified.")


if __name__ == "__main__":
    audit_exchange_calendar()

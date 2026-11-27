"""Read-only, reproducible quality audit for AURUM-AI XOM OHLC M5 raw data.

This module NEVER writes to or transforms the source dataset on disk.
Time gaps are observations, not automatically defects: holidays, session closures,
market halts and vendor schedules need a dedicated calendar to classify.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ("timestamp", "open", "high", "low", "close")
EXPECTED_STEP_MS = 5 * 60 * 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_xom_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect dataframe and return summaries + bounded evidence (no mutations)."""
    missing_columns = set(EXPECTED_COLUMNS).difference(df.columns)
    if missing_columns:
        raise ValueError("Colonnes CSV absentes : " + ", ".join(sorted(missing_columns)))

    n = len(df)
    time_ms = pd.to_numeric(df["timestamp"], errors="coerce")
    is_integer_time = time_ms.notna() & np.isfinite(time_ms) & (time_ms % 1 == 0)
    parsed_time = pd.to_datetime(time_ms.where(is_integer_time), unit="ms", utc=True, errors="coerce")
    valid_time = parsed_time.notna()
    invalid_time = ~valid_time

    prices = df.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    finite_prices = pd.DataFrame(np.isfinite(prices), index=df.index, columns=prices.columns)
    invalid_prices = ~finite_prices.all(axis=1)
    nonpositive_prices = (prices.le(0) & finite_prices).any(axis=1)

    o, h, l, c = (prices[name] for name in ("open", "high", "low", "close"))
    # The exact OHLC inequalities are part of the market-data integrity contract.
    high_below_low = h < l
    high_below_open_close = h < pd.concat([o, c], axis=1).max(axis=1)
    low_above_open_close = l > pd.concat([o, c], axis=1).min(axis=1)
    ohlc_inconsistent = (high_below_low | high_below_open_close | low_above_open_close)
    ohlc_inconsistent = ohlc_inconsistent & ~invalid_prices

    repeated_full_rows = df.duplicated(keep="first")
    repeated_timestamps = valid_time & time_ms.duplicated(keep="first")
    reversed_time = valid_time & parsed_time.diff().lt(pd.Timedelta(0))
    off_grid_time = valid_time & time_ms.mod(EXPECTED_STEP_MS).ne(0)

    # Unique valid timestamps, sorted: do not count duplicates as gaps.
    ordered = parsed_time[valid_time].drop_duplicates().sort_values().reset_index(drop=True)
    deltas = ordered.diff().dt.total_seconds().div(60)
    gap_mask = deltas.gt(5)
    gap_end_positions = np.flatnonzero(gap_mask.to_numpy())
    gap_starts = ordered.iloc[gap_end_positions - 1].reset_index(drop=True)
    gap_ends = ordered.iloc[gap_end_positions].reset_index(drop=True)
    gap_durations = deltas.iloc[gap_end_positions].reset_index(drop=True)
    # For nonmultiples of 5, this number is a nominal count rather than exact slots.
    nominal_missing_slots = np.maximum(np.floor(gap_durations.to_numpy() / 5).astype("int64") - 1, 0)
    gap_category = np.where(
        gap_starts.dt.date.to_numpy() == gap_ends.dt.date.to_numpy(),
        "Même date UTC — à examiner", "Changement de date UTC — calendrier nécessaire",
    )
    gaps = pd.DataFrame({
        "Début (UTC)": gap_starts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Fin (UTC)": gap_ends.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Durée (minutes)": gap_durations.astype(float),
        "Créneaux M5 théoriques": nominal_missing_slots,
        "Catégorie indicative": gap_category,
    })
    if not gaps.empty:
        gaps = gaps.sort_values("Durée (minutes)", ascending=False, kind="stable").reset_index(drop=True)

    freq_irregular = deltas.notna() & (np.abs(deltas.mod(5).to_numpy()) > 1e-9)
    # A 1-, 2- or 3-minute interval is also irregular even though not a gap > 5min.
    non_m5_step_count = int(freq_irregular.sum())

    issue_masks = {
        "Horodatage invalide": invalid_time,
        "Prix absent/non numérique/non fini": invalid_prices,
        "Prix nul ou négatif": nonpositive_prices,
        "Incohérence OHLC": ohlc_inconsistent,
        "Ligne identique répétée": repeated_full_rows,
        "Timestamp répété": repeated_timestamps,
        "Chronologie inversée": reversed_time,
        "Timestamp hors grille 5min": off_grid_time,
    }
    sample_issues = []
    for issue_label, mask in issue_masks.items():
        hits = df.loc[mask, list(EXPECTED_COLUMNS)].head(15).copy()
        if not hits.empty:
            hits.insert(0, "Anomalie", issue_label)
            hits.insert(1, "Ligne CSV (en-tête exclu)", hits.index + 1)
            sample_issues.append(hits)
    evidence = pd.concat(sample_issues, ignore_index=True) if sample_issues else pd.DataFrame(
        columns=["Anomalie", "Ligne CSV (en-tête exclu)", *EXPECTED_COLUMNS]
    )

    completeness = pd.DataFrame({
        "Colonne": df.columns,
        "Type détecté": [str(x) for x in df.dtypes],
        "Valeurs nulles": [int(df[c].isna().sum()) for c in df.columns],
        "Null (%)": [round(float(df[c].isna().mean() * 100), 4) if n else 0.0 for c in df.columns],
    })
    yearly = (parsed_time[valid_time].dt.year.value_counts().sort_index()
              .rename_axis("Année").reset_index(name="Lignes"))

    time_first = parsed_time[valid_time].min()
    time_last = parsed_time[valid_time].max()
    summary: dict[str, Any] = {
        "n_rows": n,
        "n_columns": int(len(df.columns)),
        "memory_mib": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        "first_utc": time_first.isoformat() if pd.notna(time_first) else None,
        "last_utc": time_last.isoformat() if pd.notna(time_last) else None,
        "null_cells": int(df.isna().sum().sum()),
        "invalid_timestamps": int(invalid_time.sum()),
        "identical_duplicate_rows": int(repeated_full_rows.sum()),
        "repeated_timestamp_rows": int(repeated_timestamps.sum()),
        "reverse_chronology_rows": int(reversed_time.sum()),
        "off_grid_timestamps": int(off_grid_time.sum()),
        "non_m5_intervals": non_m5_step_count,
        "invalid_price_rows": int(invalid_prices.sum()),
        "nonpositive_price_rows": int(nonpositive_prices.sum()),
        "ohlc_inconsistent_rows": int(ohlc_inconsistent.sum()),
        "gap_count": int(len(gaps)),
        "same_utc_date_gaps": int((gap_category == "Même date UTC — à examiner").sum()),
        "cross_utc_date_gaps": int((gap_category != "Même date UTC — à examiner").sum()),
        "theoretical_gap_slots_all_periods": int(nominal_missing_slots.sum()),
        "schema_columns": list(df.columns),
    }
    return {
        "summary": summary,
        "completeness": completeness,
        "yearly": yearly,
        "gaps": gaps,
        "evidence": evidence,
        "parsed_time": parsed_time,
        "numeric_close": prices["close"],
    }

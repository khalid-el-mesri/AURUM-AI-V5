"""AURUM-AI V5 | Read-only, gap-aware analytics for XOMUSD bid M5 OHLC.

Does not modify the raw data; no synthetic candle, volume, forward fill, or trade.
Indicators on intraday bars reset when the source is no longer time-contiguous.
Daily indicators reset after gaps of more than 7 calendar days (conservative
heuristic; NOT an exchange calendar). No Test/Validation data are inferred.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
REQUIRED = ["timestamp", *OHLC]
AGGREGATION = {
    "5 min": None,
    "1 heure": "1h",
    "1 jour": "1D",
}
MAX_CONTIGUOUS_GAP = {
    "5 min": pd.Timedelta(minutes=5),
    "1 heure": pd.Timedelta(hours=1),
    "1 jour": pd.Timedelta(days=7),
}


def read_raw(path: str | Path) -> pd.DataFrame:
    """Return *validated* dataframe indexed by strictly increasing UTC times."""
    source = pd.read_csv(path, low_memory=False)
    if list(source.columns) != REQUIRED:
        raise ValueError(f"En-têtes CSV attendus : {REQUIRED}")
    timestamps = pd.to_numeric(source["timestamp"], errors="coerce")
    if timestamps.isna().any() or (timestamps % 1 != 0).any():
        raise ValueError("Horodatage manquant, invalide ou non entier")
    times = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Timestamps non valides, non ordonnés ou dupliqués")
    price = source[OHLC].apply(pd.to_numeric, errors="coerce")
    values = price.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prix absent, négatif, nul ou non fini")
    o, h, l, c = [price[x] for x in OHLC]
    if ((h < pd.concat([o, l, c], axis=1).max(axis=1)) |
            (l > pd.concat([o, h, c], axis=1).min(axis=1))).any():
        raise ValueError("Au moins une bougie viole les contraintes OHLC")
    result = price.copy()
    result.index = pd.DatetimeIndex(times, name="utc")
    return result


def subset_utc(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Both calendar dates included, interpreted in UTC."""
    start = pd.Timestamp(start_date).tz_localize("UTC")
    end = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    if start >= end:
        raise ValueError("La date de fin doit être >= à la date de début")
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def aggregate_ohlc(raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Observed-only bars; never fill or interpolate missing periods."""
    if timeframe not in AGGREGATION:
        raise ValueError("Granularité inconnue")
    rule = AGGREGATION[timeframe]
    if rule is None:
        return raw.copy()
    # OHLC aggregation from actual observations, not synthetic time slots.
    return raw.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=OHLC, how="any")


def contiguous_group_ids(index: pd.DatetimeIndex, timeframe: str) -> pd.Series:
    """Start a new group at any discontinuity larger than a stated threshold."""
    if timeframe not in MAX_CONTIGUOUS_GAP:
        raise ValueError("Granularité inconnue")
    boundary = pd.Series(index=index, data=index.to_series().diff().gt(MAX_CONTIGUOUS_GAP[timeframe]).to_numpy())
    return boundary.cumsum().astype("int64")


def wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Seed with SMA of the first period observations, then Wilder recursion."""
    vals = s.to_numpy(dtype=float)
    smoothed = np.full(len(vals), np.nan)
    if len(vals) >= period and np.isfinite(vals[:period]).all():
        smoothed[period - 1] = float(np.mean(vals[:period]))
        for i in range(period, len(vals)):
            if np.isfinite(vals[i]):
                smoothed[i] = (smoothed[i-1] * (period-1) + vals[i]) / period
    return pd.Series(smoothed, index=s.index, dtype=float)


def _indicators_one_segment(segment: pd.DataFrame) -> pd.DataFrame:
    close = segment["close"]
    out = segment.copy()
    out["return_pct"] = close.pct_change(fill_method=None) * 100
    out["log_return"] = np.log(close).diff()
    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["volatility_20_pct"] = out["log_return"].rolling(20, min_periods=20).std() * 100

    delta = close.diff()
    gain = delta.clip(lower=0).iloc[1:]
    loss = (-delta.clip(upper=0)).iloc[1:]
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    ratio = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + ratio))
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.gt(0), 0)
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.eq(0), 50)
    out["rsi_14"] = rsi.reindex(segment.index)

    previous_close = close.shift(1)
    tr = pd.concat([
        segment["high"] - segment["low"],
        (segment["high"] - previous_close).abs(),
        (segment["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = wilder_smooth(tr, 14)
    return out


def enrich_market(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Gap-aware, causal technical indicators based on observed candles only."""
    if bars.empty:
        return bars.assign(return_pct=pd.Series(dtype=float))
    groups = contiguous_group_ids(bars.index, timeframe)
    parts = []
    for group in groups.unique():
        sub = bars.loc[groups.eq(group)]
        result = _indicators_one_segment(sub)
        result["segment_id"] = int(group)
        parts.append(result)
    return pd.concat(parts).sort_index()


def gaps_table(frame: pd.DataFrame) -> pd.DataFrame:
    """List actual inter-observation gaps, not presumed missing market bars."""
    if len(frame) < 2:
        return pd.DataFrame(columns=["Début UTC", "Fin UTC", "Durée (heures)"])
    t = frame.index.to_series()
    diff = t.diff()
    flagged = diff > pd.Timedelta(days=7)
    return pd.DataFrame({
        "Début UTC": t.shift(1)[flagged].dt.strftime("%Y-%m-%d %H:%M"),
        "Fin UTC": t[flagged].dt.strftime("%Y-%m-%d %H:%M"),
        "Durée (heures)": (diff[flagged].dt.total_seconds() / 3600).round(2),
    }).reset_index(drop=True).sort_values("Durée (heures)", ascending=False)


def month_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly OBSERVED raw candle counts: not a market-session completeness %."""
    if frame.empty:
        return pd.DataFrame(columns=["Mois UTC", "Bougies observées"])
    counts = frame.groupby(frame.index.strftime("%Y-%m")).size()
    return counts.rename_axis("Mois UTC").reset_index(name="Bougies observées")

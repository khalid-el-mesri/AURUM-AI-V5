"""Exchange-calendar eligibility for AURUM-AI historical backtesting.

Only the signal timestamp, its trading session and the published exchange
calendar are used. In particular, no future market candles are consulted.

The timestamp is the start of a five-minute Bid candle. A scheduled exit
at or after the official NYSE market close is not eligible, because this
backtest requires an in-session candle *open* for execution.

The module does not change the raw XOM dataset or historical results.
"""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

BAR_MS = 5 * 60 * 1000


def session_cutoff_mask(
    signals: pd.DataFrame,
    *,
    exit_offset_bars: int = 7,
    bar_ms: int = BAR_MS,
    exchange: str = "NYSE",
) -> pd.Series:
    """Return a boolean Series: whether a signal fits the exchange session.

    The regular 30-minute strategy schedules its exit at signal + 7 bars;
    the 60-minute strategy schedules its exit at signal + 13 bars.
    Both trade at a candle's *open*. The exit must occur strictly before
    the published session close; opening an exit at 16:00 is not supported.

    Note: This is a *calendar-only* pretrade eligibility rule. Execution
    code must still handle missing data and unexpected disruptions when
    they occur; that must not be used as a future-looking pretrade filter.
    """
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame")
    if type(exit_offset_bars) is not int or exit_offset_bars <= 0:
        raise ValueError("exit_offset_bars must be a positive integer")
    if type(bar_ms) is not int or bar_ms <= 0:
        raise ValueError("bar_ms must be a positive integer")
    if not isinstance(exchange, str) or not exchange.strip():
        raise ValueError("exchange must be a non-empty string")

    required = {"signal_timestamp_ms", "session_date"}
    missing = sorted(required.difference(signals.columns))
    if missing:
        raise ValueError(f"Missing required signal columns: {missing}")
    if signals.empty:
        return pd.Series(False, index=signals.index, name="session_eligible", dtype=bool)

    fields = signals[["signal_timestamp_ms", "session_date"]]
    if fields.isna().any().any():
        raise ValueError("Null signal timestamps or session dates")

    # Reject fractional timestamps rather than silently rounding them.
    raw_ms = pd.to_numeric(signals["signal_timestamp_ms"], errors="raise")
    if (raw_ms != raw_ms.astype("int64")).any():
        raise ValueError("signal_timestamp_ms must contain whole milliseconds")
    times_utc = pd.to_datetime(raw_ms.astype("int64"), unit="ms", utc=True)
    session_dates = pd.to_datetime(
        signals["session_date"].astype(str), format="%Y-%m-%d", errors="raise"
    ).dt.normalize()

    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(
        start_date=session_dates.min().date(),
        end_date=session_dates.max().date(),
    )

    # The schedule contains known, session-specific early closings and DST.
    opens_utc = pd.to_datetime(session_dates.map(schedule["market_open"]), utc=True)
    closes_utc = pd.to_datetime(session_dates.map(schedule["market_close"]), utc=True)
    if opens_utc.isna().any() or closes_utc.isna().any():
        invalid_days = session_dates[opens_utc.isna() | closes_utc.isna()]
        dates = sorted(invalid_days.dt.strftime("%Y-%m-%d").unique().tolist())
        raise ValueError(f"No {exchange} trading session for dates: {dates}")

    exit_times_utc = times_utc + pd.to_timedelta(exit_offset_bars * bar_ms, unit="ms")
    eligible = (
        (times_utc >= opens_utc)
        & (times_utc < closes_utc)
        & (exit_times_utc < closes_utc)
    )
    return pd.Series(eligible.to_numpy(dtype=bool), index=signals.index, name="session_eligible")

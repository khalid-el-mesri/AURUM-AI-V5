"""Pure engine tests; no Streamlit runtime or historical dataset needed."""
import numpy as np
import pandas as pd
import pytest
from frontend.services.market_analysis_engine import (
    read_raw, subset_utc, aggregate_ohlc, enrich_market, gaps_table, month_coverage,
)


def make_prices(start="2023-10-10", count=70, freq="5min", offset=0):
    dt = pd.date_range(start, periods=count, freq=freq, tz="UTC")
    p = np.arange(count, dtype=float) + 100 + offset
    return pd.DataFrame({"open": p, "high": p+1, "low":p-1, "close":p+.25}, index=dt)


def test_read_raw_and_no_source_change(tmp_path):
    frame = make_prices(count=4)
    csv = tmp_path / "raw.csv"
    origin = pd.DataFrame({"timestamp": frame.index.asi8//1_000_000, **{c:frame[c].to_numpy() for c in frame.columns}})
    origin.to_csv(csv, index=False)
    previous = csv.read_bytes()
    result = read_raw(csv)
    assert result.index.tz is not None
    assert len(result) == 4
    assert previous == csv.read_bytes()


def test_daily_aggregation_observed_only():
    base = pd.concat([make_prices(count=12), make_prices("2023-10-12", count=12, offset=40)])
    agg = aggregate_ohlc(base, "1 jour")
    assert len(agg) == 2
    assert agg.iloc[0]["open"] == base.iloc[0]["open"]
    assert agg.iloc[0]["close"] == base.iloc[11]["close"]
    assert agg.iloc[0]["high"] == base.iloc[:12]["high"].max()
    assert agg.iloc[0]["low"] == base.iloc[:12]["low"].min()


def test_no_returns_or_indicators_across_intraday_gap():
    a = make_prices(count=70)
    b = make_prices(start="2023-10-26", count=70, offset=900)
    enriched = enrich_market(pd.concat([a,b]), "5 min")
    assert np.isnan(enriched.loc[b.index[0], "return_pct"])
    assert np.isnan(enriched.loc[b.index[0], "sma_20"])
    assert np.isnan(enriched.loc[b.index[0], "rsi_14"])
    assert enriched.loc[b.index[20], "sma_20"] == pytest.approx(b["close"].iloc[1:21].mean())
    assert enriched.loc[b.index[0], "segment_id"] != enriched.loc[a.index[0], "segment_id"]


def test_daily_long_gap_breaks_indicator_chain():
    a = make_prices(count=40, freq="1D")
    b = make_prices(start="2023-12-15",count=20, freq="1D", offset=50)
    enriched = enrich_market(pd.concat([a,b]), "1 jour")
    assert np.isnan(enriched.loc[b.index[0],"return_pct"])
    assert np.isnan(enriched.loc[b.index[0],"ema_20"])


def test_future_data_cannot_change_past_indicators():
    a = make_prices(count=60)
    base = enrich_market(a, "5 min")
    updated = a.copy()
    updated.loc[updated.index[-1], "close"] *= 1.1
    future = enrich_market(updated, "5 min")
    pd.testing.assert_frame_equal(base.iloc[:-1], future.iloc[:-1])


def test_gap_coverage_dates():
    frame = pd.concat([make_prices(count=5), make_prices(start="2023-11-01", count=5)])
    assert len(gaps_table(frame)) == 1
    assert len(month_coverage(frame)) == 2
    assert len(subset_utc(frame, "2023-10-10", "2023-10-10")) == 5


def test_reject_bad_ohlc(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp":[1696896000000],"open":[101],"high":[99],"low":[98],"close":[100]}).to_csv(csv,index=False)
    with pytest.raises(ValueError,match="OHLC"):
        read_raw(csv)

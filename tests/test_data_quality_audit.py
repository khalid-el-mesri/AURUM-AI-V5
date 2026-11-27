"""Unit tests: python -m pytest tests/test_data_quality_audit.py -q"""
import pandas as pd
from frontend.services.data_quality_audit import audit_xom_frame

BASE = 1_485_388_800_000  # timestamp UTC aligned to the 5-minute grid


def frame(ts, rows=None):
    rows = rows or [(100.0, 101.0, 99.0, 100.0)] * len(ts)
    return pd.DataFrame([
        {"timestamp": t, "open": o, "high": h, "low": l, "close": c}
        for t, (o, h, l, c) in zip(ts, rows)
    ])


def test_valid_equal_ohlc_candle_is_allowed():
    df = frame([BASE, BASE + 300000], [(85.72, 85.72, 85.72, 85.72)] * 2)
    r = audit_xom_frame(df)["summary"]
    assert r["ohlc_inconsistent_rows"] == 0
    assert r["gap_count"] == 0
    assert r["identical_duplicate_rows"] == 0


def test_gap_vs_duplicate_and_reverse_order():
    df = frame([BASE + 600000, BASE, BASE, BASE + 300000])
    r = audit_xom_frame(df)["summary"]
    assert r["repeated_timestamp_rows"] == 1
    assert r["identical_duplicate_rows"] == 1
    assert r["reverse_chronology_rows"] == 1
    assert r["gap_count"] == 0  # 0, 5, 10 min after sorting unique timestamps


def test_ohlc_anomaly_and_invalid_price():
    df = frame([BASE, BASE + 300000, BASE + 600000], [
        (100.0, 99.0, 101.0, 100.0),
        (100.0, 101.0, 99.0, 100.0),
        (100.0, float("nan"), 99.0, 100.0),
    ])
    r = audit_xom_frame(df)["summary"]
    assert r["ohlc_inconsistent_rows"] == 1
    assert r["invalid_price_rows"] == 1


def test_interval_is_observation_not_declared_error():
    df = frame([BASE, BASE + 900000])
    a = audit_xom_frame(df)
    assert a["summary"]["gap_count"] == 1
    assert a["summary"]["theoretical_gap_slots_all_periods"] == 2
    assert a["gaps"].iloc[0]["Durée (minutes)"] == 15


def test_missing_columns_raise():
    try:
        audit_xom_frame(pd.DataFrame({"timestamp": [BASE]}))
    except ValueError as exc:
        assert "Colonnes CSV absentes" in str(exc)
    else:
        raise AssertionError("An incomplete OHLC schema must be rejected")

"""Synthetic deterministic tests of read-only AURUM-AI autonomous-agents audit."""
import json
import numpy as np
import pytest
from frontend.services.autonomous_agents_audit import (
    available_experiments, load_validation_npz, signal_decisions, signal_summary,
    timing_audit, price_target_consistency, parse_historical_ledger,
)


def example():
    end = np.array([1_000_000, 1_300_000, 1_600_000], dtype=np.int64)
    close = np.array([100., 100., 100.])
    nxt = 100 * np.exp(np.array([0.002, -0.002, 0.0]))
    return dict(y_pred=np.array([0.0004, -0.0004, 0.0], dtype=np.float32),
                y_true=np.log(nxt / close).astype(np.float32),
                end_timestamp_ms=end, target_timestamp_ms=end+300_000,
                signal_available_at_ms=end+300_000, target_available_at_ms=end+600_000,
                current_close=close, next_close=nxt)


def test_discovery(tmp_path):
    p=tmp_path/"cnn1d_60m_seed42"; p.mkdir(); (p/"validation_predictions.npz").touch()
    o=tmp_path/"other_stuff"; o.mkdir(); (o/"validation_predictions.npz").touch()
    assert set(available_experiments(tmp_path)) == {("CNN1D",60)}


def test_load_and_reject_missing(tmp_path):
    p=tmp_path/"preds.npz"; np.savez_compressed(p,**example())
    assert len(load_validation_npz(p)["y_true"])==3
    np.savez_compressed(p,y_pred=np.array([1]))
    with pytest.raises(ValueError,match="Missing fields"):
        load_validation_npz(p)


def test_invalid_values(tmp_path):
    p=tmp_path/"bad.npz"; a=example(); a["y_pred"][0]=np.nan
    np.savez(p,**a)
    with pytest.raises(ValueError,match="Non-finite"):
        load_validation_npz(p)


def test_actions_use_real_strategy_agent():
    df=signal_decisions(example(),0.0003)
    assert df["action_proposed"].tolist()==["BUY","SELL","HOLD"]
    assert signal_summary(df)["BUY"] == 1
    assert signal_summary(df)["SELL"] == 1
    assert signal_summary(df)["HOLD"] == 1


def test_threshold_changes_decisions():
    df=signal_decisions(example(),0.002)
    assert (df["action_proposed"]=="HOLD").all()


def test_consistency_and_audit():
    a=example()
    assert price_target_consistency(a)==0
    audit=timing_audit(a,5)
    assert audit["wrong_target_horizon"]==0
    assert audit["target_at_or_before_signal"]==0


def test_bad_horizon():
    assert timing_audit(example(),60)["wrong_target_horizon"]==3


def test_bad_price():
    a=example(); a["next_close"][0]=200.
    assert price_target_consistency(a)==1


def test_ledger_list():
    b=[{"index":0,"event_type":"GENESIS"},{"index":1,"event_type":"FORECAST_DECISION","payload":{"strategy":{"action":"BUY"},"risk":{"status":"APPROVED"}}}]
    df=parse_historical_ledger(b)
    assert df.loc[1,"strategy_action"]=="BUY"
    assert df.loc[1,"risk_status"]=="APPROVED"


def test_ledger_dict():
    df=parse_historical_ledger({"chain":[{"index":0,"payload":{}}]})
    assert len(df)==1


def test_bad_ledger():
    with pytest.raises(ValueError):parse_historical_ledger({"not_blocks": []})

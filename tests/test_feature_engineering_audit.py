"""Unit tests on synthetic data: never require the user's private model assets."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from frontend.services.feature_engineering_audit import (
    FIVE_MIN_MS, audit_feature_frame, datetime_axis, feature_missing_table,
    find_feature_sets, find_window_sets, inspect_window_file,
    inspect_window_preview, split_boundary_check,
)


def make_frame():
    t0 = 1700000100000
    return pd.DataFrame({
        'timestamp': [t0, t0+FIVE_MIN_MS, t0+2*FIVE_MIN_MS,
                      t0+5*FIVE_MIN_MS, t0+6*FIVE_MIN_MS],
        'segment_id': [1,1,1,2,2],
        'close': [100.,101.,102.,130.,131.],
        'return_1': [np.nan,.01,.0099,np.nan,.0077],
        'feature_available_at_ms': [t0+FIVE_MIN_MS,t0+2*FIVE_MIN_MS,
                                    t0+3*FIVE_MIN_MS,t0+6*FIVE_MIN_MS,
                                    t0+7*FIVE_MIN_MS],
    })


def make_npz(path: Path, *, override=None, n=6):
    end = (np.arange(n)*FIVE_MIN_MS + 1700000100000).astype(np.int64)
    arrays = {
        'X': np.ones((n,20,2),np.float32), 'y': np.zeros(n,np.float32),
        'feature_names': np.asarray(['a','b']),
        'end_timestamp_ms': end, 'target_timestamp_ms': end+FIVE_MIN_MS*12,
        'signal_available_at_ms': end+FIVE_MIN_MS,
        'target_available_at_ms': end+FIVE_MIN_MS*13,
    }
    if override:
        arrays.update(override)
    np.savez(path, **arrays)
    return path


def test_feature_audit_regular_and_segment_reset():
    summary = audit_feature_frame(make_frame())
    assert summary['rows'] == 5
    assert summary['n_segments'] == 2
    assert summary['gaps_within_segment'] == 0
    assert summary['available_before_timestamp'] == 0
    assert summary['reverse_chronology'] == 0


def test_feature_audit_flags_intrasegment_gaps():
    d = make_frame()
    d.loc[3, 'segment_id'] = 1
    summary = audit_feature_frame(d)
    assert summary['gaps_within_segment'] == 1


def test_feature_audit_flags_invalid_availability():
    d = make_frame()
    d.loc[1, 'feature_available_at_ms'] = d.loc[1,'timestamp'] - 5000
    report = audit_feature_frame(d)
    assert report['available_before_timestamp'] == 1


def test_audit_missing_warmup_and_nonmutation():
    d = make_frame()
    copy = d.copy(deep=True)
    report = audit_feature_frame(d)
    assert report['null_counts']['return_1'] == 2
    assert feature_missing_table(d).loc[3,'Null'] == 2
    pd.testing.assert_frame_equal(d, copy)


def test_flags_duplicates_and_inversions():
    d = make_frame()
    d.loc[2,'timestamp'] = d.loc[0,'timestamp']
    r = audit_feature_frame(d)
    assert r['duplicate_timestamps'] == 1
    assert r['reverse_chronology'] == 1


def test_datetime_axis_correct_utc():
    result = datetime_axis(make_frame())
    assert len(result) == 5 and str(result.dt.tz) == 'UTC'


def test_npz_archive_readonly_good(tmp_path):
    p = make_npz(tmp_path/'train.npz')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    data = inspect_window_file(p)
    assert data['samples'] == 6
    assert data['fields']['X']['shape'] == [6,20,2]
    assert data['feature_names'] == ['a','b']
    assert not data['errors']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    assert inspect_window_preview(p,0).shape == (20,2)


def test_npz_target_precedes_signal_flags_error(tmp_path):
    p = tmp_path / 'bad.npz'
    end = 1700000100000 + np.arange(4)*FIVE_MIN_MS
    make_npz(p, override={'target_available_at_ms':end.astype(np.int64)},n=4)
    result = inspect_window_file(p)
    assert result['errors']
    assert result['timing']['signal_available_at_ms_ge_target_available_at_ms'] == 4


def test_npz_mismatched_names_and_targets(tmp_path):
    p = make_npz(tmp_path/'bad.npz', override={
        'feature_names':np.array(['only_one']), 'y':np.zeros(5,dtype=np.float32),
    })
    assert len(inspect_window_file(p)['errors']) >= 2


def test_npz_without_availability_is_not_certified(tmp_path):
    p = tmp_path / 'old.npz'
    np.savez(p, X=np.ones((2,20,1)),y=np.zeros(2),
             feature_names=np.array(['a']),
             end_timestamp_ms=np.array([1000,2000]),
             target_timestamp_ms=np.array([3000,4000]))
    result = inspect_window_file(p)
    assert not result['errors']
    assert any('absent' in text for text in result['warnings'])


def test_boundary_comparison_train_before_val(tmp_path):
    train = make_npz(tmp_path/'train.npz',n=3)
    end = np.arange(3)*FIVE_MIN_MS + 1700000100000 + 10_000_000
    val = make_npz(tmp_path/'val.npz',n=3, override={
        'end_timestamp_ms':end, 'target_timestamp_ms':end+12*FIVE_MIN_MS,
        'signal_available_at_ms':end+FIVE_MIN_MS,
        'target_available_at_ms':end+13*FIVE_MIN_MS,
    })
    status = split_boundary_check(train, val)
    assert status['status'] == 'ordre conforme'
    assert status['strict_availability_check'] is True


def test_boundary_overlap_detected(tmp_path):
    train = make_npz(tmp_path/'train.npz')
    val = make_npz(tmp_path/'val.npz')
    status = split_boundary_check(train,val)
    assert status['status'] == 'chevauchement à examiner'


def test_file_inventory(tmp_path):
    features = tmp_path/'data'/'03_features'
    windows = tmp_path/'data'/'04_windows'/'nyse_v3'
    features.mkdir(parents=True)
    windows.mkdir(parents=True)
    (features/'a.parquet').write_bytes(b'')
    (windows/'b.npz').write_bytes(b'')
    assert len(find_feature_sets(tmp_path)) == 1
    assert len(find_window_sets(tmp_path)) == 1

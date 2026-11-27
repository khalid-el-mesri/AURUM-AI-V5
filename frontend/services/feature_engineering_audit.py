"""AURUM-AI V5 feature/window explorer: READ ONLY, no training or mutation.

Temporal checks indicate *potential* look-ahead; they are not proof of
leakage-free preprocessing/scaling. Source parquet and NPZ remain untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

FIVE_MIN_MS = 5 * 60 * 1000


def find_feature_sets(root: Path) -> list[Path]:
    folder = Path(root) / 'data' / '03_features'
    return sorted(p for p in folder.glob('*.parquet') if p.is_file())


def find_window_sets(root: Path) -> list[Path]:
    folder = Path(root) / 'data' / '04_windows'
    return sorted(p for p in folder.rglob('*.npz') if p.is_file())


def read_features(path: Path) -> pd.DataFrame:
    """Read an existing parquet only; pyarrow or fastparquet required locally."""
    path = Path(path)
    if path.suffix.lower() != '.parquet' or not path.is_file():
        raise ValueError('Fichier parquet inexistant ou extension invalide')
    return pd.read_parquet(path)


def datetime_axis(frame: pd.DataFrame) -> pd.Series:
    if 'datetime_utc' in frame:
        return pd.to_datetime(frame['datetime_utc'], errors='coerce', utc=True)
    if 'timestamp' in frame:
        return pd.to_datetime(pd.to_numeric(frame['timestamp'], errors='coerce'), unit='ms', errors='coerce', utc=True)
    return pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns, UTC]')


def audit_feature_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit only properties observed in the supplied data, without mutation."""
    report: dict[str, Any] = {
        'rows': int(len(frame)), 'columns': int(frame.shape[1]),
        'column_names': list(map(str, frame.columns)),
        'null_cells': int(frame.isna().sum().sum()),
        'null_counts': {str(k): int(v) for k, v in frame.isna().sum().items()},
        'n_segments': None, 'duplicate_timestamps': None,
        'reverse_chronology': None, 'invalid_timestamps': None,
        'gaps_within_segment': None, 'available_before_timestamp': None,
        'available_at_timestamp': None, 'invalid_feature_availability': None,
        'datetime_mismatch': None, 'first_utc': None, 'last_utc': None,
        'nonfinite_numeric_cells': 0,
    }
    numeric = frame.select_dtypes(include=[np.number])
    # NaNs in initial SMA / returns are warm-up values; count separately.
    if not numeric.empty:
        report['nonfinite_numeric_cells'] = int(np.isinf(numeric.to_numpy(dtype='float64', copy=False)).sum())
    if 'segment_id' in frame:
        report['n_segments'] = int(frame['segment_id'].nunique(dropna=True))
    if 'timestamp' in frame:
        ts = pd.to_numeric(frame['timestamp'], errors='coerce')
        invalid = ts.isna() | pd.to_datetime(ts, unit='ms', utc=True, errors='coerce').isna()
        report['invalid_timestamps'] = int(invalid.sum())
        report['duplicate_timestamps'] = int(ts[~invalid].duplicated().sum())
        diff = ts.diff()
        report['reverse_chronology'] = int((diff < 0).sum())
        if 'segment_id' in frame:
            same_segment = frame['segment_id'].eq(frame['segment_id'].shift())
            report['gaps_within_segment'] = int(((diff > FIVE_MIN_MS) & same_segment).sum())
        dt = pd.to_datetime(ts, unit='ms', errors='coerce', utc=True)
        valid_dt = dt.dropna()
        if not valid_dt.empty:
            report['first_utc'] = valid_dt.min().isoformat()
            report['last_utc'] = valid_dt.max().isoformat()
        if 'datetime_utc' in frame:
            stated = pd.to_datetime(frame['datetime_utc'], errors='coerce', utc=True)
            mismatch = (dt.notna() & stated.notna() & dt.ne(stated))
            report['datetime_mismatch'] = int(mismatch.sum())
        if 'feature_available_at_ms' in frame:
            available = pd.to_numeric(frame['feature_available_at_ms'], errors='coerce')
            report['invalid_feature_availability'] = int(available.isna().sum())
            valid = (~invalid) & available.notna()
            report['available_before_timestamp'] = int((valid & (available < ts)).sum())
            report['available_at_timestamp'] = int((valid & (available == ts)).sum())
    return report


def feature_missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    count = frame.isna().sum()
    return pd.DataFrame({
        'Colonne': frame.columns.astype(str),
        'Type': [str(t) for t in frame.dtypes],
        'Null': count.to_numpy(dtype='int64'),
        'Null (%)': (100 * count / max(1, len(frame))).round(3).to_numpy(),
    })


def _field_details(arr: np.ndarray) -> dict[str, Any]:
    return {'shape': list(map(int, arr.shape)), 'dtype': str(arr.dtype)}


def inspect_window_file(path: Path) -> dict[str, Any]:
    """Inspect NPZ with pickle prohibited. Does not write or transform windows."""
    path = Path(path)
    if path.suffix.lower() != '.npz' or not path.is_file():
        raise ValueError('Fichier NPZ absent ou extension invalide')
    result: dict[str, Any] = {
        'filename': path.name, 'fields': {}, 'samples': None,
        'feature_names': [], 'warnings': [], 'errors': [], 'timing': {},
        'first_end_utc': None, 'last_end_utc': None,
    }
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    result['fields'] = {k: _field_details(v) for k, v in arrays.items()}
    X = arrays.get('X')
    y = arrays.get('y')
    if X is None or X.ndim != 3:
        result['errors'].append('X doit avoir la forme (samples, sequence_length, features)')
    else:
        result['samples'] = int(X.shape[0])
        if np.issubdtype(X.dtype, np.number):
            result['timing']['nonfinite_X_cells'] = int((~np.isfinite(X)).sum())
        if y is not None and (not y.shape or y.shape[0] != X.shape[0]):
            result['errors'].append('X et y ne possèdent pas le même nombre d’échantillons')
        if y is not None and np.issubdtype(y.dtype, np.number):
            result['timing']['nonfinite_y_cells'] = int((~np.isfinite(y)).sum())
    if 'feature_names' in arrays:
        names = arrays['feature_names']
        result['feature_names'] = [str(v) for v in names.tolist()]
        if X is not None and X.ndim == 3 and len(result['feature_names']) != X.shape[2]:
            result['errors'].append('Nombre de feature_names différent de la dernière dimension de X')
    else:
        result['warnings'].append('feature_names absent : variables non identifiables dans ce NPZ')

    n = X.shape[0] if X is not None and X.ndim == 3 else None
    for name in ('end_timestamp_ms', 'target_timestamp_ms', 'signal_available_at_ms', 'target_available_at_ms'):
        arr = arrays.get(name)
        if n is not None and arr is not None and (arr.ndim != 1 or len(arr) != n):
            result['errors'].append(f'{name} doit avoir {n} valeurs')
    def pair_counts(left: str, right: str, *, strict: bool = False):
        a, b = arrays.get(left), arrays.get(right)
        if a is None or b is None or a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            result['warnings'].append(f'Contrôle {left} / {right} indisponible')
            return
        if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
            result['warnings'].append(f'Horodatages non numériques : {left} / {right}')
            return
        bad = (a >= b) if strict else (a > b)
        result['timing'][f'{left}_{"ge" if strict else "gt"}_{right}'] = int(np.count_nonzero(bad))
        if np.any(bad):
            result['errors'].append(f'Ordre temporel suspect : {left} / {right} ({int(np.count_nonzero(bad))} lignes)')

    pair_counts('end_timestamp_ms', 'target_timestamp_ms', strict=True)
    pair_counts('end_timestamp_ms', 'signal_available_at_ms')
    pair_counts('signal_available_at_ms', 'target_available_at_ms', strict=True)
    pair_counts('target_timestamp_ms', 'target_available_at_ms')
    if 'signal_available_at_ms' not in arrays:
        result['warnings'].append('signal_available_at_ms absent : chronologie de disponibilité non vérifiable complètement')
    if 'target_available_at_ms' not in arrays:
        result['warnings'].append('target_available_at_ms absent : moment de disponibilité de la cible inconnu')
    ends = arrays.get('end_timestamp_ms')
    if ends is not None and ends.ndim == 1 and ends.size and np.issubdtype(ends.dtype, np.number):
        stamps = pd.to_datetime([int(ends.min()), int(ends.max())], unit='ms', utc=True, errors='coerce')
        result['first_end_utc'] = None if pd.isna(stamps[0]) else stamps[0].isoformat()
        result['last_end_utc'] = None if pd.isna(stamps[1]) else stamps[1].isoformat()
        if int(np.count_nonzero(np.diff(ends) < 0)):
            result['warnings'].append('end_timestamp_ms n’est pas chronologiquement trié')
    return result


def inspect_window_preview(path: Path, sample: int = 0) -> pd.DataFrame:
    """Small first/selected X sample for display; never exports a whole dataset."""
    with np.load(path, allow_pickle=False) as data:
        X = data['X']
        if X.ndim != 3 or X.shape[0] == 0:
            return pd.DataFrame()
        idx = max(0, min(int(sample), X.shape[0] - 1))
        names = ([str(x) for x in data['feature_names'].tolist()]
                 if 'feature_names' in data.files else [])
        if len(names) != X.shape[2]:
            names = [f'feature_{i + 1}' for i in range(X.shape[2])]
        return pd.DataFrame(X[idx], columns=names)


def split_boundary_check(train_path: Path, validation_path: Path) -> dict[str, Any]:
    """Check only availability boundaries; no claim about preprocessing fit."""
    def keys(path: Path):
        with np.load(path, allow_pickle=False) as data:
            fields = set(data.files)
            return {
                k: data[k]
                for k in ('target_available_at_ms', 'signal_available_at_ms',
                          'target_timestamp_ms', 'end_timestamp_ms') if k in fields
            }
    a, b = keys(Path(train_path)), keys(Path(validation_path))
    train_key = 'target_available_at_ms' if 'target_available_at_ms' in a else 'target_timestamp_ms'
    val_key = 'signal_available_at_ms' if 'signal_available_at_ms' in b else 'end_timestamp_ms'
    if train_key not in a or val_key not in b:
        return {'status': 'inconnu', 'reason': 'Métadonnées temporelles insuffisantes'}
    if a[train_key].size == 0 or b[val_key].size == 0:
        return {'status': 'inconnu', 'reason': 'Un des ensembles est vide'}
    train_last, val_first = int(a[train_key].max()), int(b[val_key].min())
    result = {
        'status': 'ordre conforme' if train_last < val_first else 'chevauchement à examiner',
        'train_last_ms': train_last, 'validation_first_ms': val_first,
        'train_field': train_key, 'validation_field': val_key,
        'strict_availability_check': train_key == 'target_available_at_ms' and val_key == 'signal_available_at_ms',
    }
    return result

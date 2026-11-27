"""AURUM-AI Research Center / Experiments: immutable local Validation registry.

Scopes are fixed to the existing 12 NYSE V3 model/horizon combinations.
No checkpoints are loaded, no arbitrary folders are traversed, and no writes occur.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

MODELS = ("LSTM", "GRU", "CNN1D")
HORIZONS = (5, 15, 30, 60)
EXPERIMENTS_DIR = Path("artifacts/experiments/nyse_v3")
PREDICTIONS = "validation_predictions.npz"
REPORT = "report.json"
REPORT_LIMIT = 2 * 1024 * 1024
CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})
SENSITIVE_FRAGMENTS = ("password", "secret", "private_key", "api_key", "access_token", "authorization", "credential", "smtp_password", "client_secret")


@dataclass(frozen=True)
class Experiment:
    model: str
    horizon_minutes: int
    run_id: str
    status: str
    validation_present: bool
    report_present: bool
    validation_size_bytes: int | None
    report_size_bytes: int | None
    checkpoint_count: int
    last_modified_utc: str | None


def _safe_file(path: Path, root: Path) -> bool:
    """Reject file and ancestor symlinks; allow only files within project root."""
    try:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            return False
        rel = path.relative_to(root)
        if rel.is_absolute() or ".." in rel.parts:
            return False
        curr = root
        for part in rel.parts:
            curr = curr / part
            if curr.is_symlink():
                return False
        return path.is_file() and path.resolve(strict=True).is_relative_to(resolved_root)
    except (ValueError, OSError, RuntimeError):
        return False


def _run_directory(root: Path, model: str, horizon: int) -> Path:
    if model not in MODELS or type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("Modèle/horizon non autorisé")
    return root / EXPERIMENTS_DIR / f"{model.lower()}_{horizon}m_seed42"


def artifact_path(root: Path | str, model: str, horizon: int, kind: str) -> Path:
    """Return validated existing report or prediction artifact, never a user path."""
    project = Path(root).resolve(strict=True)
    if kind not in ("report", "validation"):
        raise ValueError("Type d'artefact non autorisé")
    name = REPORT if kind == "report" else PREDICTIONS
    path = _run_directory(project, model, horizon) / name
    if not _safe_file(path, project):
        raise FileNotFoundError(f"Artefact absent ou lien symbolique refusé : {model} {horizon}m {kind}")
    return path


def _stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def list_experiments(repo_root: Path | str) -> list[Experiment]:
    """12 expected combinations, including missing runs, deterministic order."""
    root = Path(repo_root).resolve()
    results: list[Experiment] = []
    for horizon in HORIZONS:
        for model in MODELS:
            folder = _run_directory(root, model, horizon)
            # Never descend from a symlink (nor from a junction via resolve check).
            folder_ok = folder.is_dir() and not folder.is_symlink()
            if folder_ok:
                try:
                    folder_ok = folder.resolve(strict=True).is_relative_to(root.resolve(strict=True))
                except (OSError, RuntimeError, ValueError):
                    folder_ok = False
            validation = folder / PREDICTIONS
            report = folder / REPORT
            vp = folder_ok and _safe_file(validation, root)
            rp = folder_ok and _safe_file(report, root)
            present = [p for p, ok in ((validation, vp), (report, rp)) if ok]
            if vp and rp:
                status = "Complet"
            elif vp:
                status = "Rapport absent"
            elif rp:
                status = "Prédictions absentes"
            else:
                status = "Absent"
            # Count checkpoint NAMES only, and never load binary/pickle content.
            checkpoints = 0
            if folder_ok:
                try:
                    checkpoints = sum(1 for p in folder.iterdir()
                                      if p.suffix.lower() in CHECKPOINT_SUFFIXES
                                      and _safe_file(p, root))
                except OSError:
                    checkpoints = 0
            results.append(Experiment(
                model=model, horizon_minutes=horizon, run_id=folder.name,
                status=status, validation_present=bool(vp), report_present=bool(rp),
                validation_size_bytes=validation.stat().st_size if vp else None,
                report_size_bytes=report.stat().st_size if rp else None,
                checkpoint_count=checkpoints,
                last_modified_utc=max((_stamp(p) for p in present), default=None),
            ))
    return results


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str):
    raise ValueError("NaN / Infinity interdit dans les métadonnées JSON")


def redact(value: Any) -> Any:
    """Avoid accidental disclosure of credential-looking metadata fields."""
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if any(s in str(k).lower() for s in SENSITIVE_FRAGMENTS)
                          else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value[:200]]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_report(repo_root: Path | str, model: str, horizon: int) -> dict[str, Any]:
    path = artifact_path(repo_root, model, horizon, "report")
    if path.stat().st_size > REPORT_LIMIT:
        raise ValueError("Rapport trop volumineux pour un aperçu sécurisé")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("Le rapport doit être un objet JSON")
    return redact(data)


def scalar_metadata(data: dict[str, Any], max_depth: int = 4, limit: int = 300) -> dict[str, str]:
    """Flatten only scalar metadata. Do not infer meaning from unknown report schema."""
    output: dict[str, str] = {}
    def visit(obj: Any, prefix: str, depth: int):
        if len(output) >= limit or depth > max_depth:
            return
        if isinstance(obj, dict):
            for k in sorted(obj, key=str):
                if len(output) >= limit:
                    break
                visit(obj[k], f"{prefix}.{k}" if prefix else str(k), depth + 1)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            value = str(obj)
            output[prefix] = value[:160] + ("…" if len(value) > 160 else "")
    visit(redact(data), "", 0)
    return output


def compare_report_metadata(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, str]]:
    lft, rgt = scalar_metadata(left), scalar_metadata(right)
    return [
        {"Champ JSON": key, "A": lft.get(key, "(absent)"),
         "B": rgt.get(key, "(absent)"),
         "Comparaison": "identique" if key in lft and key in rgt and lft[key] == rgt[key] else "différent"}
        for key in sorted(lft.keys() | rgt.keys())
    ]


def manifest(repo_root: Path | str, runs: list[Experiment], with_hashes: bool = False) -> dict[str, Any]:
    """In-memory inventory only. Optional hashes require an explicit UI click."""
    root = Path(repo_root)
    records = []
    for run in runs:
        entry = asdict(run)
        entry["source_scope"] = "historical_validation_only"
        if with_hashes:
            digests: dict[str, str] = {}
            for kind, exists in (("report", run.report_present), ("validation", run.validation_present)):
                if exists:
                    digests[kind] = sha256(artifact_path(root, run.model, run.horizon_minutes, kind))
            entry["sha256"] = digests
        records.append(entry)
    return {
        "project": "AURUM-AI V5", "scope": "historical_validation_only",
        "source_folder": EXPERIMENTS_DIR.as_posix(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "no_test_results": True, "no_training_or_execution": True,
        "records": records,
        "limitations": [
            "Missing artifacts are not reconstructed.",
            "Report schemas are displayed, not certified or reinterpreted.",
            "File checks do not prove an absence of data leakage or predictive performance on Test.",
            "Local SHA-256 is not external certification."
        ],
    }


def manifest_json_bytes(repo_root: Path | str, runs: list[Experiment], with_hashes: bool = False) -> bytes:
    return json.dumps(manifest(repo_root, runs, with_hashes), ensure_ascii=False, indent=2,
                      allow_nan=False).encode("utf-8")


def registry_csv_bytes(runs: list[Experiment]) -> bytes:
    out = io.StringIO()
    cols = list(Experiment.__dataclass_fields__)
    writer = csv.DictWriter(out, fieldnames=cols)
    writer.writeheader()
    for row in runs:
        writer.writerow(asdict(row))
    return out.getvalue().encode("utf-8-sig")

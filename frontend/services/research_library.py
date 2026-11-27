"""Read-only, allowlisted catalog for AURUM-AI V5 Research Center / Library.

This service never creates, changes, or deletes any project file.  SHA-256 and
preview reads are performed only at a user's explicit request in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Iterable


ALLOWED_EXTENSIONS = frozenset({
    ".json", ".csv", ".parquet", ".npz", ".md", ".txt", ".pdf",
    ".png", ".jpg", ".jpeg", ".svg", ".html", ".ipynb",
    ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".xlsx",
})
# Deny executable/source, credentials, caches and potentially sensitive paths.
SKIP_DIRECTORIES = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".streamlit", "node_modules", "secrets", "credentials", "auth", "sessions",
})
BLOCKED_NAME_FRAGMENTS = (
    ".env", "password", "passwd", "secret", "credential", "private_key",
    "access_token", "api_key", "oauth", "session_cookie", "service_account",
)

# Scopes do not overlap; their names are displayed directly in the UI.
SCOPES = (
    ("Rapports", "reports"),
    ("Expériences IA", "artifacts/experiments"),
    ("Documents", "docs"),
    ("Documents", "documentation"),
    ("Documents", "research"),
    ("Features", "data/03_features"),
    ("Fenêtres ML", "data/04_windows"),
    ("Model Data", "data/05_model_data"),
)
ROOT_DOCUMENTS = ("README.md", "README.txt", "PROJECT_OVERVIEW.md")
MAX_CATALOG_FILES = 5000


@dataclass(frozen=True)
class LibraryItem:
    category: str
    relative_path: str  # POSIX path, rooted in AURUM-AI
    name: str
    extension: str
    size_bytes: int
    modified_at_utc: str


def _is_blocked_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    if any(part.lower() in SKIP_DIRECTORIES for part in parts[:-1]):
        return True
    return any(fragment in part.lower() for part in parts for fragment in BLOCKED_NAME_FRAGMENTS)


def _allowed_path(path: Path, root: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root)
        return (
            resolved_path.suffix.lower() in ALLOWED_EXTENSIONS
            and not _is_blocked_path(relative)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _make_item(path: Path, root: Path, category: str) -> LibraryItem | None:
    if not _allowed_path(path, root):
        return None
    try:
        stat = path.stat()
        relative = path.relative_to(root)
        if _is_blocked_path(relative):
            return None
        return LibraryItem(
            category=category,
            relative_path=relative.as_posix(),
            name=path.name,
            extension=path.suffix.lower(),
            size_bytes=stat.st_size,
            modified_at_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .strftime("%Y-%m-%d %H:%M UTC"),
        )
    except (OSError, RuntimeError, ValueError, OverflowError):
        return None


def list_library(repo_root: Path | str, max_files: int = MAX_CATALOG_FILES) -> list[LibraryItem]:
    """Catalog pre-existing allowlisted research outputs; never follow symlinks."""
    if max_files < 1:
        return []
    root = Path(repo_root).resolve()
    if not root.is_dir():
        return []
    found: list[LibraryItem] = []
    seen: set[str] = set()

    def add(path: Path, category: str) -> bool:
        item = _make_item(path, root, category)
        if item is not None and item.relative_path not in seen:
            seen.add(item.relative_path)
            found.append(item)
        return len(found) >= max_files

    for name in ROOT_DOCUMENTS:
        if add(root / name, "Documents"):
            break

    if len(found) < max_files:
        for category, location in SCOPES:
            base = root / location
            if not base.is_dir() or base.is_symlink():
                continue
            for folder, directories, filenames in os.walk(base, followlinks=False):
                directories[:] = sorted(
                    name for name in directories
                    if name.lower() not in SKIP_DIRECTORIES
                    and not (Path(folder) / name).is_symlink()
                    and not any(frag in name.lower() for frag in BLOCKED_NAME_FRAGMENTS)
                )
                for name in sorted(filenames):
                    if add(Path(folder) / name, category):
                        break
                if len(found) >= max_files:
                    break
            if len(found) >= max_files:
                break

    return sorted(found, key=lambda item: (item.category, item.relative_path.casefold()))


def filter_library(
    items: Iterable[LibraryItem],
    search: str = "",
    category: str = "Toutes",
    extension: str = "Toutes",
) -> list[LibraryItem]:
    needle = search.strip().casefold()
    return [
        item for item in items
        if (category == "Toutes" or item.category == category)
        and (extension == "Toutes" or item.extension == extension)
        and (not needle or needle in item.relative_path.casefold())
    ]


def resolve_library_item(repo_root: Path | str, item: LibraryItem) -> Path:
    """Revalidate an item at read time, including its scope and symlink status."""
    root = Path(repo_root).resolve(strict=True)
    relative = Path(item.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Chemin hors du projet interdit")
    path = root / relative
    if not _allowed_path(path, root):
        raise ValueError("Fichier refusé, absent ou hors périmètre autorisé")
    valid_scope = any(
        path.is_relative_to(root / folder) and item.category == name
        for name, folder in SCOPES
    ) or (item.relative_path in ROOT_DOCUMENTS and item.category == "Documents")
    if not valid_scope:
        raise ValueError("Périmètre du catalogue non valide")
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a digest by streaming bytes (does not alter file)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if value < 1024 or unit == "Go":
            return f"{value:.1f} {unit}" if unit != "o" else f"{size_bytes} o"
        value /= 1024
    return f"{size_bytes} o"


def catalog_summary(items: Iterable[LibraryItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts

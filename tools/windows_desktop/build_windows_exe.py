"""Generate a conservative PyInstaller --onedir preview on the user's Windows PC.

Usage, from project root:
  python tools/windows_desktop/build_windows_exe.py --check
  python tools/windows_desktop/build_windows_exe.py --build

Do not upload the resulting bundle publicly before auditing included files.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re

LEGACY_BACKUP_PATTERN = re.compile(r'(?:^|[_\-.])(?:backup|before|old|bak|copy|archive)(?:[_\-.]|$)', re.I)
import subprocess
import sys
from pathlib import Path

SOURCE_FOLDERS = (
    'frontend', 'agents', 'blockchain', 'risk', 'backtesting', 'features',
    'pipelines', 'pipeline', 'strategies', 'execution', 'portfolio',
    'models', 'model', 'src', 'data_pipeline', 'utils', 'config',
)
DATA_FOLDERS = ('assets', 'artifacts', 'reports', 'data', 'datasets')
RUNTIME_EXTS = {
    '.py', '.pyi', '.json', '.csv', '.npz', '.npy', '.parquet', '.md',
    '.txt', '.yaml', '.yml', '.toml', '.png', '.jpg', '.jpeg', '.svg',
    '.ico', '.html', '.css', '.js', '.jsonl', '.feather', '.arrow',
    '.pkl', '.pickle', '.joblib', '.onnx', '.pt', '.pth', '.ckpt', '.h5',
    '.hdf5', '.safetensors', '.bin', '.sqlite', '.db', '.pdf',
}
# Never package credentials/keys or local browser/account stores silently.
BLOCKED_SUFFIXES = {'.pem', '.key', '.p12', '.pfx', '.crt', '.cer'}
BLOCKED_DIRS = {'.venv', '.git', '.pytest_cache', '__pycache__',
                'node_modules', 'build', 'dist', '.streamlit', 'tests'}
BLOCKED_NAMES = {'.env', '.env.local', '.env.production',
                 'secrets.toml', 'credentials.json', 'token.json',
                 'users.db', 'accounts.db', 'users.sqlite', 'accounts.sqlite'}
BASIC_PACKAGES = ('streamlit', 'webview', 'pandas', 'numpy', 'plotly',
                  'altair', 'pyarrow', 'reportlab', 'PIL')
OPTIONAL_PACKAGES = ('scipy', 'sklearn', 'xgboost', 'torch', 'tensorflow',
                     'keras', 'matplotlib', 'sqlalchemy', 'requests')


def safe_file(path: Path) -> bool:
    if path.name.lower() in BLOCKED_NAMES or path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if any(part.lower() in BLOCKED_DIRS for part in path.parts):
        return False
    name = path.name.lower()
    # Keep source backups locally, never in the distributed application.
    if path.suffix.lower() == '.py' and LEGACY_BACKUP_PATTERN.search(path.stem):
        return False
    # Keep authentication SOURCE modules (e.g. password_unlock.py): only block
    # secret-bearing DATA filenames, never generic Python filenames.
    if path.suffix.lower() != '.py' and (name.startswith('.env') or 'secret' in name or 'password' in name):
        return False
    if path.suffix.lower() not in RUNTIME_EXTS:
        return False
    return True


def enumerate_runtime_data(root: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for folder in SOURCE_FOLDERS + DATA_FOLDERS:
        d = root / folder
        if not d.is_dir():
            continue
        for p in d.rglob('*'):
            if p.is_file() and safe_file(p.relative_to(root)):
                records.append((str(p), str(p.parent.relative_to(root))))
    # A few repository root metadata files are sometimes read by reports.
    for name in ('README.md', 'pyproject.toml'):
        file = root / name
        if file.is_file():
            records.append((str(file), '.'))
    return sorted(set(records))


def scan_imports(root: Path, records: list[tuple[str, str]]) -> set[str]:
    roots: set[str] = set()
    for file, _dest in records:
        if not file.endswith('.py'):
            continue
        try:
            tree = ast.parse(Path(file).read_text(encoding='utf-8-sig'))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                roots.update(v.name.partition('.')[0] for v in n.names)
            if isinstance(n, ast.ImportFrom) and n.module:
                roots.add(n.module.partition('.')[0])
    return roots


def pkg_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def make_spec(root: Path, records: list[tuple[str, str]], packages: list[str], console: bool) -> str:
    # Precomputed validation-only UI bundle. No deep learning training.

    icon = root / 'assets' / 'branding' / 'aurum_ai.ico'
    entry = root / 'tools' / 'windows_desktop' / 'frozen_entry.py'
    spec = f'''# AURUM-AI V5 - Lean Windows desktop preview (validation/read-only).
# Review the data manifest before distribution. Training frameworks excluded.
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, collect_dynamic_libs, copy_metadata,
)

project_root = Path({str(root)!r})
datas = {records!r}
binaries = []

# Streamlit dynamically loads modules, but do not recursively collect every
# submodule of NumPy, PyArrow, ReportLab, Torch, etc.  Standard PyInstaller
# hooks manage their necessary binary extensions and imports.
hiddenimports = collect_submodules(
    'streamlit',
    filter=lambda name: not any(
        section in ('test', 'tests', 'testing')
        for section in name.split('.')
    ),
    on_error='warn once',
)
hiddenimports += ['webview.platforms.winforms', 'webview.platforms.edgechromium']
hiddenimports += ['keyring', 'keyring.backends.Windows']

# Streamlit loads frontend/auth/login_ui.py through runpy, as DATA rather than
# through PyInstaller Analysis. Its import of frontend.auth.security then loads
# argon2.PasswordHasher. Explicit collection is required or the frozen window
# fails with ModuleNotFoundError: No module named 'argon2'.
# Keep the Argon2 CFFI backend and its native binary for password verification.
for auth_module in ('argon2', '_argon2_cffi_bindings'):
    hiddenimports += collect_submodules(
        auth_module,
        filter=lambda name: not any(
            part in ('test', 'tests', 'testing') for part in name.split('.')
        ),
        on_error='warn once',
    )
hiddenimports += ['argon2', '_argon2_cffi_bindings', 'cffi', 'frontend.auth.security']
# The authentication views are executed by runpy, not statically imported by
# the frozen entry point. Include their module graph too, to prevent a chain
# of follow-up missing imports for accounts, mail and login validation.
hiddenimports += collect_submodules(
    'frontend.auth',
    filter=lambda name: not any(
        (part in ('test', 'tests', 'testing') or
         any(piece in ('backup', 'before', 'old', 'bak', 'copy', 'archive')
             for piece in part.replace('-', '_').split('_')))
        for part in name.split('.')
    ),
    on_error='warn once',
)
binaries += collect_dynamic_libs('_argon2_cffi_bindings')

datas += copy_metadata('keyring')

for pkg in ('streamlit', 'webview', 'altair', 'plotly', 'reportlab'):
    datas += collect_data_files(pkg, excludes=['**/tests/**', '**/test/**', '**/__pycache__/**'])

for dist_name in ('streamlit', 'pywebview', 'altair', 'plotly', 'pandas',
                  'numpy', 'pyarrow', 'reportlab', 'tornado', 'protobuf',
                  'argon2-cffi', 'argon2-cffi-bindings', 'cffi'):
    try:
        datas += copy_metadata(dist_name)
    except Exception:
        pass

# Frozen UI consumes precomputed experiment artifacts rather than retraining.
# This prevents huge optional DL frameworks and their benchmark/test modules
# from being pulled in through optional imports.
ui_excludes = [
    'torch', 'torchvision', 'torchaudio', 'tensorflow',
    'keras', 'triton', 'torchmetrics',
]

a = Analysis(
    [{str(entry)!r}], pathex=[str(project_root)], binaries=binaries,
    datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={{}},
    runtime_hooks=[], excludes=ui_excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='AURUM-AI V5',
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=False, console={console!r}, icon={str(icon)!r})
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name='AURUM-AI V5')
'''
    return spec


def prepare(root: Path, check_only: bool, console: bool) -> Path:
    if not (root / 'frontend' / 'app.py').is_file():
        raise FileNotFoundError('frontend/app.py missing: run from AURUM-AI root')
    icon = root / 'assets' / 'branding' / 'aurum_ai.ico'
    if not icon.is_file():
        raise FileNotFoundError(f'Logo ICO missing: {icon}')
    entry = root / 'tools' / 'windows_desktop' / 'frozen_entry.py'
    if not entry.is_file():
        raise FileNotFoundError(f'Launcher missing: {entry}')
    if not pkg_present('PyInstaller'):
        raise RuntimeError('PyInstaller not installed in the current .venv')
    if not pkg_present('streamlit') or not pkg_present('webview'):
        raise RuntimeError('Install streamlit and pywebview in .venv')
    if not pkg_present('argon2') or not pkg_present('_argon2_cffi_bindings'):
        raise RuntimeError(
            'Authentication backend missing in .venv: '
            'python -m pip install argon2-cffi'
        )
    data = enumerate_runtime_data(root)
    imports = scan_imports(root, data)
    chosen = [p for p in BASIC_PACKAGES if pkg_present(p)]
    # UI-only preview: keep built-in training frameworks OUT of the bundle.
    size = sum(Path(f).stat().st_size for f, _ in data)
    print(f'Files included in prototype: {len(data)} | approx {size / 1024**3:.2f} GiB')
    print('UI packages detected:', ', '.join(chosen))
    print('Optimized collection: Streamlit and Argon2 auth, no collect_all(torch/numpy/pyarrow).')
    print('Warning: training/inference that imports torch/tensorflow is NOT supported in this preview.')
    ui_imports = {name for name in ('torch', 'tensorflow', 'keras') if name in scan_imports(root, [rec for rec in data if ('frontend' + os.sep) in rec[0]])}
    if ui_imports:
        print('ATTENTION: imports lourds detectes dans frontend:', ', '.join(sorted(ui_imports)))
        print('Certaines pages ne fonctionneront pas sans version ML complete.')
    print('Filtered secrets, virtual environments, tests and caches are excluded.')
    print('VERIFY all 20 pages after building; a passing compilation is not proof.')
    print('CAUTION: locally created user accounts/secrets are NOT bundled.')
    spec_path = root / 'AURUM-AI-V5-Windows.spec'
    if check_only:
        print('Preflight only, no files written.')
        return spec_path
    spec_path.write_text(make_spec(root, data, chosen, console), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', str(spec_path)],
                   cwd=root, check=True)
    print('Build output:', root / 'dist' / 'AURUM-AI V5' / 'AURUM-AI V5.exe')
    return spec_path


def main() -> int:
    p = argparse.ArgumentParser(description='AURUM-AI V5 Windows EXE pre-release')
    operation = p.add_mutually_exclusive_group(required=True)
    operation.add_argument('--check', action='store_true')
    operation.add_argument('--build', action='store_true')
    p.add_argument('--console', action='store_true',
                   help='Build console EXE for easier diagnostics, without changing the app')
    args = p.parse_args()
    if sys.platform != 'win32':
        raise RuntimeError('Build must run on Windows for a Windows executable')
    root = Path(__file__).resolve().parents[2]
    prepare(root, check_only=args.check, console=args.console)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

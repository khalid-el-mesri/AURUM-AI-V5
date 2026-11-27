"""Read-only static smoke audit for AURUM-AI's Python and onedir Windows shell.

Does not start login, inspect SQLite contents or claim runtime page functionality.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

EXPECTED = {
    'executive': 'executive.py',
    'market-analysis': 'market_analysis.py',
    'data-quality': 'data_quality.py',
    'feature-engineering': 'feature_engineering.py',
    'ai-prediction': 'ai_prediction.py',
    'ai-model-lab': 'ai_model_lab.py',
    'autonomous-agents': 'autonomous_agents.py',
    'agent-decision-room': 'agent_decision_room.py',
    'digital-twin': 'digital_twin.py',
    'portfolio-simulator': 'portfolio_simulator.py',
    'risk-management': 'risk_management.py',
    'backtesting': 'backtesting.py',
    'stress-testing': 'stress_testing.py',
    'what-if-simulator': 'what_if_simulator.py',
    'blockchain-ledger': 'blockchain_ledger.py',
    'blockchain-explorer': 'blockchain_explorer.py',
    'library': 'library.py',
    'experiments': 'experiments.py',
    'reports': 'reports.py',
}


def inspect_routes(text: str) -> dict[str, str]:
    tree = ast.parse(text)
    routes: dict[str,str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'page':
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        filename = node.args[0].value
        url = next((kw.value.value for kw in node.keywords if kw.arg == 'url' and isinstance(kw.value, ast.Constant)), None)
        if url is None and len(node.args) >= 4 and isinstance(node.args[3], ast.Constant):
            url = node.args[3].value
        if isinstance(url,str) and isinstance(filename,str):
            if url in routes and routes[url] != filename:
                raise ValueError('Route dupliquee : '+url)
            routes[url] = filename
    return routes


def audit(root: Path, packaged: bool = False) -> list[str]:
    root = root.resolve()
    app_root = root / 'dist' / 'AURUM-AI V5' / '_internal' if packaged else root
    problems: list[str] = []
    app = app_root / 'frontend' / 'app.py'
    if not app.is_file():
        return [f'Application absente : {app}']
    try:
        routes = inspect_routes(app.read_text(encoding='utf-8-sig'))
    except (SyntaxError, UnicodeError, ValueError) as exc:
        return [f'Navigation non analysable : {exc}']
    for url,filename in EXPECTED.items():
        if routes.get(url) != filename:
            problems.append(f'Route absente ou differente : /{url} -> {filename}')
    for url in routes:
        if url not in EXPECTED and url != 'login':
            problems.append(f'Route non prevue : /{url}')
    for filename in (*EXPECTED.values(), 'auth.py'):
        if not (app_root / 'frontend' / 'v5_pages' / filename).is_file():
            problems.append(f'Page manquante : {filename}')
    for extra in ('frontend/auth/database.py', 'frontend/auth/login_ui.py',
                  'assets/branding/aurum_ai_icon.png', 'assets/branding/aurum_ai_wordmark.png'):
        if not (app_root / extra).is_file():
            problems.append(f'Ressource manquante : {extra}')
    if packaged:
        exe = root / 'dist' / 'AURUM-AI V5' / 'AURUM-AI V5.exe'
        if not exe.is_file():
            problems.append('EXE absent : '+str(exe))
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AURUM-AI desktop static preflight (read only)')
    parser.add_argument('--packaged', action='store_true', help='Verify dist onedir (after recompiling)')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    errors = audit(args.project_root, packaged=args.packaged)
    print(f'AURUM-AI audit fichiers : {len(EXPECTED)} routes authentifiees + login')
    print('Mode : '+('exe onedir' if args.packaged else 'sources Python'))
    if errors:
        for error in errors:
            print('ERREUR : '+error)
        parser.exit(1, f'{len(errors)} probleme(s) a examiner.\n')
    print('OK : navigation et ressources attendues presentes. (Ne prouve pas le fonctionnement des pages.)')

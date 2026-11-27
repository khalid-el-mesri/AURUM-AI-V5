"""Tests for isolated source-code patches; no third-party dependency required."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import zipfile

import pytest

MODULE_FILE = Path(__file__).resolve().parents[1] / "tools/windows_desktop/fix_phase22_1_ui.py"
spec = importlib.util.spec_from_file_location("phase22_1_fix", MODULE_FILE)
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)

LEDGER_STUB = '''import streamlit as st\n\ndef _hero() -> None:\n    st.markdown(\'\'\'<style>\n    .foo{ color: red }\n    </style>\n    <div>Header</div>\'\'\',\n        unsafe_allow_html=True)\n\ndef _hash_card(title, text):\n    pass\n'''
TWIN_STUB = '''import pandas as pd\ndef f(report):\n    return pd.DataFrame([{"Contrôle": k, "Valeur": v} for k, v in report.items()])\n'''
EXPERIMENTS_STUB = '''import pandas as pd\ndef f(scalar):\n    return pd.DataFrame([{"Champ": k, "Valeur": v} for k,v in scalar.items()])\n'''


def test_header_patched_only_one_function():
    result, changed = fix.patch_ledger(LEDGER_STUB)
    assert changed
    assert 'def _hash_card(title, text):' in result
    assert 'def _hero()' in result
    ast.parse(result)


def test_header_dedents_before_markdown():
    result, _ = fix.patch_ledger(LEDGER_STUB)
    module = ast.parse(result)
    hero = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_hero")
    called = []
    class StreamlitStub:
        def markdown(self, text, **kw):
            called.append((text,kw))
    namespace = {'st': StreamlitStub()}
    exec(compile(ast.Module(body=[hero], type_ignores=[]), '<hero>', 'exec'), namespace)
    namespace['_hero']()
    assert called[0][0].startswith('<style>')
    assert '\n<div' in called[0][0]
    assert called[0][1]['unsafe_allow_html'] is True
    assert 'Blockchain Ledger' in called[0][0]


def test_header_idempotent():
    result, _ = fix.patch_ledger(LEDGER_STUB)
    second, changed = fix.patch_ledger(result)
    assert not changed and second == result


def test_header_refuses_unexpected():
    with pytest.raises(ValueError):
        fix.patch_ledger('def _hero() -> None:\n    pass\ndef _hash_card(): pass')


def test_twin_casts_display_only():
    result, changed = fix.patch_twin(TWIN_STUB)
    assert changed
    assert '{"Contrôle": str(k), "Valeur": str(v)}' in result
    assert 'report.items()' in result
    ast.parse(result)


def test_twin_idempotent():
    result, _ = fix.patch_twin(TWIN_STUB)
    assert fix.patch_twin(result) == (result, False)


def test_twin_refuses_unknown():
    with pytest.raises(ValueError): fix.patch_twin('no dataframe')


def test_experiments_casts_display_only():
    result, changed = fix.patch_experiments(EXPERIMENTS_STUB)
    assert changed and '"Valeur": str(v)' in result
    ast.parse(result)


def test_experiments_idempotent():
    result,_ = fix.patch_experiments(EXPERIMENTS_STUB)
    assert fix.patch_experiments(result) == (result,False)


def test_experiments_refuses_unknown():
    with pytest.raises(ValueError): fix.patch_experiments('no dataframe')


def test_prepare_no_mutation(tmp_path):
    for relative,s in [(fix.LEDGER_REL,LEDGER_STUB),(fix.TWIN_REL,TWIN_STUB),
                       (fix.EXPERIMENTS_REL,EXPERIMENTS_STUB)]:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s,encoding='utf8')
    assert len(fix.prepare(tmp_path))==3
    assert (tmp_path/fix.LEDGER_REL).read_text(encoding='utf8')==LEDGER_STUB


def test_apply_backup_and_idempotent(tmp_path):
    for relative,s in [(fix.LEDGER_REL,LEDGER_STUB),(fix.TWIN_REL,TWIN_STUB),
                       (fix.EXPERIMENTS_REL,EXPERIMENTS_STUB)]:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s,encoding='utf8')
    assert all(changed for _,changed in fix.run(tmp_path,apply=True))
    assert all(not changed for _,changed in fix.run(tmp_path,apply=False))
    assert len(list(tmp_path.rglob('*.bak')))==3


def test_preserve_windows_newlines(tmp_path):
    for relative,s in [(fix.LEDGER_REL,LEDGER_STUB),(fix.TWIN_REL,TWIN_STUB),
                       (fix.EXPERIMENTS_REL,EXPERIMENTS_STUB)]:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(s.replace('\n','\r\n').encode())
    fix.run(tmp_path,apply=True)
    assert b'\r\n' in (tmp_path/fix.LEDGER_REL).read_bytes()

"""AURUM-AI V5 / Reports — immutable historical research-report assembly.

Only known Validation artifacts are read. No arbitrary paths, checkpoint loads,
new predictions, live trades, external calls, or on-disk exports. Output bytes
are created in memory. All conclusions are descriptive and source-bounded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
import csv
import hashlib
import io
import json
import math

MODELS = ("LSTM", "GRU", "CNN1D")
HORIZONS = (5, 15, 30, 60)
EXPERIMENTS = Path("artifacts/experiments/nyse_v3")
BACKTEST = Path("reports/backtesting/combined_holding_30m_60m")
BLOCKCHAIN = Path("reports/blockchain/consensus_trend_60m_validation")
FILES = {
    "historical_summary": BACKTEST / "comparison_summary.json",
    "decisions_60m": BACKTEST / "60m_decisions.csv",
    "transactions_60m": BACKTEST / "60m_transactions.csv",
    "ledger": BLOCKCHAIN / "ledger.json",
    "blockchain_audit": BLOCKCHAIN / "audit_summary.json",
}
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CSV_BYTES = 12 * 1024 * 1024
MAX_CSV_ROWS = 100000
MAX_DISPLAY_SCALARS = 50
REDACTION_TERMS = (
    "password", "passwd", "secret", "private_key", "api_key", "access_token",
    "authorization", "credential", "smtp_password", "client_secret", "token",
)


def run_path(model: str, horizon: int) -> Path:
    if model not in MODELS or type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("Modèle ou horizon non autorisé")
    return EXPERIMENTS / f"{model.lower()}_{horizon}m_seed42" / "report.json"


def allowed_sources() -> dict[str, Path]:
    sources = dict(FILES)
    for horizon in HORIZONS:
        for model in MODELS:
            sources[f"validation_{model.lower()}_{horizon}m"] = run_path(model, horizon)
    return sources


def safe_source(root: Path | str, key: str) -> Path:
    """Whitelisted file, rejecting symlinked ancestors and file symlinks."""
    rel = allowed_sources().get(key)
    if rel is None:
        raise ValueError("Source non autorisée")
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError("Racine invalide")
    path = base
    for segment in rel.parts:
        path = path / segment
        if path.is_symlink():
            raise ValueError("Lien symbolique refusé")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(base):
        raise FileNotFoundError(rel.as_posix())
    return path


def reject_constant(value: str) -> None:
    raise ValueError("NaN ou Infinity interdits")


def redact(data: Any, depth: int = 0) -> Any:
    """Defensive redaction and bounded representation for presentation/export."""
    if depth > 12:
        return "[DEPTH LIMIT]"
    if isinstance(data, dict):
        return {
            str(k)[:150]: ("[REDACTED]" if any(term in str(k).lower() for term in REDACTION_TERMS)
                           else redact(v, depth + 1))
            for k, v in list(data.items())[:500]
        }
    if isinstance(data, list):
        return [redact(x, depth + 1) for x in data[:300]]
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if data is None or isinstance(data, (str, int, float, bool)):
        return data if not isinstance(data, str) else data[:2000]
    return str(data)[:200]


def read_json(root: Path | str, key: str) -> dict[str, Any]:
    path = safe_source(root, key)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON trop volumineux")
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle, parse_constant=reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("Objet JSON attendu")
    return redact(raw)


def csv_profile(root: Path | str, key: str, preview: int = 12) -> dict[str, Any]:
    if key not in ("decisions_60m", "transactions_60m"):
        raise ValueError("CSV non autorisé")
    path = safe_source(root, key)
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV trop volumineux")
    total = 0
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) > 120:
            raise ValueError("Trop de colonnes dans le CSV")
        for row in reader:
            total += 1
            if total > MAX_CSV_ROWS:
                raise ValueError("Trop de lignes dans le CSV")
            if len(records) < max(0, min(int(preview), 30)):
                records.append({field[:120]: str(row.get(field) or "")[:250]
                                for field in fields
                                if not any(s in field.lower() for s in REDACTION_TERMS)})
    return {"rows": total, "columns": [f for f in fields if not any(s in f.lower() for s in REDACTION_TERMS)],
            "preview": records}


def flatten_scalars(data: dict[str, Any], *, limit: int = MAX_DISPLAY_SCALARS) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    def walk(obj: Any, key: str, depth: int) -> None:
        if len(output) >= limit or depth > 5:
            return
        if isinstance(obj, dict):
            for k, value in obj.items():
                walk(value, f"{key}.{k}" if key else str(k), depth + 1)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            val = str(obj)
            output.append({"Champ source": key[:160], "Valeur source": val[:240]})
    walk(redact(data), "", 0)
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def source_inventory(root: Path | str, *, with_hashes: bool = False) -> list[dict[str, Any]]:
    base = Path(root)
    inventory = []
    for key, rel in allowed_sources().items():
        row: dict[str, Any] = {"key": key, "path": rel.as_posix(), "status": "absent", "bytes": None}
        try:
            path = safe_source(base, key)
            row["status"] = "present"
            row["bytes"] = path.stat().st_size
            if with_hashes:
                row["sha256"] = sha256(path)
        except (FileNotFoundError, OSError):
            pass
        except ValueError as exc:
            row["status"] = "refusé"
            row["reason"] = str(exc)
        inventory.append(row)
    return inventory


def _read_or_error(root: Path, key: str, fmt: str) -> tuple[Any, str | None]:
    try:
        return (read_json(root, key) if fmt == "json" else csv_profile(root, key)), None
    except (FileNotFoundError, OSError, ValueError, TypeError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def report_snapshot(root: Path | str, *, with_hashes: bool = False) -> dict[str, Any]:
    """Gather a factual snapshot. Missing/invalid sources are NOT reconstructed."""
    base = Path(root).resolve(strict=True)
    sources = source_inventory(base, with_hashes=with_hashes)
    by_key = {r["key"]: r for r in sources}
    model_runs = []
    for horizon in HORIZONS:
        for model in MODELS:
            key = f"validation_{model.lower()}_{horizon}m"
            entry = {"model": model, "horizon_minutes": horizon,
                     "source": allowed_sources()[key].as_posix(),
                     "file_status": by_key[key]["status"], "json_valid": False}
            if by_key[key]["status"] == "present":
                report, error = _read_or_error(base, key, "json")
                entry["json_valid"] = error is None
                if error:
                    entry["error"] = error[:300]
                elif isinstance(report, dict):
                    # Do not impose a schema or infer a model winner.
                    entry["report_excerpt"] = flatten_scalars(report, limit=12)
            model_runs.append(entry)

    historic, hist_error = _read_or_error(base, "historical_summary", "json")
    ledger_audit, audit_error = _read_or_error(base, "blockchain_audit", "json")
    decisions, dec_error = _read_or_error(base, "decisions_60m", "csv")
    transactions, tx_error = _read_or_error(base, "transactions_60m", "csv")

    return {
        "project": "AURUM-AI V5", "report_scope": "Historical XOM / NYSE V3 Validation only",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "no_live_trading": True, "test_generalization_not_established": True,
        "sources": sources, "model_runs": model_runs,
        "model_reports_valid": sum(bool(x["json_valid"]) for x in model_runs),
        "historical_summary": historic,
        "historical_summary_error": hist_error,
        "historical_decisions": decisions,
        "historical_decisions_error": dec_error,
        "historical_transactions": transactions,
        "historical_transactions_error": tx_error,
        "blockchain_audit": ledger_audit,
        "blockchain_audit_error": audit_error,
        "ledger_present": by_key["ledger"]["status"] == "present",
        "limitations": [
            "Les performances affichées sont historiques, sur Validation, et ne garantissent pas de performances futures.",
            "Aucun résultat Test indépendant n'est inféré des artefacts Validation.",
            "Le reporting ne relance ni entraînement, ni prédiction, ni agent, ni ordre.",
            "La présence d'une chaîne SHA-256 locale n'est pas une authentification externe ni une blockchain publique.",
            "L'absence d'anomalie de métadonnées ne prouve pas l'absence de fuite temporelle.",
            "Le dossier historique a pu être construit avec une sélection rétrospective de signaux : interprétation prudente.",
        ],
    }


def json_report_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def inventory_csv_bytes(snapshot: dict[str, Any]) -> bytes:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["key", "path", "status", "bytes", "sha256", "reason"],
                            extrasaction="ignore")
    writer.writeheader()
    for row in snapshot["sources"]:
        writer.writerow(row)
    return out.getvalue().encode("utf-8-sig")


def html_report_bytes(snapshot: dict[str, Any]) -> bytes:
    """Self-contained printable report, all file-derived values HTML-escaped."""
    e = lambda x: escape(str(x), quote=True)
    runs = "".join(
        f"<tr><td>{e(r['model'])}</td><td>{r['horizon_minutes']} min</td>"
        f"<td>{e(r['file_status'])}</td><td>{'Oui' if r['json_valid'] else 'Non'}</td></tr>"
        for r in snapshot["model_runs"]
    )
    summary_rows = flatten_scalars(snapshot["historical_summary"] or {}, limit=18)
    historic = "".join(f"<tr><td>{e(x['Champ source'])}</td><td>{e(x['Valeur source'])}</td></tr>"
                       for x in summary_rows) or '<tr><td colspan="2">Source absente ou non exploitable</td></tr>'
    files = "".join(f"<tr><td>{e(r['path'])}</td><td>{e(r['status'])}</td></tr>" for r in snapshot["sources"])
    risks = "".join(f"<li>{e(item)}</li>" for item in snapshot["limitations"])
    tx = snapshot["historical_transactions"] or {}
    de = snapshot["historical_decisions"] or {}
    css = """body{font:14px/1.55 Arial,sans-serif;color:#162536;margin:40px auto;max-width:960px;padding:0 22px}
    header{background:#0a2942;color:#fff;padding:28px;border-radius:10px}h1{margin:0;font-size:28px}
    header p{color:#aed8e6}.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}
    .metric{padding:16px;background:#edf6fa;border:1px solid #b2d7e5;border-radius:8px;flex:1}
    .metric strong{display:block;font-size:22px}h2{color:#0d4664;border-bottom:2px solid #27b7d8;padding-bottom:6px;margin-top:30px}
    table{width:100%;border-collapse:collapse;font-size:12px;word-break:break-word}td,th{padding:7px;border:1px solid #d4dfe8;text-align:left}
    th{background:#e7f2f7}tr:nth-child(even){background:#f7f9fb}.note{background:#fff5dd;padding:14px;border-left:4px solid #c58c21}
    footer{color:#536c7c;font-size:11px;margin-top:40px} @media print{body{margin:0}.metric,header{print-color-adjust:exact}h2{break-after:avoid}table{break-inside:auto}tr{break-inside:avoid}}"""
    html = f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>AURUM-AI V5 · Rapport historique</title>
    <style>{css}</style></head><body><header><h1>AURUM-AI V5 · Research Report</h1>
    <p>XOM | Validation historique | Simulation uniquement | {e(snapshot['generated_utc'])}</p></header>
    <div class="metrics"><div class="metric">Rapports de modèles valides<strong>{snapshot['model_reports_valid']}/12</strong></div>
    <div class="metric">Décisions CSV<strong>{e(de.get('rows', 'N/D'))}</strong></div>
    <div class="metric">Transactions CSV<strong>{e(tx.get('rows', 'N/D'))}</strong></div></div>
    <p class="note"><b>Portée :</b> résultats historiques descriptifs sur Validation. Ni Test indépendant, ni profit futur démontré.
    Aucune opération réelle exécutée par ce rapport.</p>
    <h2>1. Inventaire des modèles</h2><table><tr><th>Modèle</th><th>Horizon</th><th>Artefact</th><th>JSON lisible</th></tr>{runs}</table>
    <h2>2. Backtesting historique — champs de la source</h2><table><tr><th>Champ du rapport</th><th>Valeur du rapport</th></tr>{historic}</table>
    <h2>3. Provenance des documents</h2><table><tr><th>Chemin relatif</th><th>État</th></tr>{files}</table>
    <h2>4. Limites et méthode</h2><ul>{risks}</ul>
    <footer>AURUM-AI · Document de recherche généré en mémoire depuis les fichiers locaux autorisés. Aucune modification des fichiers sources.</footer>
    </body></html>'''
    return html.encode("utf-8")


def pdf_report_bytes(snapshot: dict[str, Any]) -> bytes:
    """Produce an A4 PDF in memory; reportlab is optional and imported on demand."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from xml.sax.saxutils import escape as xml_escape

    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.7 * cm,
                            rightMargin=1.7 * cm, topMargin=1.7 * cm, bottomMargin=1.6 * cm,
                            title="AURUM-AI V5 — Historical Research Report",
                            author="AURUM-AI Research Center")
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="AurumTitle", parent=ss["Heading1"], fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#0c5670"), fontSize=18, leading=22, alignment=TA_CENTER))
    ss.add(ParagraphStyle(name="AurumBody", parent=ss["BodyText"], fontSize=8.7, leading=12, spaceAfter=6))
    ss.add(ParagraphStyle(name="AurumSmall", parent=ss["BodyText"], fontSize=7.2, leading=9))
    p = lambda value, style="AurumSmall": Paragraph(xml_escape(str(value))[:600], ss[style])
    story: list[Any] = [p("AURUM-AI V5 | Research Report", "AurumTitle"), Spacer(1, 9),
                       p(snapshot["report_scope"], "AurumBody"),
                       p("Généré : " + snapshot["generated_utc"], "AurumBody"),
                       p("SIMULATION UNIQUEMENT | VALIDATION HISTORIQUE | PAS DE TRADING RÉEL", "AurumBody"),
                       Spacer(1, 9)]

    def heading(text: str):
        story.append(Spacer(1, 12)); story.append(p(text, "Heading2"))
    def tab(headers: list[str], rows: list[list[Any]], widths: list[float]):
        table = Table([[p(x) for x in headers]] + [[p(x) for x in r] for r in rows],
                      colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef4")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c3d5df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    heading("1. Synthèse documentaire")
    de, tx = snapshot["historical_decisions"] or {}, snapshot["historical_transactions"] or {}
    story.append(p(f"Rapports des modèles JSON lisibles : {snapshot['model_reports_valid']}/12. "
                   f"Décisions historiques : {de.get('rows', 'N/D')}. "
                   f"Transactions historiques : {tx.get('rows', 'N/D')}. ", "AurumBody"))
    story.append(p("Les chiffres ci-dessus dénombrent les fichiers/lignes historiques lus ; "
                   "ils ne constituent pas une nouvelle validation scientifique.", "AurumBody"))
    heading("2. Inventaire de l'expérimentation")
    tab(["Modèle", "Horizon", "Fichier", "JSON valide"],
        [[r["model"], f"{r['horizon_minutes']} min", r["file_status"], "Oui" if r["json_valid"] else "Non"]
         for r in snapshot["model_runs"]], [3.3*cm, 3*cm, 6.2*cm, 4.4*cm])
    heading("3. Synthèse du backtesting : valeurs rapportées")
    historic = flatten_scalars(snapshot["historical_summary"] or {}, limit=16)
    if historic:
        tab(["Champ source", "Valeur source"], [[row["Champ source"], row["Valeur source"]]
                                            for row in historic], [8.45*cm, 8.45*cm])
    else:
        story.append(p("Rapport de synthèse indisponible ou non exploitable."))
    heading("4. Provenance des artefacts")
    tab(["Source locale", "État"],
        [[r["path"], r["status"]] for r in snapshot["sources"]], [13.6*cm, 3.3*cm])
    heading("5. Limites scientifiques")
    for limitation in snapshot["limitations"]:
        story.append(p("• " + limitation, "AurumBody"))
    pdf.build(story)
    return buffer.getvalue()

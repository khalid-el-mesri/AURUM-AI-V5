AURUM-AI V5 — Phase 18 / Research Center / Reports
===================================================

BUT
- Consolider en lecture seule les preuves de Validation historiques (XOM / NYSE V3).
- Consulter les rapports LSTM / GRU / CNN1D, le backtesting 60 min,
  l'audit blockchain et les sources effectivement presentes.
- Exporter un rapport scientifique imprimable HTML, un PDF (ReportLab optionnel),
  un instantane JSON et un inventaire CSV, sans modifier les fichiers du projet.

INSTALLATION DANS LE REPERTOIRE RACINE AURUM-AI (PowerShell, .venv actif)

1. Sauvegarde de la page actuelle :
   Copy-Item .\frontend\v5_pages\reports.py .\frontend\v5_pages\reports_before_phase18.py

2. Verifier que l'archive est dans Downloads :
   $zip = "$env:USERPROFILE\Downloads\AURUM_AI_V5_Reports_Phase18.zip"
   Test-Path $zip
   # Le resultat doit etre True; sinon corriger le chemin reel.

3. Extraire le ZIP directement a la racine du projet :
   Expand-Archive -LiteralPath $zip -DestinationPath . -Force

4. Optionnel : activer le bouton PDF si ReportLab n'est pas installe :
   python -m pip install "reportlab>=4.2,<5"
   # Sans cela, les exports HTML, CSV, JSON et les pages restent disponibles.

5. Executer les tests :
   python -m pytest tests/test_research_reports_audit.py -q

6. Lancer :
   python -m streamlit run frontend/app.py
   Puis ouvrir Research Center > Reports.

CARACTERISTIQUES
- 5 onglets : Synthese, Modeles IA, Simulation, Integrite, Exports.
- 17 chemins de sources autorises : 12 reports des modeles, 3 sources
  de backtesting et 2 sources blockchain.
- Si une source manque, afficher son absence; ne pas fabriquer un resultat.
- Aucune modification d'artifact et aucun entrainement/inference/test/trade.
- Champs de nature sensible masques dans les apercus et exports.
- Les empreintes SHA-256 sont calculees seulement a la demande.
- La verification de chaine utilise la logique blockchain.ledger existante.
- Les chiffres issus des CSV sont des NOMBRES DE LIGNES (pas de nouvelles
  performances ou transactions). Les autres valeurs sont affichees telles
  que decrites dans les sources.

ATTENTION / LIMITES
- Une evaluation Validation seule ne prouve pas la generalisation Test.
- La coherence locale d'une chaine SHA-256 n'est pas une certification externe.
- Les scenarios historiques What-If/Stress sont distincts des experiences Test.
- L'export PDF depend de reportlab (nouvelle dependance optionnelle).
- Le programme conserve les autres pages et le shell d'authentification V5.
- Les avertissements use_container_width des autres modules restent a traiter
  en phase finale de maintenance et packaging Windows.

FICHIERS DU PACKAGE
- frontend/services/research_reports.py
- frontend/v5_pages/reports.py
- tests/test_research_reports_audit.py
- README_PHASE18.txt

FINALE : empaquetage de bureau Windows apres validation des modules et tests
complets. La Phase 18 n'est PAS encore l'application .exe.

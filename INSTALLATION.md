# AURUM-AI V5 — Phase 13 / Blockchain Ledger

## Objectif

Remplacer uniquement la page Streamlit provisoire `frontend/v5_pages/blockchain_ledger.py`
par un tableau de bord d'audit cryptographique **en lecture seule**. La vérification repose sur
le moteur existant du projet `blockchain/ledger.py` via `BlockchainLedger.load_json()`.

## Fichiers contenus dans l'archive

```
frontend/services/blockchain_ledger_audit.py
frontend/v5_pages/blockchain_ledger.py
tests/test_blockchain_ledger_audit.py
INSTALLATION.md
```

L'archive **NE contient PAS** et **NE modifie PAS** :
- `blockchain/ledger.py` ;
- `reports/blockchain/consensus_trend_60m_validation/ledger.json` ;
- des décisions, prédictions, portefeuilles, données brutes ou résultats de backtesting.

## Commandes PowerShell

Exécuter depuis le dossier racine `AURUM-AI`, avec `.venv` activé.

1. Arrêter le processus Streamlit courant : `Ctrl + C` dans son terminal.

2. Sauvegarder la page existante :

```powershell
Copy-Item .\frontend\v5_pages\blockchain_ledger.py .\frontend\v5_pages\blockchain_ledger_before_phase13.py -Force
```

3. Vérifier le téléchargement :

```powershell
Test-Path "$env:USERPROFILE\Downloads\AURUM_AI_V5_BlockchainLedger.zip"
```

Si `False`, télécharger l'archive depuis ChatGPT, puis rechercher son emplacement. Ne pas
utiliser `Expand-Archive` tant que le fichier n'existe pas.

4. Décompresser dans la racine du projet :

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_BlockchainLedger.zip" -DestinationPath . -Force
```

5. Vérifier les fichiers :

```powershell
Test-Path .\frontend\services\blockchain_ledger_audit.py
Test-Path .\tests\test_blockchain_ledger_audit.py
Test-Path .\reports\blockchain\consensus_trend_60m_validation\ledger.json
```

6. Tester :

```powershell
python -m pytest tests/test_blockchain_ledger_audit.py -q
python -m pytest tests/test_blockchain_ledger.py tests/test_blockchain_integration.py -q
```

La première série comporte **40 tests** dans l'environnement de création. Leur résultat
sur ton ordinateur reste à confirmer. Les tests existants servent à détecter des régressions.

7. Lancer :

```powershell
python -m streamlit run frontend/app.py
```

Puis ouvrir `http://localhost:8501/blockchain-ledger`.

## Comportement attendu

- Quand le `ledger.json` existant est intact, la page doit afficher une chaîne locale vérifiée ;
  le nombre exact de blocs et d'événements est calculé à partir de TON fichier.
- En cas de fichier absent ou corrompu, la page affiche une erreur, **sans montrer les
  événements comme fiables**.
- Les six onglets affichent l'overview, l'inspection d'un bloc, l'intégrité complète,
  l'historique filtrable, la provenance et les exports.
- Aucune fonction `append_event` ou `save_json` n'est appelée par cette page.
- Les exports sont construits en mémoire avec les données après vérification.

## Limites méthodologiques

1. Une chaîne SHA-256 locale, même intègre, **n'est pas une blockchain publique ni une
   preuve externe d'authenticité** : un auteur peut réécrire les événements puis recalculer
   l'ensemble des hash.
2. Le hash SHA-256 du fichier JSON est mesuré à la lecture. Sans copie ancrée hors projet,
   une comparaison indépendante n'est pas possible.
3. L'intégrité cryptographique ne garantit ni l'authenticité des données de marché ni
   la validité des prédictions ni la rentabilité.
4. Les résultats affichés proviennent de la simulation historique `Consensus + Trend 60m`
   sur Validation, pas des modèles sélectionnés dans AI Prediction.
5. Les avertissements Streamlit `use_container_width` dans d'autres pages sont indépendants
   de cette phase.

## En cas de problème

- Si `ModuleNotFoundError: blockchain`: vérifie que Streamlit est lancé avec
  `python -m streamlit run frontend/app.py` **depuis la racine AURUM-AI**.
- Si la page n'apparaît pas, vérifie que `frontend/app.py` pointe déjà vers
  `frontend.v5_pages.blockchain_ledger` (l'ancien placeholder était accessible à
  `/blockchain-ledger`). Ne change pas le routeur si ce lien est déjà actif.
- Si l'intégrité échoue, conserver le fichier original pour diagnostic au lieu de
  recalculer ou remplacer les hash du rapport.

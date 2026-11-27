AURUM-AI V5 — Phase 21.5.1 : correctif WinError 32 SQLite
=========================================================

PROBLEME
Sous Windows, test_migrate_preserves_all_tables echouait au moment de supprimer
le fichier aurum_auth_migration_*.sqlite3. Contrairement a une idee courante,
``with sqlite3.connect(...) as con`` ne ferme PAS la connexion : il ne fait
que gerer commit/rollback. Un handle SQLite restait donc ouvert.

CORRECTION
Le script tools/windows_desktop/migrate_auth_storage.py utilise desormais
contextlib.closing pour toutes les connexions ouvertes dans inspect_db() et
migrate(), y compris en cas d'erreur. Les bases originales ne sont pas
modifiees et une cible existante n'est jamais remplacee.

FICHIERS DU ZIP
- tools/windows_desktop/migrate_auth_storage.py  (remplace le script bogue)
- tests/test_phase21_5_sqlite_windows_lock.py    (3 nouveaux tests)
- README_PHASE21_5_1.txt

ETAPES — DEPUIS LA RACINE AURUM-AI DANS POWERSHELL
1. Fermez les fenetres AURUM-AI Python/EXE.
2. Sauvegardez le script actuel :
   Copy-Item .\tools\windows_desktop\migrate_auth_storage.py .\tools\windows_desktop\migrate_auth_storage_before_phase21_5_1.py.bak
3. Extrayez le ZIP :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase21_5_1_SQLiteWindowsFix.zip" -DestinationPath . -Force
4. Lancez les tests locaux :
   python -m pytest tests/test_phase21_5_auth_storage.py tests/test_phase21_5_sqlite_windows_lock.py -q
5. Verifiez les bases (lecture seule, aucune migration) :
   python tools/windows_desktop/migrate_auth_storage.py --check

IMPORTANT
N'executez PAS --apply si la base persistent existe deja ou si vous n'avez
pas identifie dans quelle base se trouve le compte attendu. Ne transmettez
pas de fichier users.sqlite3, de mot de passe ni de hash. Le test passe sur
Linux de notre cote, mais doit aussi etre valide sur votre Windows.
Une nouvelle compilation PyInstaller n'est PAS necessaire pour ce correctif
car il s'agit d'un outil de migration lance depuis Python du projet.
Le chemin persistant de la base est gere separement par la Phase 21.5.

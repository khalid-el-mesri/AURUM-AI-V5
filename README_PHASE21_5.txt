AURUM-AI V5 - PHASE 21.5 - Stockage local persistant de l'authentification

Cause confirmée : database.py utilisait __file__ pour construire :
    AURUM-AI/runtime/auth/users.sqlite3 (version Python)
    dist/AURUM-AI V5/_internal/runtime/auth/users.sqlite3 (version EXE)
Le diagnostic précédent ne cherchait pas dans runtime/ : faux négatif.

Le nouveau code utilisera :
    %LOCALAPPDATA%/AURUM-AI/auth/users.sqlite3
pour TOUTES LES VERSIONS (Python et Windows EXE).

IMPORTANT :
- Fermer l'application originale et l'EXE avant une migration.
- Sauvegarder frontend/auth/database.py avant de l'écraser.
- Ne jamais publier/copier la base users.sqlite3 dans un ZIP ou dépôt Git.
- Ce package NE CONTIENT AUCUN compte ou base personnelle.
- Argon2, statuts de vérification et tables originales restent inchangés.

À la racine du projet dans PowerShell (venv actif) :

1) Décompresser ce package.
2) Lire sans modifier :
   python tools/windows_desktop/migrate_auth_storage.py --check
3) Les 3 emplacements sont affichés avec existence, nombre de comptes,
   et nombre de comptes confirmés (pas d'email, ni hash).
4) SI la base 'project' contient tes comptes ET 'persistent' est absente :
   python tools/windows_desktop/migrate_auth_storage.py --apply --source project
   Si la base du projet est absente mais celle du package contient ton compte,
   tu peux examiner la source 'packaged' avant de la copier :
   python tools/windows_desktop/migrate_auth_storage.py --apply --source packaged
   Aucun fichier existant ne sera écrasé. En cas de conflit, STOP.
5) Vérifier le résultat :
   python tools/windows_desktop/migrate_auth_storage.py --check
   python -m pytest tests/test_phase21_5_auth_storage.py -q
   python -m pytest -q --disable-warnings --maxfail=1
6) IMPORTANT : reconstruire le .exe pour intégrer le nouveau database.py :
   python tools/windows_desktop/build_windows_exe.py --build --console
   & ".\dist\AURUM-AI V5\AURUM-AI V5.exe"

Si aucune source ne possède le compte, la migration ne peut pas deviner
le mot de passe ni contourner la vérification de l'adresse email.

Après connexion : changer le mot de passe qui avait été visible sur une capture.

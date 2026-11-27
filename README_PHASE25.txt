AURUM-AI V5 - Phase 25 - Audit prepublication

ETAT: controle en lecture seule. Ne modifie pas l'application, les comptes, les artefacts XOM ni l'installateur.

1. Fermer AURUM-AI avant toute manipulation de fichiers. La version installee peut rester presente.
2. Dans PowerShell du projet avec (.venv) :

  Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase25_DistributionAudit.zip" -DestinationPath . -Force

3. Tests de l'outil :

  python -m pytest tests/test_phase25_distribution_audit.py -q

4. Audit de ta copie reelle :

  python tools/windows_desktop/audit_distribution_phase25.py

Le script ne cherche pas a lire le contenu des comptes ni a modifier des fichiers. Un code retour 2 signifie qu'au moins un constat BLOQUANT necessite une correction avant distribution. Un code retour 0 ne signifie PAS que la distribution est sure : seule une revue manuelle et des essais sur Windows propre permettront de la valider.

5. Lire docs/PHASE25_PLAN_VALIDATION.md. Ne pas desinstaller l'application du PC de travail.

IMPORTANT: ne pas envoyer a ChatGPT de mots de passe, .env, hashes Argon2, cookies, fichiers utilisateurs, contenus de base SQLite ou cles privees. Partager uniquement le resultat texte du script, en masquant les chemins personnels si necessaire.

AURUM-AI V5 — Phase 21.4: Diagnostic de stockage des comptes (lecture seule)

But : déterminer pourquoi l'application Windows compilée refuse un compte
alors que l'interface Streamlit originale peut utiliser une autre base locale.

Le diagnostic NE CHANGE NI CODE D'AUTHENTIFICATION, NI BASE, NI MOT DE PASSE.
Il n'affiche aucun email, mot de passe, hash, code OTP ou ligne de base.
Il inspecte seulement des fichiers source et la présence de stockages candidats.

Depuis la racine AURUM-AI, dans PowerShell (.venv active) :

1. Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase21_4_AuthStorageDiagnostic.zip" -DestinationPath . -Force
2. python -m pytest tests/test_diagnose_auth_storage.py -q
3. python tools/windows_desktop/diagnose_auth_storage.py

Option, si le premier rapport ne permet pas de déterminer si le compte existe :
   python tools/windows_desktop/diagnose_auth_storage.py --check-account
   Saisir uniquement l'adresse email du compte lorsque Python la demande.
   La saisie est masquée; ne jamais saisir un mot de passe à ce stade.

Partager le résultat TEXTUEL du diagnostic, SANS fichiers de base de données,
SANS contenu de frontend/auth/security.py ni secrets.toml.

Hypothèses à confirmer :
- Les modules sous dist/.../_internal peuvent utiliser des chemins relatifs
  différents de ceux du projet source.
- La base de comptes est volontairement exclue de la distribution (bonne pratique).
- Le compte peut exister mais ne pas être vérifié, ou le mot de passe différer.

Après diagnostic, corrigera explicitement la localisation vers un stockage
permanent du profil Windows (%LOCALAPPDATA%/AURUM-AI), si nécessaire, puis
ajoutera une migration non destructive et des tests d'authentification.
Ne pas distribuer la base privée dans l'installeur Windows.

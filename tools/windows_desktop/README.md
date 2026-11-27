# AURUM-AI V5 — Phase 20 / Préversion Desktop Windows

Ce paquet est **additif** : il n'écrase aucun module, test ou artefact historique d'AURUM-AI. Il sert à vérifier l'expérience de lancement en fenêtre native **avant** de construire un installateur `.exe`.

## Installation à la racine du projet (PowerShell)

Lancer ces commandes depuis `C:\Users\Pc\Desktop\projet de deep learning\AURUM-AI` avec `.venv` activé.

```powershell
$zip = "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase20_DesktopPreview.zip"
Test-Path $zip
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
python -m pip install pywebview
python -m pytest tests/test_desktop_launcher_phase20.py -q
python desktop_launcher.py
```

**Résultat attendu :** une fenêtre AURUM-AI dédiée s'ouvre avec le même écran de connexion et les mêmes pages qu'avant. Fermer la fenêtre arrête le serveur Streamlit lancé par ce programme. L'ancien `python -m streamlit run frontend/app.py` reste possible et inchangé.

Le serveur de cette préversion écoute **uniquement `127.0.0.1`**, sur un port local automatiquement choisi. Il n'est pas accessible via l'adresse Wi-Fi/LAN par ce processus. Le journal du serveur est dans `%LOCALAPPDATA%\AURUM-AI\logs\desktop_streamlit.log`.

### Si la fenêtre ne s'ouvre pas

Vérifier que `pywebview` s'est bien installé dans `.venv` et que Microsoft Edge WebView2 Runtime fonctionne sur Windows. Consulter le journal mentionné ci-dessus. N'exécuter **aucune** modification des modules `frontend`, `risk`, `agents`, `blockchain`, des modèles ou des données.

### Périmètre

- **Fait :** prototype de fenêtre de bureau, lancement/arrêt du serveur, connexion locale seulement, conservation de l'interface authentifiée.
- **À faire :** audit UI/sécurité de bout en bout, test Windows sur le PC utilisateur, gestion des ressources et des chemins hors dépôt, création de l'exécutable et d'un véritable installateur, éventuelle signature.
- **Non couvert :** cette préversion n'est **pas** un `.exe` autonome; elle utilise Python et les dépendances de `.venv`.

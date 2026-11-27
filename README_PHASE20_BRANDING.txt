AURUM-AI V5 | Phase 20 - Identite visuelle Bull et raccourci Windows
=====================================================================

IMPORTANT
- Ce package contient les images officielles de la variante Bull dore / bleu.
- Il NE contient PAS de .exe Windows compilé. Il installe l'icone, le favicon,
  le logo dans la navigation Streamlit et un lanceur Desktop.
- Aucun modele ni donnees brutes ne sont modifies.
- Le launcher desktop est une preversion locale qui utilise pywebview et Streamlit.
- Les fichiers existants frontend/app.py et desktop_launcher.py sont sauvegardes
  automatiquement par le script avant modification.
- Ne pas ecraser manuellement frontend/app.py avec un ancien ZIP.

Installation dans PowerShell, depuis la racine AURUM-AI et .venv actif :

1. Sauvegarde de precaution generale recommandee :
   Copy-Item .\frontend\app.py .\frontend\app_before_logo_manual.bak

2. Decompresser ce ZIP dans le projet :
   $zip = "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase20_BullBranding.zip"
   Test-Path $zip
   Expand-Archive -LiteralPath $zip -DestinationPath . -Force

3. Installer pywebview et tester le nouveau code :
   python -m pip install "pywebview>=6.2"
   python -m pytest tests/test_aurum_branding_phase20.py -q

4. Controler sans modifier les fichiers (DRY RUN) :
   python tools/windows_desktop/apply_branding.py --check

5. Appliquer le branding (sauvegardes .bak automatiques) :
   python tools/windows_desktop/apply_branding.py --apply

6. Controler tests existants :
   python -m pytest -q --disable-warnings --maxfail=1

7. Tester la version desktop en mode terminal pour voir les erreurs :
   python desktop_launcher.py

8. Creer l'icone cliquable sur le Bureau :
   .\tools\windows_desktop\create_shortcut.ps1

9. Double-cliquer sur 'AURUM-AI V5' sur le Bureau pour ouvrir l'app.

La fenetre utilise une adresse locale 127.0.0.1. Vous devez garder les fichiers du
projet et l'environnement virtuel .venv presents a cet emplacement. Une application
.exe installable autonome demande une phase distincte de packaging et tests Windows.

Si le launcher ne demarre pas, executez python desktop_launcher.py et consultez :
  $env:LOCALAPPDATA\AURUM-AI\logs\desktop_streamlit.log

Pour annuler : restaurer le dernier frontend/app.py.before_branding_*.bak ainsi que
le dernier desktop_launcher.py.before_branding_*.bak apres avoir ferme l'application.

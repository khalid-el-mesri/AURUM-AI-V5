AURUM-AI V5 - Phase 21.2 - Correction du demarrage Streamlit PyInstaller
===========================================================================
Probleme reproduit dans le journal Windows :
RuntimeError: server.port does not work when global.developmentMode is true.

Dans le package PyInstaller, le code Streamlit ne se trouve plus sous site-packages.
Sa detection automatique peut donc activer global.developmentMode=true.
Le mode developpement est incompatible avec le port local configure.

Le correctif concerne UN SEUL fichier de runtime :
 tools/windows_desktop/frozen_entry.py
Il definit STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false dans le processus serveur et
ajoute --global.developmentMode=false aux options CLI Streamlit.
Ne modifie pas les 20 pages, le dataset ou les modeles.

TEST IMMEDIAT DE L'ANCIEN EXE (aucune recompilation):
 $env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE = 'false'
 & '.\dist\AURUM-AI V5\AURUM-AI V5.exe'
 Pour revenir a l'environnement original ensuite :
 Remove-Item Env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE -ErrorAction SilentlyContinue

INSTALLATION DU CORRECTIF SOURCE depuis la racine AURUM-AI :
 Copy-Item .\tools\windows_desktop\frozen_entry.py .\tools\windows_desktop\frozen_entry_before_phase21_2.py.bak
 Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase21_2_StreamlitStartFix.zip" -DestinationPath . -Force
 python -m pytest tests/test_phase21_streamlit_devmode_fix.py tests/test_windows_packaging_phase21.py -q

RECONSTRUCTION (une fois le test temporaire de l'EXE concluant) :
 python tools/windows_desktop/build_windows_exe.py --build --console
 & '.\dist\AURUM-AI V5\AURUM-AI V5.exe'

ATTENTION : PyInstaller doit copier TOUS les fichiers du dossier dist pour
fonctionner. Testez l'authentification, 20 pages, export PDF, provenance,
permissions et expiration de session avant livraison. Ne distribuez pas des
fichiers secrets ou des donnees de comptes utilisateurs. Aucun resultat de
trading reel n'est produit : simulations et donnees historiques uniquement.

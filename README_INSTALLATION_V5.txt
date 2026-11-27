AURUM-AI V5 — PHASE 1 : SECURE NAVIGATION (18 October 2026)
==============================================================

CONTENU
-------
19 pages accessibles après connexion, + l'écran Login / Sign Up / Locked.
- Executive Dashboard, NYSE V3 Backtesting, Blockchain Explorer : réutilisation
  fidèle du code fourni, exécuté derrière le contrôle d'authentification.
- AI Model Lab : comparaison des 12 rapports Validation fournis par votre projet.
- 15 autres pages : interfaces clairement marquées « Module prévu ».

Le dossier frontend/v5_legacy contient des copies dérivées des 2 dashboards.
Vos anciens frontend/dashboard_v3.py, dashboard_v4.py et frontend/pages/
NYSE_V3_Backtesting.py restent INCHANGES.

PRÉREQUIS
---------
- Le système d'authentification existant doit déjà être installé :
  frontend/auth/{login.py,accounts.py,database.py,registration.py,
  security.py,verification.py,email_service.py,idle_guard.py}
- Le composant HTML doit rester installé :
  frontend/auth/idle_component/index.html
- Les artefacts du backtesting, blockchain et 12 expériences doivent rester
  aux emplacements attendus par les scripts originaux.
- Streamlit avec st.navigation/st.Page et st.fragment ; Plotly, Pandas, etc.
- NE PAS partager de .env, SQLite, mot de passe ni token Gmail.

INSTALLATION (PowerShell, RACINE du projet)
-------------------------------------------
1. Stoppez Streamlit : Ctrl+C.
2. Sauvegardez :
   Copy-Item .\frontend\app.py .\frontend\app_before_v5.py
   Copy-Item .\frontend\auth\login_ui.py .\frontend\auth\login_ui_before_v5.py
3. Vérifiez que votre timeout est à 300 :
   Select-String .\frontend\auth\idle_guard.py -Pattern IDLE_TIMEOUT_SECONDS
4. Téléchargez AURUM_AI_V5_Phase1.zip dans Downloads, puis :
   Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\AURUM_AI_V5_Phase1.zip" -DestinationPath . -Force
5. Lancez UNIQUEMENT :
   python -m streamlit run frontend/app.py

TESTS SÉCURITÉ & NAVIGATION
---------------------------
A. Ouvrez http://localhost:8501. Seul Login doit apparaître.
B. Essayez /backtesting ou une ancienne URL /NYSE_V3_Backtesting
   sans être connecté : elles ne doivent PAS donner accès au contenu.
C. Connectez-vous : V5 doit afficher la navigation et Executive Dashboard.
D. Ouvrez Backtesting, Blockchain Explorer, AI Model Lab.
E. Laissez l'application inactive 5 minutes : Session Locked (mot de passe seul).
F. Testez un mauvais mot de passe, puis le vrai.
G. Déconnectez-vous : plus aucune page ne doit rester accessible.
H. Vérifiez l'inscription / OTP si vous souhaitez rouvrir sa validation.

NOTE SUR L'ERREUR StreamlitWidgetAlreadyInstantiatedError
--------------------------------------------------------
login_ui.py corrige le changement de mode des onglets après inscription
ou OTP : on met aurum_next_auth_mode et le changement est appliqué lors
DU PROCHAIN rerun, avant le widget segmented_control.

RESTAURATION RAPIDE
-------------------
Arrêtez Streamlit puis :
 Copy-Item .\frontend\app_before_v5.py .\frontend\app.py -Force
 Copy-Item .\frontend\auth\login_ui_before_v5.py .\frontend\auth\login_ui.py -Force

LIMITES ACTUELLES
-----------------
- OAuth Google non implémenté.
- OTP pas validé bout-en-bout selon le dernier test utilisateur.
- Les pages « Module prévu » ne font aucun calcul fictif.
- Garde basée sur la session Streamlit (application locale académique) ;
  ce patch n'est pas un audit de sécurité d'un déploiement Internet.
- Testé par analyse/syntaxe Python uniquement ; exécuter les tests locaux.

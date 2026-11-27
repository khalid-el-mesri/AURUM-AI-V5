# AURUM-AI V5 — Phase 25 : validation avant distribution

**Version sous vérification : 0.9.0 (préversion).** Ne pas renommer en 1.0.0 avant la vérification de bout en bout. Tous les signaux financiers sont issus de données historiques de simulation, pas de trading réel.

## Étape A — Audit automatisé sans aucune modification

1. Sur le PC de développement, **conserver** la version installée et tous les fichiers du projet.
2. Dans VS Code PowerShell à la racine AURUM-AI, activer `.venv`.
3. Exécuter :

   ```powershell
   python -m pytest tests/test_phase25_distribution_audit.py -q
   python tools/windows_desktop/audit_distribution_phase25.py
   ```

4. Envoyer le rapport textuel. Ne pas partager de secrets, hashes de mots de passe, base SQLite ou contenu de fichiers privés.
5. Si des **BLOQUANT** apparaissent, NE PAS distribuer l'installateur. Réviser le packaging et reconstruire **d'abord l'EXE, ensuite l'installateur**. Un test passant ne prouve pas qu'il n'existe aucun secret dans les archives compilées.

## Ce que le contrôle vérifie réellement

- Existence de la distribution PyInstaller onedir, de l'installateur et du script Inno Setup.
- Présence de fichiers aux noms sensibles dans la distribution (bases de données, secrets, clés).
- Présence de copies Python de sauvegarde ou d'anciennes versions dans le bundle.
- Recherche heuristique d'assignations de secrets **littéraux** dans le code d'authentification et certains fichiers `.py` livrés (valeurs masquées).
- `HOST=127.0.0.1` déclaré dans le lanceur et paramètre d'adresse explicite.
- Empreinte SHA-256 du programme installé comparée à celle du build courant.
- Existence d'une base d'authentification dans `%LOCALAPPDATA%\AURUM-AI\auth`, sans lire son contenu.
- Recherche de règles de désinstallation manifestement dangereuses pour la base de comptes.

**Limites :** Ce n'est pas une inspection des modules Python compilés au sein du PYZ/EXE, un antivirus, un audit des dépendances, un test de trafic réel, ni une certification de sécurité. La présence d'une base SQLite persistante ne démontre pas que l'installateur la préserve lors d'une mise à jour. Le scan par noms peut rater des secrets intégrés ailleurs. Le test SHA-256 ne compare que le lanceur EXE, pas chaque ressource.

## Étape B — Contrôles manuels essentiels

- [ ] Confirmer que l'EXE fonctionne après fermeture de VS Code et arrêt de toute instance Streamlit externe.
- [ ] Depuis le menu Démarrer, connexion, Executive, Prediction, Backtesting, Ledger, Reports.
- [ ] Vérifier 20 pages et **utiliser réellement** les filtres et téléchargements : CSV, JSON, PDF. Ouvrir les fichiers téléchargés.
- [ ] Déconnexion, reconnexion ; verrouillage après 5 minutes d'inactivité et déverrouillage par mot de passe.
- [ ] Vérifier sur la machine d'essai que le serveur écoute seulement `127.0.0.1` (vérification système, pas seulement le code).
- [ ] Vérifier absence d'identifiants SMTP, API ou comptes dans `frontend/auth`, configs et sources intégrées. Ne pas publier le code ni les données sans audit.
- [ ] Confirmer que les **données XOM**, fichiers de modèles et registres d'audit indispensables sont réellement disponibles hors de l'environnement de développement.
- [ ] Confirmer que le message « Research only, 100% simulated, no live trading » est présent dans toutes les pages concernées.

## Étape C — Installer/désinstaller uniquement sur Windows propre

**Attention : ne pas désinstaller sur le PC actuel avec ses comptes.** Utiliser une VM/snapshot Windows ou un compte de test isolé ; n'utiliser ni vrais secrets ni véritables comptes.

- [ ] Créer un instantané de VM Windows. Prévoir le runtime Microsoft Edge WebView2 s'il est absent.
- [ ] Installer `AURUM-AI-V5-Setup-0.9.0.exe` dans le dossier proposé et noter la taille/durée approximatives.
- [ ] Vérifier raccourcis Bureau (si sélectionné), menu Démarrer, taureau, lancement sans `python` installé, aucune fenêtre de terminal visible.
- [ ] Créer un **compte fictif** de test ; vérifier email uniquement si un service de mail d'essai est configuré (ne pas contourner la vérification).
- [ ] Naviguer dans toutes les pages et tester les exports.
- [ ] Fermer l'application. Vérifier l'arrêt du serveur local sans processus orphelin.
- [ ] Désinstaller via Paramètres Windows > Applications.
- [ ] Vérifier la disparition du dossier du programme et des raccourcis.
- [ ] Vérifier que les données persistantes du compte fictif sont préservées selon la politique annoncée.
- [ ] Réinstaller et vérifier que l'accès au compte de test est conservé.
- [ ] Documenter erreurs, captures sans données personnelles et comportement réel.

## Étape D — Finitions et publication

- [ ] Corriger « Deploy » toujours visible dans Secure Access en conservant les défenses et fonctions de Streamlit.
- [ ] Uniformiser Backtesting et lisibilité/zoom de l'interface.
- [ ] Reconstruire `dist`, puis Inno Setup, après toute modification de code ; incrémenter la version à 1.0.0 seulement après tous les contrôles.
- [ ] Examiner les dépendances et droits de redistribution des bibliothèques et jeux de données ; signer l'installateur si distribué publiquement.
- [ ] Fournir README utilisateur, changelog, avertissement sur la simulation et politique de conservation/suppression des comptes.

# AURUM-AI V5 — Windows Desktop Acceptance Checklist

*Version de test : Phase 22 · Simulation historique XOM · aucune transaction réelle*

**Ne transmettez jamais de mot de passe, code OTP, hash Argon2, fichier SQLite ou capture contenant des identifiants.**

## 1. Démarrage, sécurité et arrêt

- [ ] Exécuter `AURUM-AI V5.exe` depuis le dossier `dist/AURUM-AI V5` et voir le logo du taureau (fenêtre / barre des tâches).
- [ ] Page Secure Access visible ; l'en-tête présente le taureau et le nom AURUM-AI.
- [ ] Élément `Deploy` absent.
- [ ] Vérifier que le tableau de bord ne s'ouvre pas sans connexion, même via un lien direct `http://127.0.0.1:<port>/executive`.
- [ ] Compte existant reconnu, sans création d'un nouveau compte et sans perte de confirmation email.
- [ ] Déconnexion : les pages sécurisées ne sont plus accessibles sans identification.
- [ ] Verrouillage après cinq minutes d'inactivité et déverrouillage par mot de passe (la vérification doit être effectuée dans l'EXE).
- [ ] Fermer la fenêtre Desktop : le processus serveur associé s'arrête (vérifier dans le Gestionnaire des tâches si nécessaire).

## 2. Pages authentifiées — navigation manuelle

Cocher une page seulement si elle s'ouvre **sans erreur**, affiche son contenu et, le cas échéant, accepte les filtres/exports attendus. Tester les pages depuis le menu **de l'EXE**, pas uniquement dans le navigateur de développement.

### Overview
- [ ] `/executive` — Executive Dashboard (résultats simulés lisibles)

### Market & AI
- [ ] `/market-analysis` — Market Analysis
- [ ] `/data-quality` — Data Quality
- [ ] `/feature-engineering` — Feature Engineering
- [ ] `/ai-prediction` — AI Prediction
- [ ] `/ai-model-lab` — AI Model Lab (12 expériences Validation et graphiques)

### Autonomous Intelligence
- [ ] `/autonomous-agents` — Autonomous Agents
- [ ] `/agent-decision-room` — Agent Decision Room
- [ ] `/digital-twin` — Digital Twin

### Portfolio & Risk
- [ ] `/portfolio-simulator` — Portfolio Simulator
- [ ] `/risk-management` — Risk Management
- [ ] `/backtesting` — Backtesting (Validation, pas Test indépendant)
- [ ] `/stress-testing` — Stress Testing
- [ ] `/what-if-simulator` — What-If Simulator

### Blockchain
- [ ] `/blockchain-ledger` — Blockchain Ledger (chaîne SHA-256 locale)
- [ ] `/blockchain-explorer` — Blockchain Explorer

### Research Center
- [ ] `/library` — Library (recherche/aperçu)
- [ ] `/experiments` — Experiments (registre et métadonnées)
- [ ] `/reports` — Reports (aperçu, téléchargement PDF/HTML/JSON/CSV)

## 3. Données / reproductibilité
- [ ] Les fichiers `01_RAW` et les modèles originaux ne sont pas modifiés.
- [ ] Aucune nouvelle prédiction 'Test' ni performance future n'est prétendue.
- [ ] Fichiers Validation présents et explicitement marqués comme tels.
- [ ] Rapports, graphiques et fichiers d'expériences utiles s'ouvrent sous le dossier `dist`.
- [ ] Les éventuels journaux d'erreur ne divulguent pas d'informations de connexion.

**Résultat à rapporter :** total de pages sur 19 ouvertes, tout message d'erreur non sensible, et état des tests logout/verrouillage/exports. Les contrôles automatiques de la Phase 22 ne prouvent pas ces comportements : ils doivent être vérifiés réellement.

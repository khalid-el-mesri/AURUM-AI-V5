
"""AURUM-AI — Premium authentication interface.

Standalone development page:
python -m streamlit run frontend/auth/login_ui.py

Supports:
- Local email/password login
- Account registration
- Gmail OTP verification and resend
- Local session logout

Google OAuth is not yet implemented. Automatic inactivity lock is enabled.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.auth.accounts import AccountAlreadyExistsError
from frontend.auth.login import authenticate_user
from frontend.auth.registration import (
    AccountVerificationError,
    RegistrationDeliveryError,
    confirm_email_registration,
    register_and_send_otp,
    resend_verification_otp,
)
from frontend.auth.verification import OTPRequestError
from frontend.auth.idle_guard import (
    check_local_session,
    end_local_session,
    render_idle_watchdog,
    start_local_session,
    unlock_local_session,
)


# ------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# ------------------------------------------------------------

if not st.session_state.get("aurum_v5_mode", False):
    st.set_page_config(
        page_title="AURUM-AI | Secure Authentication",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


# ------------------------------------------------------------
# 2. PREMIUM BACKGROUND
# ------------------------------------------------------------

BACKGROUND_PATH = ROOT / "frontend" / "assets" / "refinery_banner.jpg"


def load_background() -> str:
    if not BACKGROUND_PATH.is_file():
        return ""

    return base64.b64encode(
        BACKGROUND_PATH.read_bytes()
    ).decode("ascii")


background = load_background()

style = """
<style>
.stApp {
    background:
        linear-gradient(
            105deg,
            rgba(2, 13, 29, 0.93) 0%,
            rgba(3, 20, 39, 0.79) 46%,
            rgba(2, 12, 29, 0.94) 100%
        ),
        url("data:image/jpeg;base64,__BACKGROUND__");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #F0F7FF;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1320px;
    padding-top: 4rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F0F7FF !important;
}

.aurum-brand {
    font-size: 44px;
    font-weight: 900;
    letter-spacing: -2px;
    color: white;
    margin-bottom: 5px;
}

.aurum-brand span {
    color: #20C7FF;
}

.aurum-eyebrow {
    color: #49D5FF;
    font-size: 12px;
    letter-spacing: 4px;
    font-weight: 700;
    margin-bottom: 45px;
}

.aurum-headline {
    font-size: 39px;
    font-weight: 850;
    line-height: 1.22;
    color: #F0F7FF;
    margin-bottom: 18px;
}

.aurum-highlight {
    color: #2FD6FF;
}

.aurum-description {
    color: #BCD1E4;
    font-size: 16px;
    line-height: 1.8;
    max-width: 560px;
    margin-bottom: 35px;
}

.aurum-feature {
    background: rgba(5, 25, 46, 0.79);
    border: 1px solid rgba(48, 169, 230, 0.38);
    border-radius: 12px;
    padding: 17px 20px;
    margin-bottom: 13px;
    color: #EAF5FF;
}

.aurum-feature-title {
    font-size: 15px;
    font-weight: 750;
}

.aurum-feature-sub {
    font-size: 12px;
    color: #9FBBD3;
    margin-top: 4px;
}

[data-testid="stForm"] {
    background: rgba(3, 18, 36, 0.93);
    border: 1px solid #23618B;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

div[data-testid="stTextInput"] input {
    background: #0B223A;
    color: white;
    border: 1px solid #285377;
    border-radius: 8px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #27C5F5;
}

div[data-testid="stFormSubmitButton"] button[kind="primary"],
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #03B9DF, #087CE4);
    color: white;
    border: none;
    font-weight: 700;
}

div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: linear-gradient(90deg, #14CEF4, #1599FA);
}

/* AURUM_CONTRAST_V1 - Secure Access */
.st-key-aurum_use_remembered_login button,
.st-key-aurum_forget_remembered_login button {
    background: #123957 !important;
    color: #F4FBFF !important;
    border: 1px solid #4B9EC8 !important;
    font-weight: 650 !important;
    opacity: 1 !important;
}

.st-key-aurum_use_remembered_login button *,
.st-key-aurum_forget_remembered_login button * {
    color: #F4FBFF !important;
    opacity: 1 !important;
}

.st-key-aurum_use_remembered_login button:hover,
.st-key-aurum_forget_remembered_login button:hover {
    background: #205579 !important;
    border-color: #35C7F4 !important;
    color: #FFFFFF !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label *,
[data-testid="stWidgetLabel"] {
    color: #D6E8F7 !important;
    opacity: 1 !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #A9C4D9 !important;
    opacity: 1 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #BFD7E9 !important;
}

.aurum-footer {
    font-size: 12px;
    color: #90ABC5;
    text-align: center;
    margin-top: 20px;
}

@media (max-width: 800px) {
    .block-container {
        padding-top: 1.5rem;
    }

    .aurum-brand {
        font-size: 32px;
    }

    .aurum-headline {
        font-size: 29px;
    }
}
</style>
"""

st.markdown(
    style.replace("__BACKGROUND__", background),
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. SESSION STATE
# ------------------------------------------------------------

if "aurum_auth_mode" not in st.session_state:
    st.session_state["aurum_auth_mode"] = "Connexion"

# Defer changes to segmented-control state until the NEXT script run,
# before that widget is instantiated. Fixes widget-state mutation errors.
next_auth_mode = st.session_state.pop("aurum_next_auth_mode", None)
if next_auth_mode in ("Connexion", "Créer un compte", "Vérifier l'email"):
    st.session_state["aurum_auth_mode"] = next_auth_mode

if "aurum_pending_email" not in st.session_state:
    st.session_state["aurum_pending_email"] = ""

if "aurum_local_user" not in st.session_state:
    st.session_state["aurum_local_user"] = None

# Enforce expiration whenever the full Streamlit script executes.
check_local_session()


# ------------------------------------------------------------
# 4. PAGE LAYOUT
# ------------------------------------------------------------

left, right = st.columns([1.05, 0.95], gap="large")


with left:

    # AURUM PHASE22: official bull + wordmark in secure access
    _aurum_icon = ROOT / "assets" / "branding" / "aurum_ai_icon.png"
    _aurum_wordmark = ROOT / "assets" / "branding" / "aurum_ai_wordmark.png"
    if _aurum_icon.is_file() and _aurum_wordmark.is_file():
        _aurum_symbol, _aurum_label = st.columns(
            [1, 3.3], gap="small", vertical_alignment="center"
        )
        with _aurum_symbol:
            st.image(str(_aurum_icon), width="stretch")
        with _aurum_label:
            st.image(str(_aurum_wordmark), width="stretch")
    else:
        st.markdown("## AURUM-AI")


    st.markdown(
        """

        <div class="aurum-eyebrow">
            QUANTITATIVE INTELLIGENCE
        </div>

        <div class="aurum-headline">
            Intelligent Investment<br>
            <span class="aurum-highlight">
                Simulation Platform
            </span>
        </div>

        <div class="aurum-description">
            Plateforme académique d'intelligence artificielle
            dédiée à l'analyse des données historiques
            d'Exxon Mobil (XOM), à la recherche quantitative
            et à la simulation d'investissement autonome.
            <br><br>
            Aucune transaction réelle.
        </div>

        <div class="aurum-feature">
            <div class="aurum-feature-title">
                📈 Historical Market Research
            </div>
            <div class="aurum-feature-sub">
                Exxon Mobil (XOM) · 2017–2026 · Données 5 minutes
            </div>
        </div>

        <div class="aurum-feature">
            <div class="aurum-feature-title">
                🧠 AI & Deep Learning
            </div>
            <div class="aurum-feature-sub">
                LSTM · GRU · CNN1D · 12 expériences
            </div>
        </div>

        <div class="aurum-feature">
            <div class="aurum-feature-title">
                🛡️ Autonomous Risk Management
            </div>
            <div class="aurum-feature-sub">
                Agents de stratégie, risque et portefeuille
            </div>
        </div>

        <div class="aurum-feature">
            <div class="aurum-feature-title">
                🔗 Blockchain Audit
            </div>
            <div class="aurum-feature-sub">
                Traçabilité et vérification d'intégrité
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

    st.markdown("## 🔐 Secure Access")

    st.caption(
        "Bienvenue sur AURUM-AI — Accès à votre espace de recherche."
    )

    # --------------------------------------------------------
    # AUTHENTICATED USER
    # --------------------------------------------------------

    if st.session_state.pop("aurum_clear_login_inputs", False):
        st.session_state.pop("aurum_login_email", None)
        st.session_state.pop("aurum_login_password", None)

    if st.session_state.pop("aurum_remember_save_failed", False):
        st.warning("Connexion reussie, mais adresse email non memorisee.")

    user = st.session_state["aurum_local_user"]

    if user is not None and st.session_state.get("aurum_session_locked", False):

        # The account remains known, but access is disabled until the
        # password for THIS SAME account has been successfully verified.
        st.markdown("### 🔒 Session Locked")
        st.warning(
            st.session_state.get("aurum_session_lock_reason")
            or "Votre session est verrouillée."
        )
        st.markdown(f"**Compte :** {user['display_name']}")
        st.caption(f"{user['email']}")

        with st.form("aurum_unlock_form", clear_on_submit=True):
            unlock_password = st.text_input(
                "Mot de passe",
                type="password",
                placeholder="Votre mot de passe",
            )
            unlock_submitted = st.form_submit_button(
                "🔓 Déverrouiller ma session",
                type="primary",
                width="stretch",
            )

        if unlock_submitted:
            if unlock_local_session(unlock_password):
                st.rerun()
            else:
                st.error("Mot de passe incorrect. La session reste verrouillée.")

    elif user is not None:

        # Periodic backend expiration check + genuine browser activity events.
        render_idle_watchdog()

        st.success(
            f"Bienvenue, {user['display_name']} !"
        )

        st.info(
            "Votre connexion locale est validée. "
            "La liaison avec le Dashboard sera ajoutée "
            "après les tests de sécurité."
        )

        st.write(f"**Email :** {user['email']}")

        if st.button(
            "🚪 Se déconnecter",
            width="stretch",
        ):
            end_local_session()
            st.rerun()

    else:

        mode = st.segmented_control(
            "Authentification",
            [
                "Connexion",
                "Créer un compte",
                "Vérifier l'email",
            ],
            key="aurum_auth_mode",
        )

        notice = st.session_state.pop(
            "aurum_auth_notice",
            None,
        )

        if notice:
            st.success(notice)

        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------

        if mode == "Connexion":

            st.markdown("### Welcome Back")

            from frontend.auth.remember_login import (
                enabled as remember_enabled_check,
                remembered_email,
                save_email,
                forget_credentials,
            )

            remember_enabled = remember_enabled_check()

            try:
                saved_email = remembered_email() if remember_enabled else None
            except Exception:
                saved_email = None

            if saved_email:
                st.caption(f"Compte memorise : {saved_email}")

                use_col, delete_col = st.columns([2, 1])

                if use_col.button(
                    "Utiliser mon compte enregistre",
                    key="aurum_use_remembered_login",
                    width="stretch",
                ):
                    try:
                        st.session_state["aurum_login_email"] = saved_email
                        st.session_state.pop("aurum_login_password", None)
                    except Exception:
                        st.error("Impossible de charger les identifiants Windows.")

                if delete_col.button(
                    "Effacer",
                    key="aurum_forget_remembered_login",
                    width="stretch",
                ):
                    try:
                        forget_credentials()
                        st.session_state.pop("aurum_login_email", None)
                        st.session_state.pop("aurum_login_password", None)
                        st.rerun()
                    except Exception:
                        st.error("Suppression des identifiants impossible.")

            with st.form("aurum_login_form"):

                email = st.text_input(
                    "Adresse email",
                    placeholder="votre.email@example.com",
                    key="aurum_login_email",
                )

                password = st.text_input(
                    "Mot de passe",
                    type="password",
                    key="aurum_login_password",
                )

                remember_choice = st.checkbox(
                    "Se souvenir de mon adresse email sur ce PC",
                    value=False,
                    disabled=not remember_enabled,
                )

                submitted = st.form_submit_button(
                    "Se connecter",
                    type="primary",
                    width="stretch",
                )

            if submitted:

                user = authenticate_user(
                    email=email,
                    password=password,
                )

                if user is None:
                    st.error(
                        "Connexion refusee. Verifiez vos identifiants "
                        "et la confirmation de votre adresse email."
                    )

                else:
                    if remember_enabled and remember_choice:
                        try:
                            save_email(user["email"])
                        except Exception:
                            st.session_state["aurum_remember_save_failed"] = True

                    st.session_state["aurum_clear_login_inputs"] = True
                    start_local_session(user)
                    st.rerun()

            st.button(
                "Continuer avec Google — bientôt disponible",
                disabled=True,
                width="stretch",
            )

        # ----------------------------------------------------
        # SIGN UP
        # ----------------------------------------------------

        elif mode == "Créer un compte":

            st.markdown("### Create Your Account")

            with st.form("aurum_signup_form"):

                display_name = st.text_input(
                    "Nom complet",
                    placeholder="Votre nom",
                )

                email = st.text_input(
                    "Adresse email",
                    placeholder="votre.email@example.com",
                )

                password = st.text_input(
                    "Mot de passe (12 caractères minimum)",
                    type="password",
                )

                confirmation = st.text_input(
                    "Confirmer le mot de passe",
                    type="password",
                )

                submitted = st.form_submit_button(
                    "Créer mon compte",
                    type="primary",
                    width="stretch",
                )

            if submitted:

                if password != confirmation:
                    st.error(
                        "Les mots de passe ne correspondent pas."
                    )

                else:

                    try:
                        register_and_send_otp(
                            display_name=display_name,
                            email=email,
                            password=password,
                        )

                    except AccountAlreadyExistsError:
                        st.error(
                            "Cette adresse est déjà enregistrée."
                        )

                    except RegistrationDeliveryError as exc:
                        st.warning(str(exc))

                    except ValueError as exc:
                        st.error(str(exc))

                    else:
                        st.session_state[
                            "aurum_pending_email"
                        ] = email.strip()

                        st.session_state[
                            "aurum_auth_notice"
                        ] = (
                            "Compte créé ! Consultez votre email "
                            "pour obtenir le code de vérification."
                        )

                        st.session_state[
                            "aurum_next_auth_mode"
                        ] = "Vérifier l'email"

                        st.rerun()

        # ----------------------------------------------------
        # EMAIL VERIFICATION
        # ----------------------------------------------------

        elif mode == "Vérifier l'email":

            st.markdown("### Gmail Verification")

            st.caption(
                "Entrez le code à 6 chiffres reçu par email."
            )

            with st.form("aurum_verify_form"):

                email = st.text_input(
                    "Adresse email",
                    value=st.session_state[
                        "aurum_pending_email"
                    ],
                )

                otp = st.text_input(
                    "Code OTP",
                    max_chars=6,
                    placeholder="000000",
                )

                submitted = st.form_submit_button(
                    "Vérifier mon compte",
                    type="primary",
                    width="stretch",
                )

            if submitted:

                verified = confirm_email_registration(
                    email=email,
                    submitted_code=otp,
                )

                if verified:

                    st.session_state[
                        "aurum_auth_notice"
                    ] = (
                        "Adresse email vérifiée ! "
                        "Vous pouvez vous connecter."
                    )

                    st.session_state[
                        "aurum_next_auth_mode"
                    ] = "Connexion"

                    st.rerun()

                else:
                    st.error(
                        "Code invalide, expiré ou bloqué."
                    )

            st.markdown("#### Renvoyer un code")

            with st.form("aurum_resend_form"):

                resend_email = st.text_input(
                    "Email du compte",
                    value=st.session_state[
                        "aurum_pending_email"
                    ],
                    key="aurum_resend_email",
                )

                resend_password = st.text_input(
                    "Mot de passe du compte",
                    type="password",
                    key="aurum_resend_password",
                )

                resend = st.form_submit_button(
                    "Renvoyer le code",
                    width="stretch",
                )

            if resend:

                try:
                    resend_verification_otp(
                        email=resend_email,
                        password=resend_password,
                    )

                except (
                    AccountVerificationError,
                    RegistrationDeliveryError,
                    OTPRequestError,
                    ValueError,
                ) as exc:
                    st.error(str(exc))

                else:
                    st.success(
                        "Un nouveau code a été envoyé."
                    )

    st.markdown(
        """
        <div class="aurum-footer">
            🔒 Authentication · SQLite · Argon2id · Gmail OTP
            <br><br>
            Academic Research Only · No Live Trading
        </div>
        """,
        unsafe_allow_html=True,
    )

"""AURUM-AI V5 / Research Center / Library — read-only catalog."""
from __future__ import annotations

from pathlib import Path
import json
import streamlit as st
import pandas as pd

from frontend.v5_pages._common import demand_session
from frontend.services.research_library import (
    catalog_summary,
    filter_library,
    format_size,
    list_library,
    resolve_library_item,
    sha256_file,
)

# Keep V5's authentication / idle-lock behavior intact.
demand_session()
REPO_ROOT = Path(__file__).resolve().parents[2]

st.markdown("## 📚 Research Library")
st.caption("AURUM-AI V5 · XOM · Documents, rapports, expériences et artefacts de recherche")
st.info(
    "Consultation locale uniquement. Aucun fichier source n'est modifié, "
    "aucun modèle n'est entraîné, aucun ordre n'est exécuté."
)

@st.cache_data(show_spinner=False, ttl=120)
def _load_catalog(root: str):
    return list_library(Path(root))

try:
    records = _load_catalog(str(REPO_ROOT))
except (OSError, RuntimeError, ValueError) as exc:
    st.error(f"Inventaire indisponible : {exc}")
    st.stop()

summary = catalog_summary(records)
a, b, c, d = st.columns(4)
a.metric("Fichiers indexés", len(records))
b.metric("Expériences IA", summary.get("Expériences IA", 0))
c.metric("Rapports", summary.get("Rapports", 0))
d.metric("Catégories", len(summary))

st.caption(
    "Le catalogue est limité aux dossiers de recherche autorisés ; "
    "les fichiers d'authentification et le dataset RAW ne sont pas parcourus. "
    "Actualisation du catalogue toutes les 2 minutes ou manuellement."
)
if st.button("↻ Actualiser l'inventaire"):
    _load_catalog.clear()
    st.rerun()

if not records:
    st.warning("Aucun document trouvé dans les dossiers autorisés de ce projet.")
    st.stop()

st.markdown("#### Explorer les ressources")
search_col, cat_col, ext_col = st.columns([2, 1, 1])
search = search_col.text_input("Rechercher", placeholder="Nom, chemin, horizon, modèle...")
categories = ["Toutes"] + sorted(summary)
selected_category = cat_col.selectbox("Catégorie", categories)
extensions = ["Toutes"] + sorted({item.extension for item in records})
selected_extension = ext_col.selectbox("Format", extensions)
filtered = filter_library(records, search, selected_category, selected_extension)
st.caption(f"{len(filtered)} résultat(s) sur {len(records)} fichiers indexés")

if not filtered:
    st.warning("Aucun résultat avec ces filtres.")
    st.stop()

# The table contains metadata only. Do not load file content here.
st.dataframe(
    pd.DataFrame([
        {
            "Catégorie": item.category,
            "Nom": item.name,
            "Chemin relatif": item.relative_path,
            "Format": item.extension,
            "Taille": format_size(item.size_bytes),
            "Modification (UTC)": item.modified_at_utc,
        }
        for item in filtered
    ]),
    hide_index=True,
    width="stretch",
    height=340,
)

st.markdown("#### Inspecter une ressource")
paths = [item.relative_path for item in filtered]
selected_path = st.selectbox("Fichier", paths, format_func=lambda p: p)
selected = next(item for item in filtered if item.relative_path == selected_path)
try:
    path = resolve_library_item(REPO_ROOT, selected)
except (ValueError, OSError) as exc:
    st.error(f"Ressource inaccessible : {exc}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.write(f"**Chemin :** `{selected.relative_path}`")
    st.write(f"**Catégorie :** {selected.category} · **Format :** `{selected.extension}`")
with right:
    st.write(f"**Taille :** {format_size(selected.size_bytes)}")
    st.write(f"**Modifié :** {selected.modified_at_utc}")

if st.button("Calculer SHA-256 (fichier sélectionné)"):
    try:
        with st.spinner("Calcul de l'empreinte locale en lecture seule..."):
            st.code(sha256_file(path), language="text")
    except OSError as exc:
        st.error(f"Lecture impossible : {exc}")

PREVIEW_LIMIT = 1_000_000
DOWNLOAD_LIMIT = 20 * 1024 * 1024

st.markdown("##### Aperçu")
try:
    if selected.extension == ".json" and selected.size_bytes <= PREVIEW_LIMIT:
        with path.open("r", encoding="utf-8-sig") as handle:
            st.json(json.load(handle))
    elif selected.extension in (".md", ".txt") and selected.size_bytes <= PREVIEW_LIMIT:
        with path.open("r", encoding="utf-8-sig") as handle:
            st.code(handle.read(150_000), language="text")
    elif selected.extension == ".csv" and selected.size_bytes <= 10 * 1024 * 1024:
        st.dataframe(pd.read_csv(path, nrows=25), hide_index=True, width="stretch")
        st.caption("Aperçu limité aux 25 premières lignes : aucun fichier n'est modifié.")
    elif selected.extension == ".parquet":
        st.caption("Jeu Parquet conservé sans lecture complète ; utilisez les pages Data Quality / Feature Engineering pour l'analyser.")
    elif selected.extension == ".npz":
        st.caption("Fenêtres et prédictions NPZ référencées sans chargement de tableaux en mémoire.")
    else:
        st.caption("Aperçu textuel non disponible pour ce format ou cette taille. Métadonnées disponibles ci-dessus.")
except (ValueError, UnicodeError, OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.warning(f"Aperçu indisponible : {exc}")

# Intentionally disallow potentially executable/pickle/model artifacts in downloads.
SAFE_DOWNLOADS = {".json", ".csv", ".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".svg"}
if selected.extension in SAFE_DOWNLOADS and selected.size_bytes <= DOWNLOAD_LIMIT:
    try:
        st.download_button(
            "↓ Télécharger une copie (lecture seule)",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream",
        )
    except OSError as exc:
        st.warning(f"Téléchargement indisponible : {exc}")
else:
    st.caption("Téléchargement depuis cette page désactivé pour les artefacts volumineux ou les formats modèles/binaires.")

st.caption(
    "Limites : inventaire des fichiers existants uniquement, pas une preuve de validité scientifique. "
    "Les artefacts Validation ne sont pas des résultats Test indépendants. "
    "L'empreinte SHA-256 locale n'établit pas à elle seule une certification externe."
)

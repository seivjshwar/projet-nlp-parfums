"""
Interface Streamlit - Recommandation de parfums
Lancer depuis la racine du projet :  streamlit run app.py
Dépendances :  pip install streamlit plotly

Contrat d'interface :
  - app.py produit `user_input` (Format 1) -> lu par nlp_engine.py / scoring.py
  - scoring.py produit `results` (Format 2) -> lu par app.py / visualizations.py
"""
import os
import random

import streamlit as st

# ----------------------------------------------------------------------
# BRANCHEMENT DES MODULES DU GROUPE  (adapter les noms ici si besoin)
# ----------------------------------------------------------------------
try:
    # HYPOTHESE : src/scoring.py expose  get_recommendations(user_input) -> results
    from src.scoring import get_recommendations
    SCORING_DISPONIBLE = True
except ImportError:
    SCORING_DISPONIBLE = False

from src.visualizations import plot_family_radar, plot_top3_breakdown

# ----------------------------------------------------------------------
# Constantes du contrat
# ----------------------------------------------------------------------
FAMILLES = [
    "Woody", "Citrus & Fresh", "Aromatic", "Floral",
    "Fruity", "Amber & Spicy", "Sweet & Gourmand", "Leather & Powdery",
]
GENRES = {"Tous": "all", "Femme": "female", "Homme": "male", "Unisexe": "unisex"}
# A VERIFIER avec le groupe : la 3e tranche est une hypothèse
BUDGETS = ["€ (< 60€)", "€€ (60-150€)", "€€€ (> 150€)"]
MOMENTS = ["Day", "Evening / Night", "Day & Night"]
SAISONS = ["Spring", "Summer", "Autumn", "Winter"]

# ----------------------------------------------------------------------
# Thème : crème, café, caramel
# ----------------------------------------------------------------------
BROWN_DARK, BROWN, CARAMEL, SAND, COPPER = "#3E2A1E", "#6F4A35", "#B08B5E", "#E3D5BE", "#9C5B3F"
TEXTE = "#4A3426"
BARRES_COULEURS = [BROWN_DARK, CARAMEL, "#D9C7A8", COPPER]   # semantic, family, occasion, intensity
BARRES_TEXTE = ["#FFFFFF", "#FFFFFF", BROWN_DARK, "#FFFFFF"]

CONFIG_TOML = """[theme]
base = "light"
primaryColor = "#6F4A35"
backgroundColor = "#FAF6EE"
secondaryBackgroundColor = "#F1E8D8"
textColor = "#4A3426"
font = "sans serif"
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --creme: #FAF6EE; --carte: #FFFDF8; --texte: #4A3426; --muted: #8B6F5A;
  --brown-dark: #3E2A1E; --brown: #6F4A35; --caramel: #B08B5E; --sand: #E3D5BE;
}

/* ---------- Base : force le mode clair ---------- */
:root, .stApp { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--creme) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stHeader"] button, [data-testid="stHeader"] svg { color: var(--brown) !important; }
.block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* ---------- Typographie ---------- */
.stApp p, .stApp label, .stApp li, .stApp small, input, textarea, [data-baseweb="select"] * { font-family: 'Inter', sans-serif; }
.stApp p, .stApp label, .stApp li, .stApp small,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * { color: var(--texte) !important; }
[data-testid="stWidgetLabel"] p { font-weight: 500; font-size: .92rem; }
[data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; color: var(--brown-dark) !important; letter-spacing: .3px; }
h1 { font-size: 3.4rem !important; text-align: center; font-weight: 600 !important; padding-bottom: 0 !important; }
h2 { font-size: 2.2rem !important; }
h3 { font-size: 1.6rem !important; }
.eyebrow { text-align: center; letter-spacing: .35em; font-size: .72rem; text-transform: uppercase;
           color: var(--caramel); font-family: 'Inter', sans-serif; font-weight: 500; margin-top: .5rem; }
.sous-titre { text-align: center; color: var(--muted); font-family: 'Cormorant Garamond', Georgia, serif;
              font-size: 1.35rem; font-style: italic; margin: .2rem 0 1.8rem; }

/* ---------- Bandeau image ---------- */
[data-testid="stImage"] img { width: 100%; max-height: 280px; object-fit: cover; border-radius: 18px;
                              box-shadow: 0 12px 32px rgba(62,42,30,.18); margin-bottom: 1.8rem; }

/* ---------- Formulaire & cartes ---------- */
[data-testid="stForm"] { background: var(--carte); border: 1px solid var(--sand); border-radius: 18px;
                         padding: 2rem 2rem 1.5rem; box-shadow: 0 8px 28px rgba(62,42,30,.06); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--sand) !important; border-radius: 18px;
                         background: var(--carte); box-shadow: 0 8px 28px rgba(62,42,30,.06); }
[data-testid="stExpander"] { border-color: var(--sand) !important; background: var(--carte); border-radius: 12px; }
hr { border-color: var(--sand) !important; }

/* ---------- Champs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  background-color: #FFFFFF !important; border-color: var(--sand) !important; border-radius: 10px !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="base-input"] input {
  background-color: #FFFFFF !important; color: var(--texte) !important; -webkit-text-fill-color: var(--texte) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--caramel) !important; box-shadow: 0 0 0 1px var(--caramel) !important; }
[data-baseweb="select"] * { color: var(--texte); }
[data-baseweb="select"] svg { fill: var(--brown) !important; }
::placeholder { color: #A8977F !important; -webkit-text-fill-color: #A8977F !important; opacity: 1 !important; }

/* ---------- Menus déroulants ---------- */
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--carte) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--texte) !important; }
[data-baseweb="popover"] li:hover { background: var(--creme) !important; }

/* ---------- Balises (tags) ---------- */
[data-baseweb="tag"] { background-color: var(--brown) !important; border-radius: 999px !important; }
[data-baseweb="tag"] span { color: var(--creme) !important; }
[data-baseweb="tag"] svg { fill: var(--creme) !important; color: var(--creme) !important; }

/* ---------- Boutons radio ---------- */
label[data-baseweb="radio"] > div:first-child { background-color: var(--carte) !important; border: 1.5px solid var(--caramel) !important; }
label[data-baseweb="radio"]:has(input:checked) > div:first-child { background-color: var(--brown) !important; border-color: var(--brown) !important; }

/* ---------- Curseurs ---------- */
[data-testid="stSlider"] [role="slider"] { background-color: var(--brown) !important; box-shadow: none !important; }
[data-testid="stSliderThumbValue"] { color: var(--brown) !important; font-weight: 600; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* ---------- Alertes ---------- */
[data-testid="stAlert"] { background: #F6EBD3 !important; border: 1px solid var(--sand); border-left: 4px solid var(--caramel);
                          border-radius: 10px; }
[data-testid="stAlert"] *, [data-testid="stAlert"] code { color: var(--texte) !important; background: transparent !important; }

/* ---------- Métriques ---------- */
[data-testid="stMetricValue"] { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 1.7rem; }

/* ---------- Bouton principal ---------- */
[data-testid="stFormSubmitButton"] button {
  background: var(--brown) !important; border: 1px solid var(--brown) !important; border-radius: 999px;
  padding: .7rem 2.6rem; box-shadow: 0 8px 20px rgba(111,74,53,.30); transition: all .2s ease; }
[data-testid="stFormSubmitButton"] button p { color: var(--creme) !important; letter-spacing: .12em;
  text-transform: uppercase; font-size: .78rem; font-weight: 600; }
[data-testid="stFormSubmitButton"] button:hover { background: var(--brown-dark) !important; border-color: var(--brown-dark) !important;
  transform: translateY(-1px); }
</style>
"""


def habiller_figure(fig, kind: str):
    """Adapte une figure plotly au thème sans toucher à visualizations.py."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXTE),
        title=dict(font=dict(family="Cormorant Garamond, Georgia, serif", size=20, color=BROWN_DARK)),
    )
    if kind == "radar":
        fig.update_traces(line=dict(color=BROWN, width=2.5), fillcolor="rgba(176,139,94,0.30)")
        fig.update_layout(polar=dict(
            bgcolor="rgba(255,253,248,0.7)",
            radialaxis=dict(gridcolor=SAND, linecolor=SAND),
            angularaxis=dict(gridcolor=SAND, linecolor=SAND)))
    else:
        for trace, couleur, txt in zip(fig.data, BARRES_COULEURS, BARRES_TEXTE):
            trace.marker.color = couleur
            trace.textfont = dict(color=txt)
        fig.update_xaxes(gridcolor=SAND, zerolinecolor=SAND)
    return fig


def afficher_plotly(fig):
    """Compatibilité entre versions de Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def verifier_theme():
    """Le thème natif (curseurs, radios, barres) vient de .streamlit/config.toml.
    Si le fichier est absent du dossier de lancement, on le crée."""
    if st.get_option("theme.primaryColor") == BROWN:
        return
    chemin = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(chemin):
        os.makedirs(".streamlit", exist_ok=True)
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(CONFIG_TOML)
        st.info("Thème installé dans `.streamlit/config.toml`. Relancez l'application (Ctrl+C puis `streamlit run app.py`).")
    else:
        st.info("Le fichier `.streamlit/config.toml` existe mais n'est pas à jour : remplacez-le puis relancez l'application.")


# ----------------------------------------------------------------------
# Fallback : résultats factices si scoring.py n'est pas encore là
# ----------------------------------------------------------------------
def mock_recommendations(user_input: dict) -> dict:
    rnd = random.Random(str(user_input))
    family_scores = {f: round(rnd.uniform(0.3, 0.8), 2) for f in FAMILLES}
    top_3 = []
    for i in range(3):
        b = {
            "semantic": round(rnd.uniform(0.25, 0.46), 2),
            "family": round(rnd.uniform(0.05, 0.15), 2),
            "occasion": round(rnd.uniform(0.05, 0.12), 2),
            "intensity": round(rnd.uniform(0.05, 0.11), 2),
        }
        score = round(sum(b.values()), 2)
        top_3.append({
            "PerfumeID": f"P{i:03d}", "Name": f"Parfum exemple {i + 1}", "Brand": "Marque test",
            "Family": rnd.choice(FAMILLES), "Budget": rnd.choice(BUDGETS),
            "TopNotes": "bergamote, citron", "HeartNotes": "lavande, géranium",
            "BaseNotes": "cèdre, musc", "Occasion": "Day (Spring, Summer)",
            "Longevity": round(rnd.uniform(2, 5), 1), "Sillage": round(rnd.uniform(1, 4), 1),
            "Rating": round(rnd.uniform(3.8, 4.6), 2), "final_score": score,
            "breakdown": b,
            "explanation": f"Recommandé à {score:.0%} : (données factices, scoring.py non branché).",
        })
    top_3.sort(key=lambda p: p["final_score"], reverse=True)
    return {"family_scores": family_scores, "top_3": top_3}


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------
def afficher_carte(rang: int, p: dict):
    with st.container(border=True):
        st.subheader(f"N° {rang} · {p['Name']}")
        st.caption(f"{p['Brand']} · {p['Family']} · {p['Budget']}")
        st.progress(min(float(p["final_score"]), 1.0), text=f"Score final : {p['final_score']:.0%}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Tête**  \n{p['TopNotes']}")
        c2.markdown(f"**Cœur**  \n{p['HeartNotes']}")
        c3.markdown(f"**Fond**  \n{p['BaseNotes']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tenue", f"{p['Longevity']}/5")
        m2.metric("Sillage", f"{p['Sillage']}/4")
        m3.metric("Note", f"{p['Rating']}/5")
        m4.metric("Occasion", p["Occasion"])
        st.info(p["explanation"])


def afficher_resultats(results: dict):
    st.divider()
    st.header("Vos recommandations")

    for i, p in enumerate(results["top_3"], start=1):
        afficher_carte(i, p)

    st.subheader("Pourquoi ces parfums ?")
    col_radar, col_bar = st.columns(2)
    with col_radar:
        afficher_plotly(habiller_figure(plot_family_radar(results["family_scores"]), "radar"))
    with col_bar:
        afficher_plotly(habiller_figure(plot_top3_breakdown(results["top_3"]), "bars"))


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Parfum Match", page_icon="🌸", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
verifier_theme()

st.markdown('<div class="eyebrow">Sélection olfactive personnalisée</div>', unsafe_allow_html=True)
st.title("Parfum Match")
st.markdown('<div class="sous-titre">Décrivez le parfum idéal, nous trouvons le vôtre.</div>', unsafe_allow_html=True)

if os.path.exists("assets/banner.png"):
    st.image("assets/banner.png")

if not SCORING_DISPONIBLE:
    st.warning("`src/scoring.py` introuvable : résultats factices affichés pour tester l'interface.")

with st.form("formulaire"):
    st.subheader("1. Dites-nous ce que vous cherchez")
    description_libre = st.text_area(
        "Description libre",
        placeholder="Un parfum frais, élégant et légèrement boisé pour le bureau",
    )
    ambiance_emotionnelle = st.text_input(
        "Ambiance émotionnelle", placeholder="Rassurant, propre, qui donne confiance"
    )

    st.subheader("2. Vos préférences")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        families = st.multiselect("Familles olfactives (optionnel)", FAMILLES)
        gender_label = st.radio("Pour qui ?", list(GENRES), horizontal=True)
        budget = st.multiselect("Budget", BUDGETS, default=BUDGETS)
    with c2:
        moment = st.radio("Moment de la journée", MOMENTS, horizontal=True)
        seasons = st.multiselect("Saisons", SAISONS, default=SAISONS)
        longevity = st.slider("Tenue souhaitée", 1, 5, 3)
        sillage = st.slider("Sillage souhaité", 1, 4, 2)

    submit = st.form_submit_button("Trouver mon parfum", type="primary")

if submit:
    # --- Format 1 : user_input -------------------------------------------------
    user_input = {
        "description_libre": description_libre.strip(),
        "ambiance_emotionnelle": ambiance_emotionnelle.strip(),
        "families": families,
        "gender": GENRES[gender_label],
        "budget": budget,
        "moment": moment,
        "seasons": seasons,
        "longevity": longevity,
        "sillage": sillage,
    }

    if not user_input["description_libre"] and not user_input["ambiance_emotionnelle"] and not families:
        st.error("Remplissez au moins une description, une ambiance ou une famille.")
    elif not budget or not seasons:
        st.error("Sélectionnez au moins un budget et une saison.")
    else:
        with st.spinner("Analyse en cours..."):
            fn = get_recommendations if SCORING_DISPONIBLE else mock_recommendations
            st.session_state["user_input"] = user_input
            st.session_state["results"] = fn(user_input)  # --- Format 2 : results

if "results" in st.session_state:
    afficher_resultats(st.session_state["results"])
    with st.expander("Voir le user_input envoyé (debug)"):
        st.json(st.session_state["user_input"])
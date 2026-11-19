"""Interface web : questionnaire → analyse sémantique → recommandation de parfums.

Lancement : streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))  # rend `src` importable

from src.config import RESPONSES_DIR
from src.embeddings import get_embedder
from src.explain import explain_perfume
from src.preprocessing import split_sentences
from src.recommender import frequent_notes, load_reference, recommend
from src.scoring import UserProfile

st.set_page_config(page_title="Recommandation de parfums", page_icon="🌸", layout="wide")
st.title("Trouvez votre parfum")
st.caption("Prototype pédagogique d'analyse sémantique : les recommandations reposent sur la similarité "
           "entre vos réponses et les descriptions des parfums.")

perfumes, families = load_reference()
all_notes = frequent_notes(perfumes)


@st.cache_resource
def embedder():
    return get_embedder()


# ---------------------------------------------------------------- Questionnaire
with st.form("questionnaire"):
    st.subheader("1. Décrivez le parfum de vos rêves")
    q_desc = st.text_area(
        "Quelle ambiance, quelles émotions doit-il évoquer ?",
        placeholder="Ex. : un parfum élégant et frais, légèrement boisé, avec des notes florales subtiles.",
    )
    q_notes = st.text_area(
        "Quelles odeurs ou ingrédients aimez-vous ? (ou un parfum que vous portez déjà)",
        placeholder="Ex. : j'adore l'odeur de la vanille et du bois de santal.",
    )
    q_occasion = st.text_area(
        "Dans quel contexte allez-vous le porter ?",
        placeholder="Ex. : en journée au bureau, plutôt au printemps et en été.",
    )

    st.subheader("2. Vos familles olfactives (1 = je n'aime pas, 5 = j'adore)")
    cols = st.columns(4)
    likert = {fam: cols[i % 4].slider(fam, 1, 5, 3) for i, fam in enumerate(families["Family"])}

    st.subheader("3. Questions guidées")
    c1, c2, c3 = st.columns(3)
    seasons = c1.multiselect("Saison(s)", ["printemps", "été", "automne", "hiver"])
    moment = c1.radio("Moment", ["jour", "soir", "les deux"], index=2, horizontal=True)
    intensity = c2.select_slider("Intensité / sillage souhaité", options=[1, 2, 3, 4, 5], value=3,
                                 help="Relative au catalogue : 1 = très discret, 5 = très puissant")
    gender = c3.radio("Pour qui ?", ["Peu importe", "Plutôt féminin", "Plutôt masculin"], horizontal=False)
    avoided = st.multiselect("Notes à éviter (cases à cocher)", all_notes)

    submitted = st.form_submit_button("Trouver mes parfums")

if not submitted:
    st.stop()

# Les réponses guidées sont converties en phrases pour enrichir le contexte
occasion_text = q_occasion
if seasons:
    occasion_text += f" Plutôt en {', '.join(seasons)}."
if moment != "les deux":
    occasion_text += " Plutôt en journée." if moment == "jour" else " Plutôt le soir."
profile = UserProfile(
    texts={
        "description": split_sentences(q_desc),
        "notes": split_sentences(q_notes),
        "occasion": split_sentences(occasion_text),
    },
    family_likert=likert,
    intensity=intensity,
    gender={"Plutôt féminin": "female", "Plutôt masculin": "male"}.get(gender),
    avoided_notes=avoided,
)
if not profile.all_sentences():
    st.warning("Merci de répondre au moins à une question en texte libre.")
    st.stop()

# Sauvegarde structurée des réponses (JSON)
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
(RESPONSES_DIR / f"response_{stamp}.json").write_text(json.dumps({
    "description": q_desc, "notes": q_notes, "occasion": q_occasion, "seasons": seasons,
    "moment": moment, "likert": likert, "intensity": intensity, "gender": gender,
    "avoided_notes": avoided,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# -------------------------------------------------------------------- Analyse
with st.spinner("Analyse sémantique en cours…"):
    top, result = recommend(profile, embedder())

# ------------------------------------------------------------------- Résultats
col1, col2 = st.columns(2)
with col1:
    st.subheader("Votre profil olfactif")
    names, values = list(result.family_affinity), list(result.family_affinity.values())
    radar = go.Figure(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself"))
    radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False)
    st.plotly_chart(radar, use_container_width=True)
with col2:
    st.subheader("Contribution des critères (top 3)")
    crit_cols = [c for c in top.columns if c.startswith("S_")]
    long = top.melt(id_vars="Name", value_vars=crit_cols, var_name="critère", value_name="similarité")
    long["critère"] = long["critère"].str.removeprefix("S_")
    st.plotly_chart(px.bar(long, x="Name", y="similarité", color="critère", barmode="group"),
                    use_container_width=True)

st.subheader("Nos 3 recommandations")
for _, row in top.iterrows():
    st.markdown(explain_perfume(row))

with st.expander("Classement complet"):
    st.dataframe(result.ranking.drop(columns=[c for c in result.ranking.columns if c.startswith("best_")]))

with st.expander("Carte de similarité : vos phrases × 10 meilleurs parfums"):
    top10 = result.ranking.head(10)["PerfumeID"]
    cols = perfumes.index[perfumes["PerfumeID"].isin(top10)]
    for crit, sim in result.criterion_sims.items():
        st.markdown(f"**Critère : {crit}**")
        st.plotly_chart(px.imshow(sim[:, cols], x=perfumes.loc[cols, "Name"], y=profile.texts[crit],
                                  color_continuous_scale="Purples", aspect="auto", zmin=0, zmax=1),
                        use_container_width=True)

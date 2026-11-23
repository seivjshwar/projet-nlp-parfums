"""
nlp_engine.py — Moteur d'analyse sémantique (Membre 1)

Rôle : transformer les fiches parfums et les réponses de l'utilisateur en
embeddings (vecteurs SBERT), puis mesurer leur proximité (similarité cosinus).

Choix de conception (option B) :
Le modèle ne lit que 128 tokens par texte. La colonne SearchText (~230 tokens)
était donc toujours tronquée et la description perdue. On encode à la place
TROIS textes courts par parfum, chacun lu en entier :
    - "profile"     : accords + pyramide olfactive (~65 tokens)
    - "occasion"    : moment de la journée + saisons (~10 tokens)
    - "description" : texte éditorial nettoyé, découpé en phrases

Corrections après tests (étape 5b) :
    - le nom de la famille n'est plus dans "profile" : il biaisait les résultats
      (« agrumes » -> famille « Fruity ») et la famille est mesurée à part ;
    - pas de phrase de repli pour les parfums sans texte éditorial : elle
      ressemblait aux requêtes courtes et gonflait leur score. Ces parfums ont
      sim_description = NaN (pas d'information), à ignorer par le scoring ;
    - correction du « hubness » (méthode CSLS) : certaines phrases génériques
      sont proches de presque toutes les requêtes. On retire à chaque phrase sa
      « popularité » (similarité moyenne avec ses HUB_K plus proches voisines).
"""
import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ Paramètres
ROOT = Path(__file__).resolve().parent.parent          # racine du projet
PERFUMES_CSV = ROOT / "data" / "reference" / "perfumes_200_clean.csv"
EMBEDDINGS_FILE = ROOT / "data" / "embeddings.npz"     # cache des embeddings
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MAX_DESC_SENTENCES = 6    # phrases de description gardées par parfum
MIN_SENTENCE_CHARS = 25   # en dessous, la phrase est trop courte pour être utile
HUB_K = 10                # voisins utilisés pour mesurer la popularité d'une phrase
EMB_VERSION = "v4"        # à changer si la façon de calculer les embeddings change

# Description rédigée de chaque famille olfactive (les 8 familles du CSV).
# On n'utilise PAS la moyenne des parfums de la famille : certaines étiquettes du
# CSV sont discutables et fausseraient le « centre » de la famille.
FAMILY_DESCRIPTIONS = {
    "Amber & Spicy": "Warm amber, incense, resins and hot spices such as cinnamon, clove and "
                     "cardamom. Sensual, enveloping, mysterious and intense, for evenings and winter.",
    "Aromatic": "Aromatic herbs: lavender, sage, rosemary, mint, geranium and pepper. Fresh, "
                "clean, energetic and herbal, a classic fougere.",
    "Citrus & Fresh": "Citrus and freshness: bergamot, lemon, mandarin, grapefruit, neroli and "
                      "marine notes. Light, sparkling, zesty and invigorating, for summer days.",
    "Floral": "Flowers: rose, jasmine, orange blossom, tuberose, peony and lily of the valley. "
              "Feminine, romantic, soft, bright and elegant.",
    "Fruity": "Juicy fruits: peach, pear, raspberry, black currant, apple and red berries. "
              "Cheerful, playful, youthful and sweet.",
    "Leather & Powdery": "Leather, suede, iris, violet and powdery musks. Soft and powdery or "
                         "dark and leathery, chic, cosy, skin-like.",
    "Sweet & Gourmand": "Sweet edible notes: vanilla, praline, caramel, coffee, chocolate, honey "
                        "and tonka bean. Comforting, addictive and indulgent.",
    "Woody": "Woods: sandalwood, cedar, vetiver, patchouli and oud. Warm, dry, earthy, natural, "
             "elegant and reassuring.",
}

# Phrases de description Fragrantica à retirer : elles ne décrivent pas
# l'ambiance (date, parfumeur, marque) ou répètent les notes, déjà dans "profile".
DROP_SENTENCE_RE = re.compile(
    r"fragrance for (?:women|men)|was launched in|launched|noses? behind|created by|"
    r"top notes? (?:is|are)|middle notes? (?:is|are)|base notes? (?:is|are)|"
    r"the fragrance features|\bml\b|\boz\b|advertis|available|campaign|bottle|"
    r"winner of award|collection|perfumer|edition|announced|notes? include|introduced",
    re.IGNORECASE,
)


# ------------------------------------------------------ 1. Chargement des données
def load_perfumes(path: Path = PERFUMES_CSV) -> pd.DataFrame:
    """Charge le référentiel et remplace les cases vides par des chaînes vides."""
    df = pd.read_csv(path)
    text_cols = ["Family", "Keywords", "OlfactoryNotes", "BestMoment", "BestSeasons", "Description"]
    df[text_cols] = df[text_cols].fillna("")
    return df


# ------------------------------------------------- 2. Construction des 3 textes
def build_profile_text(row: pd.Series) -> str:
    """Profil olfactif : accords + notes (tête, cœur, fond), sans le nom de famille."""
    notes = row["OlfactoryNotes"].replace(" | ", ". ")
    return f"Accords: {row['Keywords']}. {notes}."


def build_occasion_text(row: pd.Series) -> str:
    """Occasion : moment de la journée + saisons, en phrase naturelle."""
    moments = {
        "Day": "during the day",
        "Evening / Night": "in the evening and at night",
        "Day & Night": "both day and night",
    }
    moment = moments.get(row["BestMoment"], "day or night")
    seasons = row["BestSeasons"].lower().replace(", ", " and ") or "any season"
    return f"Best worn {moment}, in {seasons}."


def split_description(row: pd.Series) -> list[str]:
    """Description : découpée en phrases, nettoyée des phrases inutiles.

    On garde au plus MAX_DESC_SENTENCES phrases. Si rien ne reste (description
    purement « technique »), la liste est vide : le parfum n'aura pas de
    score de description (NaN) plutôt qu'un score artificiel.
    """
    text = re.sub(r"\s+", " ", row["Description"]).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)            # découpage en phrases
    kept = [s for s in sentences
            if len(s) >= MIN_SENTENCE_CHARS and not DROP_SENTENCE_RE.search(s)]
    return kept[:MAX_DESC_SENTENCES]


def build_perfume_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute au DataFrame les 3 textes qui seront encodés."""
    df = df.copy()
    df["profile_text"] = df.apply(build_profile_text, axis=1)
    df["occasion_text"] = df.apply(build_occasion_text, axis=1)
    df["description_sentences"] = df.apply(split_description, axis=1)
    return df


# ------------------------------------------------ 3. Encodage et sauvegarde
@lru_cache(maxsize=1)
def get_model():
    """Charge le modèle SBERT une seule fois (les appels suivants réutilisent le même)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def encode(texts: list[str]) -> np.ndarray:
    """Textes -> matrice (nb_textes, 384), vecteurs normalisés (norme = 1).

    Avec des vecteurs normalisés, la similarité cosinus devient un simple
    produit scalaire : cos(a, b) = a · b.
    """
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64,
                              convert_to_numpy=True, show_progress_bar=False)


def fingerprint(df: pd.DataFrame) -> str:
    """Empreinte des textes + du modèle : change si le CSV ou le modèle change."""
    parts = [MODEL_NAME, EMB_VERSION] + list(FAMILY_DESCRIPTIONS.values()) + df["profile_text"].tolist() + df["occasion_text"].tolist() \
        + [" ".join(s) for s in df["description_sentences"]]
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def sentence_popularity(desc: np.ndarray, k: int = HUB_K) -> np.ndarray:
    """Popularité (« hubness ») de chaque phrase : similarité moyenne avec ses
    k plus proches voisines du catalogue. Élevée = phrase passe-partout."""
    sims = desc @ desc.T
    np.fill_diagonal(sims, -1.0)                        # on ignore la phrase elle-même
    top_k = np.sort(sims, axis=1)[:, -k:]               # les k plus grandes similarités
    return top_k.mean(axis=1)


def build_embeddings(df: pd.DataFrame) -> dict:
    """Encode les 3 textes de chaque parfum."""
    # Description : toutes les phrases à la suite + le parfum propriétaire de chacune
    sentences = [s for sents in df["description_sentences"] for s in sents]
    owner = np.repeat(np.arange(len(df)), df["description_sentences"].str.len())
    desc = encode(sentences)
    return {
        "perfume_ids": df["PerfumeID"].to_numpy(dtype=str),
        "profile": encode(df["profile_text"].tolist()),        # (200, 384)
        "occasion": encode(df["occasion_text"].tolist()),      # (200, 384)
        "desc": desc,                                          # (nb_phrases, 384)
        "desc_owner": owner,                                   # (nb_phrases,)
        "desc_hub": sentence_popularity(desc),                 # (nb_phrases,)
        "family_names": np.array(list(FAMILY_DESCRIPTIONS), dtype=str),
        "family": encode(list(FAMILY_DESCRIPTIONS.values())),  # (8, 384)
        "fingerprint": np.array(fingerprint(df)),
    }


def load_or_build_embeddings(df: pd.DataFrame, path: Path = EMBEDDINGS_FILE) -> dict:
    """Relit les embeddings sauvegardés s'ils sont à jour, sinon les recalcule."""
    if path.exists():
        saved = dict(np.load(path))
        if "fingerprint" in saved and str(saved["fingerprint"]) == fingerprint(df):
            return saved
        print("CSV ou modèle modifié : recalcul des embeddings…")
    emb = build_embeddings(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **emb)
    return emb


# ------------------------------------------ 4. Préparation du texte utilisateur
# Noms des champs du contrat d'interface (user_input, produit par app.py).
# Les noms anglais sont acceptés aussi, au cas où le contrat évolue.
DESCRIPTION_KEYS = ("description_libre", "free_description")
MOOD_KEYS = ("ambiance_emotionnelle", "emotional_mood")

MOMENT_EN = {
    "Day": "during the day",
    "Evening / Night": "in the evening and at night",
    "Day & Night": "both day and night",
}


def _get(user_input: dict, keys: tuple) -> str:
    """Valeur du premier champ présent parmi `keys` (chaîne vide sinon)."""
    for k in keys:
        if user_input.get(k):
            return str(user_input[k])
    return ""


def clean_user_text(text: str) -> list[str]:
    """Nettoie une réponse libre et la découpe en phrases.

    Nettoyage volontairement léger : on garde mots vides, casse et ponctuation,
    car SBERT s'en sert pour comprendre le contexte.
    """
    text = unicodedata.normalize("NFC", text or "")       # accents homogènes
    text = re.sub(r"[^\w\s.,;:!?'’()&/-]", " ", text)    # retire emojis et symboles
    text = re.sub(r"\s+", " ", text).strip()                # espaces multiples
    sentences = re.split(r"(?<=[.!?;])\s+|\s*\n+\s*", text)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


def build_occasion_query(user_input: dict) -> str:
    """moment + seasons (choix du questionnaire) -> phrase anglaise comparable."""
    moment = MOMENT_EN.get(user_input.get("moment", ""), "day or night")
    seasons = [s.lower() for s in user_input.get("seasons", [])]
    return f"Best worn {moment}, in {' and '.join(seasons) or 'any season'}."


# ----------------------------------------- 5. Similarités utilisateur × parfums
PERFUME_SIM_COLUMNS = ["PerfumeID", "sim_profile", "sim_description", "sim_description_raw",
                       "sim_occasion", "best_user_phrase", "best_perfume_phrase"]

def compute_perfume_similarities(user_input: dict, df: pd.DataFrame, emb: dict) -> pd.DataFrame:
    """Similarités cosinus entre les réponses de l'utilisateur et chaque parfum.

    Renvoie un DataFrame (une ligne par parfum, même ordre que df) :
        PerfumeID, sim_profile, sim_description, sim_description_raw, sim_occasion,
        best_user_phrase, best_perfume_phrase   (meilleure paire pour la description)
    sim_description est corrigée du hubness (CSLS) ; sim_description_raw est le
    cosinus brut de la même paire, pour la transparence.
    Une colonne vaut NaN si l'utilisateur n'a pas répondu au champ correspondant
    ou si le parfum n'a pas de texte éditorial.
    """
    desc_sents = clean_user_text(_get(user_input, DESCRIPTION_KEYS))
    mood_sents = clean_user_text(_get(user_input, MOOD_KEYS))
    out = pd.DataFrame({"PerfumeID": emb["perfume_ids"]})

    # a) Description libre vs profil olfactif : meilleure phrase utilisateur
    if desc_sents:
        u = encode(desc_sents)                                  # (n_phrases_user, 384)
        out["sim_profile"] = (u @ emb["profile"].T).max(axis=0)  # max sur les phrases user
    else:
        out["sim_profile"] = np.nan

    # b) Description libre + ambiance vs phrases de description du parfum
    user_sents = desc_sents + mood_sents
    if user_sents:
        u = encode(user_sents)
        raw = u @ emb["desc"].T                                # (n_user, nb_phrases_parfums)
        hub = emb["desc_hub"]
        sims = raw - (hub - hub.mean())                        # correction CSLS
        best_sim, best_raw, best_user, best_perf = [], [], [], []
        for i in range(len(df)):
            cols = np.flatnonzero(emb["desc_owner"] == i)       # phrases du parfum i
            if len(cols) == 0:                                  # pas de texte éditorial
                best_sim.append(np.nan); best_raw.append(np.nan)
                best_user.append(""); best_perf.append("")
                continue
            block = sims[:, cols]                               # (n_user, nb_phrases_i)
            r, c = np.unravel_index(block.argmax(), block.shape)  # meilleure paire
            best_sim.append(block[r, c])
            best_raw.append(raw[r, cols[c]])
            best_user.append(user_sents[r])
            best_perf.append(df["description_sentences"].iloc[i][c])
        out["sim_description"] = best_sim
        out["sim_description_raw"] = best_raw
        out["best_user_phrase"] = best_user
        out["best_perfume_phrase"] = best_perf
    else:
        out["sim_description"] = out["sim_description_raw"] = np.nan
        out["best_user_phrase"] = out["best_perfume_phrase"] = ""

    # c) Moment + saisons vs texte d'occasion
    if user_input.get("moment") or user_input.get("seasons"):
        q = encode([build_occasion_query(user_input)])          # (1, 384)
        out["sim_occasion"] = (q @ emb["occasion"].T)[0]
    else:
        out["sim_occasion"] = np.nan

    # Ordre des colonnes figé : c'est le format promis au Membre 2
    return out[PERFUME_SIM_COLUMNS]


# ------------------------------------- 6. Similarités utilisateur × familles
FAMILY_TOP_K = 5   # nb de meilleurs parfums pris en compte par famille


def compute_family_similarities(user_input: dict, df: pd.DataFrame, emb: dict,
                                perfume_sims: pd.DataFrame | None = None,
                                method: str = "catalogue") -> dict:
    """Similarité entre l'utilisateur et chacune des 8 familles olfactives (radar).

    method="catalogue" (par défaut) : la famille est vue à travers ses parfums.
        score = moyenne des FAMILY_TOP_K meilleurs sim_profile de la famille.
        Cohérent avec les recommandations, et s'appuie sur ~25 parfums réels.
    method="description" : comparaison directe avec FAMILY_DESCRIPTIONS
        (une phrase rédigée par famille). Gardée pour comparaison : moins fiable.

    perfume_sims : résultat de compute_perfume_similarities, pour ne pas le
    recalculer s'il l'a déjà été. Renvoie {famille: score} ; NaN si pas de texte.
    """
    names = [str(n) for n in emb["family_names"]]

    if method == "description":
        sentences = clean_user_text(_get(user_input, DESCRIPTION_KEYS)) \
            + clean_user_text(_get(user_input, MOOD_KEYS))
        if not sentences:
            return {n: float("nan") for n in names}
        sims = (encode(sentences) @ emb["family"].T).max(axis=0)
        return {n: float(v) for n, v in zip(names, sims)}

    if perfume_sims is None:
        perfume_sims = compute_perfume_similarities(user_input, df, emb)
    profile = pd.Series(perfume_sims["sim_profile"].to_numpy(), index=df["Family"].to_numpy())
    if profile.isna().all():
        return {n: float("nan") for n in names}
    return {n: float(profile[profile.index == n].nlargest(FAMILY_TOP_K).mean()) for n in names}


def family_label_agreement(df: pd.DataFrame, emb: dict) -> pd.DataFrame:
    """Contrôle qualité du CSV : pour chaque parfum, la famille dont la
    description est la plus proche de son profil olfactif, comparée à
    l'étiquette « Family » du CSV."""
    names = np.array([str(n) for n in emb["family_names"]])
    sims = emb["profile"] @ emb["family"].T                    # (200, 8)
    return pd.DataFrame({
        "Name": df["Name"],
        "Family": df["Family"],
        "predicted_family": names[sims.argmax(axis=1)],
        "agree": df["Family"].to_numpy() == names[sims.argmax(axis=1)],
    })


# ------------------------------------------- 7. Point d'entrée pour le scoring
@lru_cache(maxsize=1)
def get_catalogue() -> tuple:
    """Charge le CSV, construit les textes et les embeddings UNE fois par session."""
    df = build_perfume_texts(load_perfumes())
    return df, load_or_build_embeddings(df)


def rescale(values: dict) -> dict:
    """Remet des scores à l'échelle 0–1 (meilleur = 1, moins bon = 0) pour l'affichage."""
    v = pd.Series(values, dtype=float)
    span = v.max() - v.min()
    if v.isna().all() or span == 0:
        return {k: float("nan") for k in values}
    return ((v - v.min()) / span).round(4).to_dict()


def analyze(user_input: dict) -> dict:
    """Point d'entrée unique du moteur NLP (utilisé par scoring.py).

    Entrée : user_input au format du contrat d'interface (produit par app.py).
    Sortie : voir docs/nlp_engine.md
        {
          "perfume_similarities": DataFrame (200 lignes, ordre du CSV) :
              PerfumeID, sim_profile, sim_description, sim_description_raw,
              sim_occasion, best_user_phrase, best_perfume_phrase
          "family_similarities": {famille: score brut}          (8 familles)
          "family_similarities_scaled": {famille: score 0–1}    (pour le radar)
          "user_sentences": phrases utilisateur analysées
          "occasion_query": phrase construite à partir de moment + seasons
          "model": nom du modèle SBERT
        }
    """
    df, emb = get_catalogue()
    perfume_sims = compute_perfume_similarities(user_input, df, emb)
    family_sims = compute_family_similarities(user_input, df, emb, perfume_sims=perfume_sims)
    has_occasion = bool(user_input.get("moment") or user_input.get("seasons"))
    return {
        "perfume_similarities": perfume_sims,
        "family_similarities": family_sims,
        "family_similarities_scaled": rescale(family_sims),
        "user_sentences": clean_user_text(_get(user_input, DESCRIPTION_KEYS))
                          + clean_user_text(_get(user_input, MOOD_KEYS)),
        "occasion_query": build_occasion_query(user_input) if has_occasion else "",
        "model": MODEL_NAME,
    }


# ------------------------------------------------------ Démonstration / contrôle
if __name__ == "__main__":
    df = build_perfume_texts(load_perfumes())
    emb = load_or_build_embeddings(df)

    # Exemple de user_input, au format du contrat d'interface
    user_input = {
        "description_libre": "Un parfum frais, élégant et légèrement boisé pour le bureau",
        "ambiance_emotionnelle": "Rassurant, propre, qui donne confiance",
        "families": ["Woody", "Citrus & Fresh"],
        "gender": "male",
        "budget": ["€ (< 60€)", "€€ (60-150€)"],
        "moment": "Day",
        "seasons": ["Spring", "Summer"],
        "longevity": 4,
        "sillage": 2,
    }

    print("Phrases utilisateur :", clean_user_text(user_input["description_libre"]),
          clean_user_text(user_input["ambiance_emotionnelle"]))
    print("Requête occasion    :", build_occasion_query(user_input))

    sims = compute_perfume_similarities(user_input, df, emb)
    sims = sims.join(df[["Name", "Family", "BestMoment", "BestSeasons"]])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    for col in ["sim_profile", "sim_description", "sim_occasion"]:
        print(f"\n=== Top 5 selon {col} ===")
        print(sims.nlargest(5, col)[["Name", "Family", col]].round(3).to_string(index=False))

    print("\n=== Plage des similarités ===")
    print(sims[["sim_profile", "sim_description", "sim_occasion"]].describe().loc[["min", "50%", "max"]].round(3))

    best = sims.loc[sims["sim_description"].idxmax()]
    print(f"\nParfums sans texte éditorial (sim_description = NaN) : "
          f"{sims['sim_description'].isna().sum()} / {len(sims)}")
    print(f"\nMeilleure correspondance de description : {best['Name']}")
    print(f"  utilisateur : « {best['best_user_phrase']} »")
    print(f"  parfum      : « {best['best_perfume_phrase']} »")

    # --- Étape 6 : familles
    print("\n=== Similarité par famille (radar) ===")
    fam = compute_family_similarities(user_input, df, emb, perfume_sims=sims)
    for name, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {v:.3f}  " + "█" * int(max(v, 0) * 40))

    agreement = family_label_agreement(df, emb)
    print(f"\n=== Accord entre l'étiquette du CSV et la famille prédite : "
          f"{agreement['agree'].mean():.0%} ===")
    print(pd.crosstab(agreement["Family"], agreement["predicted_family"]).to_string())
    print("\nExemples de désaccords :")
    print(agreement[~agreement["agree"]].head(8).to_string(index=False))

    # --- Étape 7 : point d'entrée unique
    print("\n=== analyze(user_input) : ce que reçoit le Membre 2 ===")
    nlp = analyze(user_input)
    for key, value in nlp.items():
        kind = f"DataFrame {value.shape}" if isinstance(value, pd.DataFrame) else repr(value)[:90]
        print(f"  {key:28s} {kind}")

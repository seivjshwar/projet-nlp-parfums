"""Paramètres centralisés du projet (chemins, modèle, poids, seuils)."""
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RESPONSES_DIR = DATA_DIR / "responses"

PERFUMES_CSV = REFERENCE_DIR / "perfumes.csv"
FAMILIES_CSV = REFERENCE_DIR / "families.csv"

# --- Modèle d'embeddings -----------------------------------------------------
# Modèle SBERT multilingue : les utilisateurs écrivent en français.
# À comparer dans un notebook avec all-MiniLM-L6-v2 (anglais) et une baseline
# Word2Vec/fastText pour justifier le choix.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Scoring -----------------------------------------------------------------
# Poids W_a de chaque critère dans le score d'affinité d'un parfum.
CRITERIA_WEIGHTS = {
    "description": 0.35,   # ce que l'utilisateur décrit (ambiance, émotions)
    "notes": 0.30,         # notes / ingrédients aimés
    "occasion": 0.15,      # contexte d'usage, saison
    "family": 0.20,        # affinité avec la famille olfactive du parfum
}

# Part de l'auto-évaluation Likert dans l'affinité d'une famille.
LIKERT_WEIGHT = 0.4

# Contraintes
INTENSITY_PENALTY = 0.08     # pénalité par point d'écart d'intensité (1–5)
AVOIDED_NOTE_PENALTY = 0.25  # pénalité par note à éviter présente
PRICE_ORDER = {"€": 1, "€€": 2, "€€€": 3}

# Seuils d'interprétation d'une similarité (explications + couverture).
HIGH_MATCH = 0.55
MEDIUM_MATCH = 0.35

TOP_K = 3

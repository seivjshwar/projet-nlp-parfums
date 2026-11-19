"""Tests du scoring avec un faux embedder (pas de téléchargement de modèle)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.build_dataset import clean_description, extract_family, strip_weights
from src.preprocessing import split_sentences
from src.recommender import frequent_notes, load_reference
from src.scoring import UserProfile, run_scoring


class KeywordEmbedder:
    """Embedder factice : un axe par mot-clé (FR et EN), suffisant pour tester la logique."""

    VOCAB = [("vanill",), ("coffee", "café"), ("sweet", "sucré"), ("citrus", "agrume"),
             ("fresh", "frais"), ("wood", "bois"), ("rose",), ("night", "soir"), ("summer", "été")]

    def encode(self, texts):
        vecs = np.array([[sum(t.lower().count(w) for w in group) for group in self.VOCAB] for t in texts],
                        dtype=float)
        return vecs + 1e-3  # évite les vecteurs nuls


def profile(**kwargs):
    base = dict(texts={"description": ["Un parfum sucré de soirée, au café et à la vanille."]})
    base.update(kwargs)
    return UserProfile(**base)


def score_of(result, name):
    return result.ranking.set_index("Name").loc[name, "score"]


# ------------------------------------------------------------ préparation données
def test_strip_weights():
    assert strip_weights("Vanilla:100|Musk:22|Amber:1", 2) == "Vanilla; Musk"
    assert strip_weights(float("nan")) == ""


def test_extract_family():
    desc = "Angel by Mugler is a Oriental Vanilla fragrance for women. Angel was launched in 1992."
    assert extract_family(desc, "sweet:100") == ("Gourmand", "Oriental Vanilla")
    assert extract_family("No family here.", "woody:100|citrus:50")[0] == "Boisé"


def test_clean_description_drops_template():
    desc = ("X by Y is a Floral fragrance for women. X was launched in 2001. Top notes are Lemon and Apple. "
            "A breezy, sparkling scent that evokes the Sicilian summer. Available as 50 ml.")
    assert clean_description(desc) == "A breezy, sparkling scent that evokes the Sicilian summer."


def test_split_sentences():
    assert split_sentences("J'aime la vanille. Et le bois !") == ["J'aime la vanille.", "Et le bois !"]


# ------------------------------------------------------------------- référentiel
def test_reference_is_consistent():
    perfumes, families = load_reference()
    assert len(perfumes) >= 50                       # minimum du sujet
    assert perfumes["PerfumeID"].is_unique
    assert set(perfumes["Family"]) <= set(families["Family"])
    assert perfumes["Intensity"].between(1, 5).all()
    assert len(frequent_notes(perfumes)) > 20


# ----------------------------------------------------------------------- scoring
def test_ranking_and_scores_in_range():
    perfumes, families = load_reference()
    r = run_scoring(profile(), perfumes, families, KeywordEmbedder()).ranking
    assert r.iloc[0]["Family"] in {"Gourmand", "Ambré"}
    assert r["score"].between(0, 1).all() and r["coverage"].between(0, 1).all()
    assert list(r["score"]) == sorted(r["score"], reverse=True)


def test_avoided_note_penalises():
    perfumes, families = load_reference()
    base = run_scoring(profile(), perfumes, families, KeywordEmbedder())
    avoid = run_scoring(profile(avoided_notes=["Vanilla"]), perfumes, families, KeywordEmbedder())
    assert score_of(avoid, "Angel") < score_of(base, "Angel")


def test_gender_penalty_spares_unisex():
    perfumes, families = load_reference()
    base = run_scoring(profile(), perfumes, families, KeywordEmbedder())
    fem = run_scoring(profile(gender="female"), perfumes, families, KeywordEmbedder())
    assert score_of(fem, "Sauvage") < score_of(base, "Sauvage")                   # masculin
    assert score_of(fem, "Tobacco Vanille") == score_of(base, "Tobacco Vanille")   # mixte


def test_likert_changes_family_affinity():
    perfumes, families = load_reference()
    low = run_scoring(profile(family_likert={"Gourmand": 1}), perfumes, families, KeywordEmbedder())
    high = run_scoring(profile(family_likert={"Gourmand": 5}), perfumes, families, KeywordEmbedder())
    assert high.family_affinity["Gourmand"] > low.family_affinity["Gourmand"]

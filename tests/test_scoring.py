"""Tests du scoring avec un faux embedder (pas de téléchargement de modèle)."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import split_sentences
from src.recommender import load_reference
from src.scoring import UserProfile, run_scoring


class KeywordEmbedder:
    """Embedder factice : un axe par mot-clé, suffisant pour tester la logique."""

    VOCAB = ["vanille", "café", "marin", "mer", "bois", "santal", "rose", "jasmin", "frais", "soirée", "été"]

    def encode(self, texts):
        vecs = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype=float)
        return vecs + 1e-3  # évite les vecteurs nuls


def profile(**kwargs):
    base = dict(texts={"description": ["Un parfum de soirée au café et à la vanille."]})
    base.update(kwargs)
    return UserProfile(**base)


def test_split_sentences():
    assert split_sentences("J'aime la vanille. Et le bois !") == ["J'aime la vanille.", "Et le bois !"]
    assert split_sentences("   ") == []


def test_reference_is_consistent():
    perfumes, families = load_reference()
    assert len(perfumes) >= 10
    assert set(perfumes["Family"]) <= set(families["Family"])


def test_ranking_and_scores_in_range():
    perfumes, families = load_reference()
    result = run_scoring(profile(), perfumes, families, KeywordEmbedder())
    r = result.ranking
    assert r.iloc[0]["Name"] == "Black Opium"
    assert r["score"].between(0, 1).all() and r["coverage"].between(0, 1).all()
    assert list(r["score"]) == sorted(r["score"], reverse=True)


def test_budget_filter():
    perfumes, families = load_reference()
    result = run_scoring(profile(max_price="€€"), perfumes, families, KeywordEmbedder())
    assert set(result.ranking["PriceRange"]) <= {"€", "€€"}


def test_avoided_note_penalises():
    perfumes, families = load_reference()
    base = run_scoring(profile(), perfumes, families, KeywordEmbedder()).ranking
    avoid = run_scoring(profile(avoided_notes=["café"]), perfumes, families, KeywordEmbedder()).ranking
    score = lambda r: r.set_index("Name").loc["Black Opium", "score"]
    assert score(avoid) < score(base)


def test_likert_changes_family_affinity():
    perfumes, families = load_reference()
    low = run_scoring(profile(family_likert={"Gourmand": 1}), perfumes, families, KeywordEmbedder())
    high = run_scoring(profile(family_likert={"Gourmand": 5}), perfumes, families, KeywordEmbedder())
    assert high.family_affinity["Gourmand"] > low.family_affinity["Gourmand"]

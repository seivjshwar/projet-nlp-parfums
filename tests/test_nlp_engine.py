"""Tests du moteur NLP (Membre 1).

Un faux modèle remplace SBERT : les tests sont rapides, ne téléchargent rien,
et vérifient le FORMAT de sortie promis au Membre 2 (pas la qualité sémantique).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import nlp_engine  # noqa: E402


class FakeModel:
    """Vecteur déterministe par texte, normalisé (comme SBERT)."""

    def encode(self, texts, **kwargs):
        rng = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=384) for t in texts]
        v = np.array(rng)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(nlp_engine, "get_model", lambda: FakeModel())
    monkeypatch.setattr(nlp_engine, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    nlp_engine.get_catalogue.cache_clear()
    yield
    nlp_engine.get_catalogue.cache_clear()


FULL_INPUT = {
    "description_libre": "Un parfum frais, élégant et légèrement boisé pour le bureau",
    "ambiance_emotionnelle": "Rassurant, propre, qui donne confiance",
    "families": ["Woody"], "gender": "male", "budget": [],
    "moment": "Day", "seasons": ["Spring", "Summer"], "longevity": 4, "sillage": 2,
}
COLUMNS = ["PerfumeID", "sim_profile", "sim_description", "sim_description_raw",
           "sim_occasion", "best_user_phrase", "best_perfume_phrase"]


def test_analyze_output_format():
    nlp = nlp_engine.analyze(FULL_INPUT)
    sims = nlp["perfume_similarities"]
    df, _ = nlp_engine.get_catalogue()
    assert list(sims.columns) == COLUMNS
    assert len(sims) == len(df)
    assert (sims["PerfumeID"].to_numpy() == df["PerfumeID"].to_numpy()).all()   # même ordre que le CSV
    assert set(nlp["family_similarities"]) == set(df["Family"])                 # 8 familles
    scaled = pd.Series(nlp["family_similarities_scaled"])
    assert scaled.min() == 0 and scaled.max() == 1
    assert nlp["occasion_query"] == "Best worn during the day, in spring and summer."


def test_similarities_are_cosines():
    sims = nlp_engine.analyze(FULL_INPUT)["perfume_similarities"]
    for col in ["sim_profile", "sim_description_raw", "sim_occasion"]:
        assert sims[col].dropna().between(-1.0001, 1.0001).all()


def test_empty_fields_give_nan():
    nlp = nlp_engine.analyze({"description_libre": "", "moment": "", "seasons": []})
    sims = nlp["perfume_similarities"]
    assert sims[["sim_profile", "sim_description", "sim_occasion"]].isna().all().all()
    assert all(np.isnan(v) for v in nlp["family_similarities"].values())


def test_perfumes_without_editorial_text_have_nan_description():
    df, _ = nlp_engine.get_catalogue()
    sims = nlp_engine.analyze(FULL_INPUT)["perfume_similarities"]
    no_text = df["description_sentences"].str.len() == 0
    assert sims.loc[no_text, "sim_description"].isna().all()
    assert sims.loc[~no_text, "sim_description"].notna().all()


def test_english_contract_keys_are_accepted():
    nlp = nlp_engine.analyze({"free_description": "Fresh citrus", "emotional_mood": "Happy"})
    assert nlp["user_sentences"] == ["Fresh citrus", "Happy"]


def test_clean_user_text():
    assert nlp_engine.clean_user_text("  Frais 🍋 et   léger.  Pour l'été !") == ["Frais et léger.", "Pour l'été !"]
    assert nlp_engine.clean_user_text("") == []

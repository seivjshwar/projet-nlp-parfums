"""Chargement du référentiel et construction des recommandations."""
import pandas as pd

from src.config import FAMILIES_CSV, PERFUMES_CSV, TOP_K
from src.embeddings import Embedder
from src.scoring import ScoringResult, UserProfile, run_scoring


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge parfums et familles, et vérifie la cohérence des familles."""
    perfumes = pd.read_csv(PERFUMES_CSV).fillna("")
    families = pd.read_csv(FAMILIES_CSV)
    unknown = set(perfumes["Family"]) - set(families["Family"])
    if unknown:
        raise ValueError(f"Familles inconnues dans perfumes.csv : {unknown}")
    return perfumes, families


def recommend(profile: UserProfile, embedder: Embedder, top_k: int = TOP_K) -> tuple[pd.DataFrame, ScoringResult]:
    """Renvoie le top k des parfums et le résultat complet du scoring."""
    perfumes, families = load_reference()
    result = run_scoring(profile, perfumes, families, embedder)
    return result.ranking.head(top_k), result

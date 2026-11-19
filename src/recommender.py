"""Chargement du référentiel et construction des recommandations."""
from collections import Counter

import pandas as pd

from src.config import FAMILIES_CSV, PERFUMES_CSV, TOP_K
from src.embeddings import Embedder
from src.scoring import ScoringResult, UserProfile, run_scoring

NOTE_COLUMNS = ("TopNotes", "HeartNotes", "BaseNotes")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge parfums et familles, et vérifie la cohérence des familles."""
    perfumes = pd.read_csv(PERFUMES_CSV)
    for col in (*NOTE_COLUMNS, "Accords", "Seasons", "Moment", "Description", "SubFamily"):
        perfumes[col] = perfumes[col].fillna("")
    families = pd.read_csv(FAMILIES_CSV)
    unknown = set(perfumes["Family"]) - set(families["Family"])
    if unknown:
        raise ValueError(f"Familles inconnues dans perfumes.csv : {unknown}")
    return perfumes, families


def frequent_notes(perfumes: pd.DataFrame, k: int = 80) -> list[str]:
    """Les k notes les plus fréquentes du catalogue (pour la question « notes à éviter »)."""
    counts = Counter(
        note.strip() for col in NOTE_COLUMNS for cell in perfumes[col] for note in cell.split(";") if note.strip()
    )
    return sorted(n for n, _ in counts.most_common(k))


def recommend(profile: UserProfile, embedder: Embedder, top_k: int = TOP_K) -> tuple[pd.DataFrame, ScoringResult]:
    """Renvoie le top k des parfums et le résultat complet du scoring."""
    perfumes, families = load_reference()
    result = run_scoring(profile, perfumes, families, embedder)
    return result.ranking.head(top_k), result

"""Mécanisme de scoring sémantique pour la recommandation de parfums.

Pour chaque parfum p et chaque critère a ∈ {description, notes, occasion} :
    S_a(p) = max_{phrase u de l'utilisateur pour ce critère} cos(emb(u), emb(texte_a(p)))
Le max signifie : « au moins une phrase de l'utilisateur correspond bien ».

Affinité famille :
    A(f) = (1 - λ) · max_u cos(emb(u), emb(description_famille f)) + λ · Likert_f normalisé
    S_family(p) = A(famille de p)

Score d'affinité sémantique (moyenne pondérée sur les critères renseignés) :
    Affinité(p) = Σ_a W_a · S_a(p) / Σ_a W_a

Score final = Affinité(p) × facteur de contraintes (intensité, notes à éviter, genre).

Score de couverture des préférences = part pondérée des critères pour
lesquels S_a(p) ≥ MEDIUM_MATCH (« ce parfum couvre 3 de vos 4 critères »).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.embeddings import Embedder, cosine_matrix

TEXT_CRITERIA = ("description", "notes", "occasion")


@dataclass
class UserProfile:
    """Réponses du questionnaire, déjà découpées en phrases."""
    texts: dict[str, list[str]]                       # critère -> phrases
    family_likert: dict[str, int] = field(default_factory=dict)   # famille -> 1..5
    intensity: int | None = None                      # 1..5
    gender: str | None = None                         # "female", "male" ou None (peu importe)
    avoided_notes: list[str] = field(default_factory=list)

    def all_sentences(self) -> list[str]:
        return [s for sents in self.texts.values() for s in sents]


@dataclass
class ScoringResult:
    ranking: pd.DataFrame                  # un parfum par ligne, trié
    family_affinity: dict[str, float]
    criterion_sims: dict[str, np.ndarray] = field(repr=False)  # critère -> (phrases, parfums)


# --------------------------------------------------------------- Textes parfums
def _join(cell: str) -> str:
    return str(cell).replace(";", ",") if isinstance(cell, str) else ""


def perfume_texts(perfumes: pd.DataFrame) -> dict[str, list[str]]:
    """Construit, pour chaque critère, le texte encodé de chaque parfum."""
    notes = [
        f"Main accords: {_join(r.Accords)}. Top notes: {_join(r.TopNotes)}. "
        f"Heart notes: {_join(r.HeartNotes)}. Base notes: {_join(r.BaseNotes)}."
        for r in perfumes.itertuples()
    ]
    moments = {"day": "during the day", "night": "in the evening and at night"}
    occasion = [
        f"Best worn in {_join(r.Seasons).replace(',', ' and') or 'any season'}, "
        f"{' and '.join(moments[m.strip()] for m in str(r.Moment).split(';') if m.strip() in moments) or 'day or night'}."
        for r in perfumes.itertuples()
    ]
    return {"description": perfumes["Description"].tolist(), "notes": notes, "occasion": occasion}


# ------------------------------------------------------------- Étapes du score
def family_affinity(profile: UserProfile, families: pd.DataFrame, embedder: Embedder) -> dict[str, float]:
    """Affinité de l'utilisateur pour chaque famille olfactive (0–1)."""
    sentences = profile.all_sentences()
    sims = cosine_matrix(embedder.encode(sentences), embedder.encode(families["Description"].tolist()))
    semantic = np.clip(sims.max(axis=0), 0, 1)
    result = {}
    for fam, sem in zip(families["Family"], semantic):
        if fam in profile.family_likert:
            self_eval = (profile.family_likert[fam] - 1) / 4
            result[fam] = float((1 - config.LIKERT_WEIGHT) * sem + config.LIKERT_WEIGHT * self_eval)
        else:
            result[fam] = float(sem)
    return result


def constraint_factor(row: pd.Series, profile: UserProfile) -> tuple[float, list[str]]:
    """Facteur multiplicatif (0–1) lié aux contraintes souples, et leurs raisons."""
    factor, reasons = 1.0, []
    if profile.intensity is not None:
        gap = abs(int(row["Intensity"]) - profile.intensity)
        factor -= config.INTENSITY_PENALTY * gap
        if gap >= 2:
            reasons.append(f"intensité {row['Intensity']}/5 éloignée de votre souhait ({profile.intensity}/5)")
    all_notes = " ".join(str(row[c]) for c in ("TopNotes", "HeartNotes", "BaseNotes")).lower()
    for note in profile.avoided_notes:
        if note.lower() in all_notes:
            factor -= config.AVOIDED_NOTE_PENALTY
            reasons.append(f"contient {note}, que vous souhaitez éviter")
    if profile.gender and row["Gender"] not in ("unisex", profile.gender):
        factor -= config.GENDER_PENALTY
        reasons.append("pensé pour " + ("les hommes" if row["Gender"] == "male" else "les femmes"))
    return max(factor, 0.0), reasons


def run_scoring(
    profile: UserProfile,
    perfumes: pd.DataFrame,
    families: pd.DataFrame,
    embedder: Embedder,
) -> ScoringResult:
    """Point d'entrée : calcule le classement de tous les parfums pour un profil."""
    if not profile.all_sentences():
        raise ValueError("Aucune réponse en texte libre à analyser.")

    fam_aff = family_affinity(profile, families, embedder)
    texts = perfume_texts(perfumes)

    # Similarités par critère (seulement si l'utilisateur a répondu à ce critère)
    criterion_sims, scores, best_sentence = {}, {}, {}
    for crit in TEXT_CRITERIA:
        sents = profile.texts.get(crit) or []
        if not sents:
            continue
        sim = cosine_matrix(embedder.encode(sents), embedder.encode(texts[crit]))
        criterion_sims[crit] = sim
        scores[crit] = np.clip(sim.max(axis=0), 0, 1)
        best_sentence[crit] = [sents[i] for i in sim.argmax(axis=0)]
    scores["family"] = perfumes["Family"].map(fam_aff).fillna(0).to_numpy()

    weights = {c: config.CRITERIA_WEIGHTS[c] for c in scores}
    w_sum = sum(weights.values())

    ranking = perfumes[["PerfumeID", "Name", "Brand", "Family", "SubFamily", "Gender",
                        "Intensity", "Rating", "URL"]].copy()
    for crit, s in scores.items():
        ranking[f"S_{crit}"] = s
    ranking["affinity"] = sum(weights[c] * scores[c] for c in scores) / w_sum
    ranking["coverage"] = sum(weights[c] * (scores[c] >= config.MEDIUM_MATCH) for c in scores) / w_sum

    factors = [constraint_factor(row, profile) for _, row in perfumes.iterrows()]
    ranking["constraint_factor"] = [f for f, _ in factors]
    ranking["constraint_reasons"] = [r for _, r in factors]
    ranking["score"] = ranking["affinity"] * ranking["constraint_factor"]
    for crit, sents in best_sentence.items():
        ranking[f"best_{crit}"] = sents

    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    return ScoringResult(ranking, fam_aff, criterion_sims)

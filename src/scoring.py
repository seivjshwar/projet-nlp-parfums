"""Score relatif 0–100 à partir de la sortie de nlp_engine.analyze()."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping

import numpy as np
import pandas as pd

# Poids initiaux à valider sur des requêtes annotées. Les critères manquants
# ne participent pas au dénominateur, pour chaque parfum individuellement.
WEIGHTS = {
    "profile": .36, "description": .14, "occasion": .06,
    "family": .18, "context": .06, "performance": .10,
    "gender": .05, "budget": .05,
}

SEASONS = {
    "printemps": "spring", "spring": "spring", "ete": "summer",
    "summer": "summer", "automne": "autumn", "autumn": "autumn",
    "hiver": "winter", "winter": "winter",
}
MOMENTS = {
    "day": {"day"}, "jour": {"day"}, "journee": {"day"},
    "evening / night": {"night"}, "evening": {"night"}, "night": {"night"},
    "soir": {"night"}, "soiree": {"night"}, "nuit": {"night"},
    "day & night": {"day", "night"}, "jour et soir": {"day", "night"},
}
GENDERS = {
    "male": "male", "homme": "male", "female": "female", "femme": "female",
    "unisex": "unisex", "unisexe": "unisex", "mixte": "unisex",
    "all": "all", "any": "all", "tous": "all",
}
SIM_COLUMNS = (
    "PerfumeID", "sim_profile", "sim_description", "sim_description_raw",
    "sim_occasion", "best_user_phrase", "best_perfume_phrase",
)


def key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def items(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (tuple, list, set)):
        raise ValueError("Ce critère doit être une chaîne ou une liste.")
    return [str(v) for v in value if str(v).strip()]


def level(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Tenue et sillage doivent être entre 1 et 5.") from exc
    if not np.isfinite(number) or not 1 <= number <= 5:
        raise ValueError("Tenue et sillage doivent être entre 1 et 5.")
    return number


def budget_level(value: object) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value in ("1", "2", "3"):
        return int(value)
    if value.startswith("€€€"):
        return 3
    if value.startswith("€€"):
        return 2
    if value.startswith("€"):
        return 1
    return None


def scaled(series: pd.Series) -> pd.Series:
    """Min-max sur le catalogue entier ; NaN ou colonne constante = absent."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty or finite.max() == finite.min():
        return pd.Series(np.nan, index=series.index, dtype=float)
    return ((values - finite.min()) / (finite.max() - finite.min())).where(np.isfinite(values))


def selections(user_input: Mapping, catalogue: pd.DataFrame) -> dict:
    families = {key(f) for f in items(user_input.get("families"))}
    unknown = families - {key(f) for f in catalogue["Family"].dropna()}
    if unknown:
        raise ValueError(f"Famille inconnue : {sorted(unknown)}")
    seasons = set()
    for season in items(user_input.get("seasons")):
        if key(season) not in SEASONS:
            raise ValueError(f"Saison inconnue : {season}")
        seasons.add(SEASONS[key(season)])
    moments = set()
    for moment in items(user_input.get("moment")):
        if key(moment) not in MOMENTS:
            raise ValueError(f"Moment inconnu : {moment}")
        moments.update(MOMENTS[key(moment)])
    raw_gender = key(user_input.get("gender") or "all")
    if raw_gender not in GENDERS:
        raise ValueError(f"Genre inconnu : {user_input.get('gender')}")
    raw_budget = user_input.get("budget")
    if isinstance(raw_budget, (list, tuple, set)):
        budget = {budget_level(v) for v in raw_budget}
        if None in budget:
            raise ValueError("Catégorie de budget inconnue.")
    elif raw_budget is None or raw_budget == "":
        budget = set()
    else:  # ancienne interface : un chiffre désigne le plafond
        ceiling = budget_level(raw_budget)
        if ceiling is None:
            raise ValueError("Catégorie de budget inconnue.")
        budget = set(range(1, ceiling + 1))
    return {
        "families": families, "seasons": seasons, "moments": moments,
        "gender": GENDERS[raw_gender], "budget": budget,
        "longevity": level(user_input.get("longevity")),
        "sillage": level(user_input.get("sillage")),
        "text": str(user_input.get("description_libre") or user_input.get("free_description") or "").strip(),
        "mood": str(user_input.get("ambiance_emotionnelle") or user_input.get("emotional_mood") or "").strip(),
    }


def optional_number(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and np.isfinite(number) else None


def cosine_number(value: object) -> float | None:
    """Évite d'afficher 1.0000001 pour un cosinus égal à 1 à l'arrondi près."""
    number = optional_number(value)
    return float(np.clip(number, -1, 1)) if number is not None else None


def score_catalog(catalogue: pd.DataFrame, user_input: Mapping, nlp: Mapping) -> dict:
    """Joint par PerfumeID, applique les filtres et classe les parfums.

    Le score 0–100 est relatif à cette requête, pas une probabilité absolue.
    """
    if not isinstance(catalogue, pd.DataFrame) or catalogue.empty:
        raise ValueError("Catalogue vide ou invalide.")
    if not isinstance(user_input, Mapping) or not isinstance(nlp, Mapping):
        raise ValueError("user_input et nlp doivent être des dictionnaires.")
    sims = nlp.get("perfume_similarities")
    if not isinstance(sims, pd.DataFrame):
        raise ValueError("perfume_similarities doit être un DataFrame.")
    required = {"PerfumeID", "Name", "Brand", "Family"}
    if required - set(catalogue) or set(SIM_COLUMNS) - set(sims):
        raise ValueError("Colonnes catalogue ou NLP manquantes.")
    frame, sims = catalogue.reset_index(drop=True), sims.reset_index(drop=True)
    ids = frame["PerfumeID"].astype(str).tolist()
    if len(ids) != len(set(ids)) or sims["PerfumeID"].astype(str).tolist() != ids:
        raise ValueError("PerfumeID absents, doublons ou ordre différent entre catalogue et NLP.")
    selected = selections(user_input, frame)
    semantic = {
        name: scaled(sims[column]) for name, column in (
            ("profile", "sim_profile"), ("description", "sim_description"),
            ("occasion", "sim_occasion"),
        )
    }
    family_scores = nlp.get("family_similarities", {})
    ranking = []
    for idx, row in frame.iterrows():
        item_gender = GENDERS.get(key(row.get("Gender", "")), "")
        if selected["gender"] != "all":
            allowed = ({"unisex"} if selected["gender"] == "unisex"
                       else {"unisex", selected["gender"]})
            if item_gender not in allowed:
                continue
        item_budget = budget_level(row.get("Budget"))
        if selected["budget"] and item_budget not in selected["budget"]:
            continue
        component = {name: float(series.iloc[idx]) for name, series in semantic.items()
                     if pd.notna(series.iloc[idx])}
        family = str(row["Family"])
        if selected["families"]:
            component["family"] = float(key(family) in selected["families"])
        context, matches = [], {}
        if selected["moments"] and pd.notna(row.get("BestMoment")):
            possible = MOMENTS.get(key(row.get("BestMoment")))
            if possible:
                matches["moment"] = len(selected["moments"] & possible) / len(selected["moments"])
                context.append(matches["moment"])
        if selected["seasons"] and pd.notna(row.get("BestSeasons")):
            possible = {SEASONS[key(v)] for v in str(row["BestSeasons"]).split(",")
                        if key(v) in SEASONS}
            if possible:
                matches["season"] = len(selected["seasons"] & possible) / len(selected["seasons"])
                context.append(matches["season"])
        if context:
            component["context"] = float(np.mean(context))
        performance = []
        for wanted, field in ((selected["longevity"], "Longevity"),
                              (selected["sillage"], "Sillage")):
            if wanted is not None:
                try:
                    actual = level(row.get(field))
                except ValueError:
                    actual = None
                if actual is not None:
                    performance.append(1 - abs(wanted - actual) / 4)
        if performance:
            component["performance"] = float(np.mean(performance))
        if selected["gender"] != "all":
            component["gender"] = 1 if item_gender == selected["gender"] else .8
        if selected["budget"]:
            component["budget"] = 1
        if not component:
            raise ValueError("Aucun critère exploitable dans la requête.")
        denom = sum(WEIGHTS[name] for name in component)
        weights = {name: WEIGHTS[name] / denom for name in component}
        contributions = {name: 100 * weights[name] * value for name, value in component.items()}
        ranking.append({
            "_index": idx, "perfume_id": ids[idx], "name": str(row["Name"]),
            "brand": str(row["Brand"]), "family": family,
            "score": round(sum(contributions.values()), 2),
            "semantic_similarity": cosine_number(sims.at[idx, "sim_profile"]),
            "profile_similarity": cosine_number(sims.at[idx, "sim_profile"]),
            "description_similarity": optional_number(sims.at[idx, "sim_description"]),
            "description_similarity_raw": cosine_number(sims.at[idx, "sim_description_raw"]),
            "occasion_similarity": cosine_number(sims.at[idx, "sim_occasion"]),
            "family_similarity": optional_number(family_scores.get(family)),
            "matched_phrases": {
                "user": str(sims.at[idx, "best_user_phrase"])
                if pd.notna(sims.at[idx, "best_user_phrase"]) else "",
                "perfume": str(sims.at[idx, "best_perfume_phrase"])
                if pd.notna(sims.at[idx, "best_perfume_phrase"]) else "",
            },
            "context_matches": matches,
            "components": {k: round(v, 4) for k, v in component.items()},
            "contributions": {k: round(v, 2) for k, v in contributions.items()},
            "weights_used": {k: round(v, 4) for k, v in weights.items()},
        })
    ranking.sort(key=lambda r: (-r["score"], -r["components"].get("profile", -1), r["perfume_id"]))
    return {
        "profile_summary": {
            **{k: sorted(v) if isinstance(v, set) else v for k, v in selected.items()},
            "budget_max_category": max(selected["budget"]) if selected["budget"] else None,
        },
        "family_scores": {k: optional_number(v) for k, v in family_scores.items()},
        "family_similarities_scaled": {
            k: optional_number(v) for k, v in nlp.get("family_similarities_scaled", {}).items()
        },
        "ranking": ranking, "scored_count": len(ranking),
    }

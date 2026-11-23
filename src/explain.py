"""Explications du Top 3 fondées sur les données et les phrases du moteur NLP."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

if __package__:
    from .scoring import SEASONS, key
else:
    from scoring import SEASONS, key

FAMILY_FR = {
    "amber & spicy": "ambré et épicé", "aromatic": "aromatique",
    "citrus & fresh": "frais et hespéridé", "floral": "floral",
    "fruity": "fruité", "leather & powdery": "cuiré et poudré",
    "sweet & gourmand": "sucré et gourmand", "woody": "boisé",
}
NOTE_FR = {
    "woody": "des notes boisées", "fresh spicy": "des épices fraîches",
    "warm spicy": "des épices chaudes", "citrus": "des agrumes",
    "vanilla": "une note vanillée", "lavender": "de la lavande",
    "rose": "de la rose", "amber": "des notes ambrées",
    "aromatic": "des accords aromatiques", "fresh": "de la fraîcheur",
    "floral": "des notes florales", "fruity": "des notes fruitées",
    "aquatic": "un accord aquatique", "marine": "un accord marin",
    "green": "des notes vertes", "musky": "des notes musquées",
    "powdery": "des notes poudrées", "leather": "des notes de cuir",
}
ALIASES = {
    "woody": ("bois", "boise"), "citrus": ("agrume", "citron", "hesperide"),
    "fresh": ("frais", "fraicheur"), "floral": ("fleur", "floral"),
    "fruity": ("fruit", "fruite"), "vanilla": ("vanille",),
    "lavender": ("lavande",), "rose": ("rose",),
    "amber": ("ambre", "ambre"), "aromatic": ("aromatique",),
}
SEASONS_FR = {"spring": "au printemps", "summer": "en été",
              "autumn": "en automne", "winter": "en hiver"}
# La meilleure paire SBERT peut être une phrase générique sur la marque ou le
# parfumeur. On garde la paire pour l'inspection, mais on ne la cite pas au client.
EDITORIAL_RE = re.compile(
    r"\b(?:perfumer|designer|creator|created|launched|ingredient|knowledge|"
    r"experience|campaign|bottle|collection|award)\w*\b", re.I
)
SENSORY_RE = re.compile(
    r"\b(?:notes?|accords?|scent|aroma|smell|fresh|woody|wood|cedar|"
    r"citrus|lemon|orange|rose|jasmine|floral|flower|spicy|spice|"
    r"sweet|salty|vanilla|amber|musk|sea|water|warm|powdery|leather|"
    r"bois|frais|agrumes?|fleurs?|epice|vanille|ambre)\w*\b", re.I
)
PHRASE_FR = (
    (r"\b(?:sea water|marine|ocean|sea breeze)\b", "une facette marine"),
    (r"\b(?:sunny warmth|warmth on your skin)\b", "une sensation chaleureuse"),
    (r"\b(?:citrus|lemon|bergamot)\b", "une fraîcheur d'agrumes"),
    (r"\b(?:wood|cedar|sandalwood|vetiver)\b", "une touche boisée"),
    (r"\b(?:rose|jasmine|flowers?)\b", "des notes florales"),
    (r"\b(?:vanilla|praline|caramel)\b", "une douceur gourmande"),
)


def _terms(row: pd.Series) -> list[str]:
    terms = []
    for field in ("Keywords", "TopNotes", "HeartNotes", "BaseNotes"):
        if pd.notna(row.get(field)):
            terms.extend(part.strip() for part in str(row[field]).split(",") if part.strip())
    return list(dict.fromkeys(terms))


def _join(values: list[str]) -> str:
    return " et ".join(values) if len(values) < 3 else ", ".join(values[:-1]) + " et " + values[-1]


def _mentioned(term: str, user_words: str) -> bool:
    normalized = key(term)
    return (len(normalized) > 3 and normalized in user_words) or any(
        alias in user_words for alias in ALIASES.get(normalized, ())
    )


def _usable_phrase(phrase: str) -> bool:
    """Écarte les phrases sur la marque, le parfumeur ou l'histoire."""
    return bool(SENSORY_RE.search(key(phrase))) and not EDITORIAL_RE.search(phrase)


def _phrase_details(phrase: str) -> list[str]:
    """Ne reformule que des sensations explicitement présentes dans la phrase."""
    if not _usable_phrase(phrase):
        return []
    return [label for pattern, label in PHRASE_FR if re.search(pattern, phrase, re.I)][:2]


def _new_details(details: list[str], already_said: str) -> list[str]:
    """Évite de répéter sous deux formulations la même note olfactive."""
    overlap = (
        ("marine", "marin"), ("agrumes", "agrume"), ("boisée", "bois"),
        ("florales", "floral"),
    )
    return [detail for detail in details if not any(
        word in detail and root in already_said for word, root in overlap
    )]


def explain_top(recommendations: list[dict], catalogue: pd.DataFrame,
                profile_summary: Mapping[str, object]) -> list[dict]:
    """Complète les résultats avec les indices de correspondance et le texte."""
    frame = catalogue.reset_index(drop=True)
    families = set(profile_summary["families"])
    seasons = set(profile_summary["seasons"])
    moments = set(profile_summary["moments"])
    shared_context = {
        field: bool(recommendations) and all(r["context_matches"].get(field, 0) > 0
                                              for r in recommendations)
        for field in ("season", "moment")
    }
    used_phrases = set()
    for position, rec in enumerate(recommendations):
        row = frame.iloc[rec.pop("_index")]
        family = FAMILY_FR.get(key(rec["family"]), rec["family"].lower())
        user_words = key(profile_summary["text"] + " " + profile_summary["mood"])
        # Correspondance lexicale attestée, sans second calcul SBERT ou scores inventés.
        all_terms = _terms(row)
        terms = [term for term in all_terms if _mentioned(term, user_words)]
        rec["matched_terms"] = [{"term": term} for term in terms[:3]]
        evidence = list(dict.fromkeys(NOTE_FR[key(term)] for term in terms
                                      if key(term) in NOTE_FR))[:2]
        other_notes = list(dict.fromkeys(NOTE_FR[key(term)] for term in all_terms
                                         if key(term) in NOTE_FR))[:2]
        pair = rec["matched_phrases"]
        parts = []
        if evidence:
            leads = ("Tu y retrouveras", "Il propose", "Son profil met en avant")
            parts.append(f"{leads[position % len(leads)]} {_join(evidence)}, en accord avec ta recherche.")
        elif families and key(rec["family"]) in families and other_notes:
            parts.append(f"Il associe {_join(other_notes)}." if position % 2 else
                         f"Son profil {family} repose sur {_join(other_notes)}.")
        elif families and key(rec["family"]) in families:
            parts.append(f"Son profil {family} correspond à ta préférence olfactive.")
        elif rec["profile_similarity"] is not None:
            parts.append(("Son profil olfactif ressort parmi les parfums proches de ta description."
                          if position % 2 == 0 else
                          "Ses accords et ses notes se rapprochent de ce que tu décris."))
        else:
            parts.append(f"Son caractère {family} mérite ton attention.")
        details = (_new_details(_phrase_details(pair["perfume"]), " ".join(parts).lower())
                   if pair["user"] else [])
        if (rec["description_similarity_raw"] is not None and details and
                pair["perfume"] not in used_phrases):
            parts.append(f"Sa description évoque {_join(details)}.")
            used_phrases.add(pair["perfume"])
        if not shared_context["moment"] and rec["context_matches"].get("moment", 0) > 0:
            parts.append("Tu peux le porter en journée." if moments == {"day"} else
                         "Il convient à une sortie en soirée." if moments == {"night"} else
                         "Tu peux le porter de jour comme de soir." if
                         rec["context_matches"]["moment"] == 1 else
                         "Il correspond à l'un des moments que tu as choisis.")
        if not shared_context["season"] and rec["context_matches"].get("season", 0) > 0:
            parts.append("Il correspond à une saison que tu as choisie.")
        if rec["components"].get("performance", 0) >= .75 and position == 0:
            parts.append("Sa tenue ou son sillage se rapproche de tes préférences.")
        if len(parts) == 1 and position > 0:
            if rec["context_matches"].get("moment", 0) > 0 and moments == {"day"}:
                parts.append("Il se porte aussi facilement en journée.")
            elif rec["context_matches"].get("season", 0) > 0 and seasons:
                parts.append("Il convient à la saison que tu as choisie.")

        caveats = []
        if families and key(rec["family"]) not in families:
            caveats.append("Sa famille olfactive diffère de ta sélection.")
        if seasons and rec["context_matches"].get("season") == 0:
            actual = {SEASONS[key(s)] for s in str(row.get("BestSeasons", "")).split(",")
                      if key(s) in SEASONS}
            if actual:
                ordered = [SEASONS_FR[s] for s in SEASONS_FR if s in actual]
                caveats.append(f"Il se porte plutôt {_join(ordered)}.")
        if moments and rec["context_matches"].get("moment") == 0:
            caveats.append("Son moment de port habituel diffère de celui que tu as choisi.")
        rec["explanation_parts"] = {"why": " ".join(parts), "caveat": " ".join(caveats)}
        rec["explanation"] = (
            f"{rec['name']} · {rec['brand']} — {rec['score']:.1f}/100\n"
            f"Pourquoi lui ? {rec['explanation_parts']['why']}"
        )
        if caveats:
            rec["explanation"] += "\nÀ savoir : " + rec["explanation_parts"]["caveat"]
    return recommendations

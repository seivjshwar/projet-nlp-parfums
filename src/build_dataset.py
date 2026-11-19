"""Construit le référentiel data/reference/perfumes.csv à partir du jeu Kaggle.

Source : « Fragrantica Perfumes » (ledecanteur) —
https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes

Usage :
    python -m src.build_dataset --input data/raw/perfumes.csv --n 300

Étapes :
1. sélection des N parfums les plus votés (parfums connus → démo parlante) ;
2. extraction de la famille olfactive Fragrantica depuis la description ;
3. nettoyage des notes et accords (on retire les poids « Vanilla:100 ») ;
4. nettoyage de la description : on retire les phrases « template »
   (date de lancement, parfumeur, liste des notes) et le texte marketing
   sans valeur olfactive (contenances, publicité) ;
5. dérivation des saisons, du moment (jour/nuit) et d'une intensité 1–5
   à partir des votes de la communauté.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.config import PERFUMES_CSV, ROOT_DIR

# Première « grande » famille Fragrantica → famille de notre référentiel
FAMILY_MAP = {
    "Floral": "Floral",
    "Oriental": "Ambré",
    "Woody": "Boisé",
    "Aromatic": "Aromatique",
    "Chypre": "Chypré",
    "Citrus": "Hespéridé",
    "Leather": "Cuir",
}
# Repli quand la description ne donne pas la famille : accord dominant → famille
ACCORD_FALLBACK = {
    "floral": "Floral", "white floral": "Floral", "rose": "Floral", "fruity": "Floral",
    "powdery": "Floral", "iris": "Floral", "tuberose": "Floral", "violet": "Floral",
    "amber": "Ambré", "vanilla": "Gourmand", "sweet": "Gourmand", "warm spicy": "Ambré",
    "oud": "Ambré", "balsamic": "Ambré", "honey": "Gourmand", "caramel": "Gourmand",
    "woody": "Boisé", "earthy": "Boisé", "patchouli": "Boisé", "mossy": "Chypré",
    "aromatic": "Aromatique", "lavender": "Aromatique", "fresh spicy": "Aromatique",
    "green": "Aromatique", "herbal": "Aromatique",
    "citrus": "Hespéridé", "fresh": "Hespéridé", "aquatic": "Hespéridé", "marine": "Hespéridé",
    "leather": "Cuir", "tobacco": "Cuir", "smoky": "Cuir", "animalic": "Cuir",
}
FAMILY_WORDS = r"(?:Oriental|Floral|Woody|Aromatic|Chypre|Citrus|Leather|Fruity|Vanilla|Spicy|Musk|" \
               r"Gourmand|Fougere|Green|Aquatic|Aldehyde|Water)"
FAMILY_RE = re.compile(rf" is an? ((?:{FAMILY_WORDS} ?)+) fragrance for")

# Phrases de la description à retirer (template ou marketing)
DROP_SENTENCE_RE = re.compile(
    r"fragrance for (?:women|men)|was launched in|nose behind|was created by|created by|"
    r"top notes? (?:is|are)|middle notes? (?:is|are)|base notes? (?:is|are)|the fragrance features|"
    r"\bml\b|\boz\b|advertis|available|launch|campaign|bottle|flacon|winner of award|new fragrance|"
    r"collection|models?\b|market|perfumer|presents|signed|edition|celebrit",
    re.IGNORECASE,
)
GENDER_LABEL = {"female": "women", "male": "men", "unisex": "women and men"}
SEASONS = ["winter", "spring", "summer", "autumn"]


# ----------------------------------------------------------------- utilitaires
def strip_weights(cell: str, k: int | None = None) -> str:
    """'Vanilla:100|Musk:22' → 'Vanilla; Musk' (dans l'ordre d'importance)."""
    if not isinstance(cell, str) or not cell.strip():
        return ""
    items = [part.split(":")[0].strip() for part in cell.split("|") if part.strip()]
    return "; ".join(items[:k] if k else items)


def extract_family(description: str, accords: str) -> tuple[str, str]:
    """Renvoie (famille du référentiel, sous-famille Fragrantica)."""
    m = FAMILY_RE.search(description or "")
    if m:
        sub = m.group(1).strip()
        # Sous-familles sucrées regroupées en « Gourmand » (sinon noyées dans Ambré)
        if "Gourmand" in sub or sub == "Oriental Vanilla":
            return "Gourmand", sub
        main = FAMILY_MAP.get(sub.split()[0])
        if main:
            return main, sub
    first_accord = strip_weights(accords, 1).lower()
    return ACCORD_FALLBACK.get(first_accord, "Ambré"), first_accord.title()


def clean_description(description: str, max_sentences: int = 4) -> str:
    """Garde les phrases éditoriales qui décrivent l'odeur ou l'ambiance."""
    text = re.sub(r"\s+", " ", description or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if len(s) > 25 and not DROP_SENTENCE_RE.search(s)]
    return " ".join(kept[:max_sentences])


def dominant(votes: pd.Series, ratio: float = 0.8) -> list[str]:
    """Catégories dont les votes atteignent au moins `ratio` × le maximum."""
    if votes.max() <= 0:
        return []
    return [k for k, v in votes.items() if v >= ratio * votes.max()]


def intensity_raw(longevity: float, sillage: float) -> float:
    """Tenue (1–5) et sillage (1–4) ramenés sur 0–1 puis moyennés."""
    return ((longevity - 1) / 4 + (sillage - 1) / 3) / 2


def intensity_levels(raw_scores: pd.Series) -> pd.Series:
    """Niveaux 1–5 par quintiles : intensité *relative* au catalogue.

    Les moyennes de votes sont très resserrées (presque tout serait « 3 ») ;
    les quintiles donnent une échelle utilisable dans le questionnaire.
    """
    return pd.qcut(raw_scores.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


# ----------------------------------------------------------------- construction
def build(raw: pd.DataFrame, n: int, min_votes: int = 500) -> pd.DataFrame:
    df = raw[raw["vote_count"] >= min_votes].copy()
    df = df.drop_duplicates(subset=["name", "brand"]).sort_values("vote_count", ascending=False)
    df = df[df["accords"].notna()].head(n)

    rows = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        family, subfamily = extract_family(r.description, r.accords)
        top, mid, base = (strip_weights(c) for c in (r.notes_top, r.notes_middle, r.notes_base))
        if not (top or mid or base):          # pas de pyramide : notes « à plat »
            mid = strip_weights(r.notes_flat)
        accords = strip_weights(r.accords, 6)
        seasons = dominant(pd.Series({s: getattr(r, s) for s in SEASONS}))
        moment = dominant(pd.Series({"day": r.day, "night": r.night}), ratio=0.85)

        # Description = phrase générée à partir des données + texte éditorial nettoyé
        article = "An" if subfamily[:1].lower() in "aeiou" else "A"
        generated = (f"{article} {subfamily.lower()} fragrance for {GENDER_LABEL.get(r.gender, 'everyone')}, "
                     f"with {accords.replace('; ', ', ')} accords.")
        editorial = clean_description(r.description)

        rows.append({
            "PerfumeID": f"P{i:04d}",
            "Name": r.name,
            "Brand": r.brand,
            "Year": int(r.year) if pd.notna(r.year) else "",
            "Gender": r.gender,
            "Family": family,
            "SubFamily": subfamily,
            "Accords": accords,
            "TopNotes": top,
            "HeartNotes": mid,
            "BaseNotes": base,
            "Description": f"{generated} {editorial}".strip(),
            "Seasons": "; ".join(seasons),
            "Moment": "; ".join(moment),
            "Intensity": intensity_raw(r.longevity_avg, r.sillage_avg),
            "Longevity": round(r.longevity_avg, 2),
            "Sillage": round(r.sillage_avg, 2),
            "Rating": round(r.rating_avg, 2),
            "Votes": int(r.vote_count),
            "URL": r.url,
        })
    out = pd.DataFrame(rows)
    out["Intensity"] = intensity_levels(out["Intensity"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT_DIR / "data" / "raw" / "perfumes.csv")
    parser.add_argument("--output", type=Path, default=PERFUMES_CSV)
    parser.add_argument("--n", type=int, default=300, help="nombre de parfums à garder")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    out = build(raw, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"{len(out)} parfums écrits dans {args.output}")
    print(out["Family"].value_counts().to_string())


if __name__ == "__main__":
    main()

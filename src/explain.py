"""Explications transparentes, générées par règles à partir des scores NLP."""
import pandas as pd

from src.config import HIGH_MATCH, MEDIUM_MATCH

CRITERION_LABELS = {
    "description": "l'ambiance recherchée",
    "notes": "les notes olfactives",
    "occasion": "l'occasion d'usage",
    "family": "la famille olfactive",
}


def level(score: float) -> str:
    if score >= HIGH_MATCH:
        return "forte"
    if score >= MEDIUM_MATCH:
        return "moyenne"
    return "faible"


def explain_perfume(row: pd.Series) -> str:
    """Explication en Markdown d'un parfum recommandé."""
    lines = [
        f"**{row['Name']}** ({row['Brand']}) · famille {row['Family']} · {row['PriceRange']} · "
        f"intensité {row['Intensity']}/5",
        f"Score {row['score']:.2f} (affinité sémantique {row['affinity']:.2f}, "
        f"couverture de vos critères {row['coverage']:.0%}).",
    ]
    crit_scores = {c: row[f"S_{c}"] for c in CRITERION_LABELS if f"S_{c}" in row}
    for crit, s in sorted(crit_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]:
        detail = f"- Correspondance {level(s)} sur {CRITERION_LABELS[crit]} ({s:.2f})"
        if f"best_{crit}" in row:
            detail += f", rapprochée de votre phrase : _« {row[f'best_{crit}']} »_"
        lines.append(detail + ".")
    for reason in row["constraint_reasons"]:
        lines.append(f"- Attention : {reason}.")
    return "\n".join(lines)

"""Point d'entrée du Top 3 pour l'application et les notebooks.

    from src.recommender import recommend
    result = recommend(user_input)

Le moteur NLP du Membre 1 possède le modèle, le catalogue et son cache .npz.
On peut injecter ses résultats dans recommend() pour tester sans téléchargement.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

if __package__:
    from .explain import explain_top
    from .scoring import score_catalog
else:
    from explain import explain_top
    from scoring import score_catalog


def recommend(
    user_input: Mapping[str, object],
    *,
    top_k: int = 3,
    nlp_result: Mapping[str, object] | None = None,
    catalogue: pd.DataFrame | None = None,
) -> dict:
    """Retourne le classement, le Top 3 expliqué et le radar des familles."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k doit être un entier positif.")
    if nlp_result is None or catalogue is None:
        if __package__:
            from .nlp_engine import analyze, get_catalogue
        else:
            from nlp_engine import analyze, get_catalogue
        if nlp_result is None:
            nlp_result = analyze(dict(user_input))
        if catalogue is None:
            catalogue, _ = get_catalogue()
    result = score_catalog(catalogue, user_input, nlp_result)
    selected = result["ranking"][:top_k]
    explain_top(selected, catalogue, result["profile_summary"])
    for rec in result["ranking"][top_k:]:
        rec.pop("_index")
    result["recommendations"] = selected
    return result

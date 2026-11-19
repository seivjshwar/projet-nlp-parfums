# Notebooks

Le sujet demande que la documentation soit intégrée aux notebooks. Proposition :

| Notebook | Contenu | Responsable |
|---|---|---|
| `01_referentiel_parfums.ipynb` | Construction, nettoyage et exploration du jeu de parfums | Membre 1 |
| `02_embeddings_comparaison.ipynb` | Mots-clés vs Word2Vec/fastText vs SBERT anglais vs SBERT multilingue | Membre 2 |
| `03_scoring_recommandation.ipynb` | Démo du pipeline complet, justification des poids et seuils | Membre 3 |
| `04_bonus_ia_generative.ipynb` | (facultatif) LLM / RAG pour les explications | Membre 3 ou 4 |

Dans un notebook, importer le code du projet :

```python
import sys; sys.path.append("..")
from src.recommender import load_reference
```

**Vider les sorties avant de commit** pour limiter les conflits.

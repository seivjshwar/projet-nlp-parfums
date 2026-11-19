# Recommandation de parfums par analyse sémantique

Projet NLP — ECE ING5 Data & IA 2026-27 (Big Data & Data Science, Sarah Malaeb) — **Thème 6 : recommandation de parfums selon les préférences olfactives et émotionnelles**.

L'utilisateur décrit en langage naturel le parfum qu'il recherche (ambiance, notes aimées, occasion), complète une auto-évaluation par famille olfactive (Likert) et quelques questions guidées (saison, moment, intensité, genre, notes à éviter). Ses réponses (en français) sont comparées sémantiquement (SBERT multilingue + similarité cosinus) à un référentiel de 300 parfums issus de Fragrantica (en anglais). Le système calcule un score d'affinité par critère et par famille, puis recommande les 3 parfums les plus adaptés, avec une explication.

> Prototype pédagogique, sans visée commerciale.

## Démarrage rapide

Prérequis : Python ≥ 3.10.

```bash
git clone <URL_DU_DEPOT>
cd projet-nlp-parfums
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Le premier lancement télécharge le modèle SBERT (~470 Mo pour le modèle multilingue).

Le référentiel `data/reference/perfumes.csv` est déjà généré et versionné. Pour le **reconstruire** (autre nombre de parfums, nouveau nettoyage) :

1. Télécharger le jeu Kaggle [Fragrantica Perfumes](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes).
2. Placer le CSV dans `data/raw/perfumes.csv` (dossier ignoré par Git : ~140 Mo).
3. Lancer :
   ```bash
   python -m src.build_dataset --n 300
   ```

Tests (sans téléchargement de modèle) :

```bash
pytest
```

## Structure du dépôt

```
├── app/
│   └── streamlit_app.py      # Questionnaire + résultats + visualisations
├── data/
│   ├── raw/                  # Jeu Kaggle brut (non versionné)
│   ├── reference/            # perfumes.csv (généré), families.csv (référentiel)
│   └── responses/            # Réponses collectées (JSON, non versionnées)
├── docs/
│   └── methodologie.md       # Choix de conception, justification du scoring
├── notebooks/                # Notebooks documentés (livrable demandé)
├── src/
│   ├── build_dataset.py      # Kaggle brut → perfumes.csv (sélection, nettoyage)
│   ├── config.py             # Chemins, modèle, poids, seuils
│   ├── preprocessing.py      # Nettoyage, découpage en phrases
│   ├── embeddings.py         # Encodage SBERT, similarité cosinus
│   ├── scoring.py            # Scores par critère, par famille, contraintes
│   ├── recommender.py        # Chargement du référentiel, top 3
│   └── explain.py            # Explications par règles / templates
└── tests/                    # Tests unitaires
```

## Pipeline et scoring

1. **Questionnaire** → 3 textes libres (ambiance, notes, occasion), Likert par famille, saison, moment, intensité, genre, notes à éviter.
2. **Prétraitement** → découpage en phrases, normalisation.
3. **Embeddings** → `paraphrase-multilingual-MiniLM-L12-v2` (réponses en français).
4. **Similarité par critère** : pour chaque parfum, `S_a` = meilleure similarité cosinus entre les phrases de l'utilisateur et le texte du parfum pour ce critère (description, notes, occasion).
5. **Affinité par famille** : similarité avec la description de la famille, combinée avec la note Likert → diagramme radar du profil olfactif.
6. **Score d'affinité** : $\text{Affinité}(p) = \frac{\sum_a W_a \cdot S_a(p)}{\sum_a W_a}$
7. **Contraintes** : pénalités d'intensité, de notes à éviter et de genre (les parfums mixtes ne sont jamais pénalisés).
8. **Couverture** : part pondérée des critères bien satisfaits (S_a ≥ seuil).
9. **Top 3 + explication** : critères qui contribuent le plus, phrase de l'utilisateur la plus proche, alertes.

Poids et seuils dans `src/config.py`, à justifier dans `docs/methodologie.md`.

## Équipe et répartition

| Membre | Rôle principal | Dossiers |
|---|---|---|
| Membre 1 — _à compléter_ | Référentiel parfums & prétraitement | `src/build_dataset.py`, `data/reference/`, `src/preprocessing.py` |
| Membre 2 — _à compléter_ | Embeddings & comparaison des modèles | `src/embeddings.py`, `notebooks/02_*` |
| Membre 3 — _à compléter_ | Scoring, explications & bonus IA gén. | `src/scoring.py`, `src/explain.py` |
| Membre 4 — _à compléter_ | Interface web & visualisations | `app/` |

Tout le monde participe à la présentation finale (évaluation individuelle).

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour le workflow Git.

## Livrables (Boostcamp)

- [x] Référentiel d'au moins 50 parfums (300, Fragrantica), origine documentée
- [ ] Code source fonctionnel + documentation dans les notebooks
- [ ] Code commenté à chaque étape
- [ ] Support de présentation
- [ ] (optionnel) Documentation PDF

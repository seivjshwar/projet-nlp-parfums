# Méthodologie (à compléter au fil du projet)

Base du rapport PDF et de la présentation. Chaque section correspond à une exigence du sujet (thème 6, parfums).

## 1. Domaine d'application
Recommandation de parfums selon les préférences olfactives et émotionnelles. Pourquoi c'est un bon cas d'analyse sémantique : le vocabulaire olfactif est subjectif et riche en synonymes (« propre », « linge frais », « musc blanc »), ce qui met en défaut une simple recherche de mots-clés.

## 2. Jeu de données de référence (exigence 2)
- Origine : Kaggle « Fragrantica Perfumes » (140 228 parfums), cf. `data/reference/README.md`.
- Sélection : les 300 parfums les plus votés.
- Structure : `perfumes.csv` (famille, accords, notes de tête/cœur/fond, description, saisons, moment, intensité, genre) et `families.csv` (8 familles olfactives décrites en langage naturel).
- Prétraitement : `src/build_dataset.py` (extraction de la famille, nettoyage des descriptions, saisons et intensité dérivées des votes).

## 3. Questionnaire (exigence 1)
| Type | Questions | Utilisation |
|---|---|---|
| Ouvertes | ambiance, notes aimées, occasion | embeddings, un critère par question |
| Likert | appréciation de chaque famille olfactive | affinité par famille |
| Guidées | saison, moment, intensité, genre | texte d'occasion, contraintes |
| Cases à cocher | notes à éviter | pénalité |

## 4. Moteur sémantique (exigence 3)
- Modèle : `paraphrase-multilingual-MiniLM-L12-v2`. Justification : réponses en français et fiches en anglais (correspondance **interlingue**), phrases courtes, modèle léger.
- Comparaison à faire : `all-MiniLM-L6-v2` (anglais), baseline Word2Vec/fastText (moyenne des vecteurs de mots), recherche par mots-clés.
- Similarité : cosinus.
- Exemples montrant l'apport sémantique : …

## 5. Scoring (exigence 4)
- `S_a(p)` = max des cosinus phrases utilisateur ↔ texte du parfum pour le critère a. Pourquoi le max : …
- Affinité famille = (1 − λ) · similarité sémantique + λ · Likert normalisé (λ = 0,4).
- Affinité(p) = Σ W_a · S_a / Σ W_a avec W = description 0,35 · notes 0,30 · famille 0,20 · occasion 0,15. Justification : …
- Contraintes : intensité (−0,08 par point d'écart), notes à éviter (−0,25 par note), genre (−0,30 si le parfum est marqué pour l'autre genre ; mixte jamais pénalisé). Pas de budget : le jeu ne contient pas de prix.
- Couverture = part pondérée des critères avec S_a ≥ 0,35.
- Calibrage des seuils : …

## 6. Recommandation & explications (exigence 5)
Top 3, score, critères contributeurs, phrase de l'utilisateur la plus proche, alertes de contraintes.

## 7. Visualisations (exigence 6)
Radar du profil olfactif, barres de contribution des critères, heatmap phrases × parfums, tableau du classement complet.

## 8. Bonus IA générative (facultatif)

## 9. Difficultés rencontrées et solutions

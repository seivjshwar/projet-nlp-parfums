# Méthodologie (à compléter au fil du projet)

Base du rapport PDF et de la présentation. Chaque section correspond à une exigence du sujet (thème 6, parfums).

## 1. Domaine d'application
Recommandation de parfums selon les préférences olfactives et émotionnelles. Pourquoi c'est un bon cas d'analyse sémantique : le vocabulaire olfactif est subjectif et riche en synonymes (« propre », « linge frais », « musc blanc »), ce qui met en défaut une simple recherche de mots-clés.

## 2. Jeu de données de référence (exigence 2)
- Origine : …
- Structure : `perfumes.csv` (notes de tête/cœur/fond, description, occasion, intensité, prix) et `families.csv` (8 familles olfactives décrites en langage naturel).
- Prétraitement : …
- Taille : ≥ 50 parfums.

## 3. Questionnaire (exigence 1)
| Type | Questions | Utilisation |
|---|---|---|
| Ouvertes | ambiance, notes aimées, occasion | embeddings, un critère par question |
| Likert | appréciation de chaque famille olfactive | affinité par famille |
| Guidées | saison, intensité, budget | texte d'occasion, contraintes |
| Choix multiples | notes à éviter | pénalité |

## 4. Moteur sémantique (exigence 3)
- Modèle : `paraphrase-multilingual-MiniLM-L12-v2`. Justification : réponses en français, phrases courtes, modèle léger.
- Comparaison à faire : `all-MiniLM-L6-v2` (anglais), baseline Word2Vec/fastText (moyenne des vecteurs de mots), recherche par mots-clés.
- Similarité : cosinus.
- Exemples montrant l'apport sémantique : …

## 5. Scoring (exigence 4)
- `S_a(p)` = max des cosinus phrases utilisateur ↔ texte du parfum pour le critère a. Pourquoi le max : …
- Affinité famille = (1 − λ) · similarité sémantique + λ · Likert normalisé (λ = 0,4).
- Affinité(p) = Σ W_a · S_a / Σ W_a avec W = description 0,35 · notes 0,30 · famille 0,20 · occasion 0,15. Justification : …
- Contraintes : budget (filtre), intensité (−0,08 par point d'écart), notes à éviter (−0,25 par note).
- Couverture = part pondérée des critères avec S_a ≥ 0,35.
- Calibrage des seuils : …

## 6. Recommandation & explications (exigence 5)
Top 3, score, critères contributeurs, phrase de l'utilisateur la plus proche, alertes de contraintes.

## 7. Visualisations (exigence 6)
Radar du profil olfactif, barres de contribution des critères, heatmap phrases × parfums, tableau du classement complet.

## 8. Bonus IA générative (facultatif)

## 9. Difficultés rencontrées et solutions

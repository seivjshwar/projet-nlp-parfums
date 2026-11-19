# Référentiel parfums

Jeu de **démarrage** : 16 parfums et 8 familles olfactives. Le sujet recommande **au moins 50 parfums**. Il faut donc l'enrichir et **vérifier les notes, prix et intensités** : ils ont été saisis comme approximations de départ.

Sources possibles pour l'enrichissement (à documenter, exigence 2) :
- jeux de données Fragrantica publiés sur Kaggle (rechercher « fragrantica perfume dataset ») ;
- pages produits des marques, fiches Fragrantica / Parfumo.

## `perfumes.csv`

| Colonne | Rôle | Utilisée pour |
|---|---|---|
| `PerfumeID`, `Name`, `Brand` | Identification | Affichage |
| `Family` | Famille olfactive (doit exister dans `families.csv`) | Affinité par catégorie |
| `TopNotes`, `HeartNotes`, `BaseNotes` | Notes séparées par `;` | Embedding « notes » + filtre notes à éviter |
| `Description` | Description sensorielle / émotionnelle | Embedding « description » |
| `Occasion`, `Season` | Contexte d'usage | Embedding « occasion » |
| `Intensity` | Sillage/tenue de 1 à 5 | Contrainte souple |
| `PriceRange` | `€` (< 60 €), `€€` (60–120 €), `€€€` (> 120 €) | Contrainte budget |
| `Keywords` | Mots-clés séparés par `;` | Ajoutés à la description |

## `families.csv`

Une ligne par famille olfactive, avec une **description en langage naturel** : c'est elle qu'on compare aux réponses de l'utilisateur pour calculer l'affinité par famille (diagramme radar).

Les textes sont en français, car le modèle par défaut est multilingue et les utilisateurs répondent en français.

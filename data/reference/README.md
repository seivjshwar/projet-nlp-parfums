# Référentiel parfums

## Origine

Jeu Kaggle **« Fragrantica Perfumes: Ratings, Notes, Votes & More »** (ledecanteur) : https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes. Il contient 140 228 parfums × 59 colonnes extraits de Fragrantica (textes en anglais). Vérifier et citer la licence indiquée sur la page Kaggle.

Le fichier brut n'est pas versionné (~140 Mo). `perfumes.csv` est **généré** par :

```bash
python -m src.build_dataset --input data/raw/perfumes.csv --n 300
```

## Prétraitement (`src/build_dataset.py`)

1. **Sélection** : parfums avec au moins 500 votes, doublons nom + marque retirés, puis les **300 plus votés**. Ce sont des parfums connus, ce qui rend la démo parlante. Le minimum du sujet est 50.
2. **Famille olfactive** : extraite de la phrase Fragrantica « … is a *Oriental Vanilla* fragrance for … », puis ramenée à 8 grandes familles (`families.csv`). Les sous-familles « Oriental Vanilla » et « … Gourmand » forment la famille Gourmand. Si la phrase est absente, la famille est déduite de l'accord dominant.
3. **Notes et accords** : on retire les poids de vote (`Vanilla:100|Musk:22` → `Vanilla; Musk`). On garde les 6 accords principaux.
4. **Description** :
   - une phrase générée à partir des données (sous-famille, genre, accords) ;
   - suivie du texte éditorial Fragrantica nettoyé : on retire les phrases « template » (lancement, parfumeur, liste de notes) et le marketing (contenances, publicité, collections). On garde au plus 4 phrases.
5. **Saisons / moment** : les saisons (et jour/nuit) qui recueillent au moins 80 % (85 % pour jour/nuit) des votes de la catégorie la plus votée.
6. **Intensité 1–5** : la moyenne de la tenue (1–5) et du sillage (1–4), normalisés, puis découpée en **quintiles** sur le catalogue. Les moyennes brutes sont trop resserrées : presque tout serait « 3 ».

Limites connues :
- Il n'y a pas de prix dans le jeu, d'où l'absence de critère budget.
- Une partie du texte éditorial reste peu olfactive (contexte de marque).

## `perfumes.csv`

| Colonne | Contenu | Utilisée pour |
|---|---|---|
| `PerfumeID`, `Name`, `Brand`, `Year`, `URL` | Identification | Affichage, lien Fragrantica |
| `Gender` | female / male / unisex | Contrainte genre |
| `Family`, `SubFamily` | Famille (8 classes) / sous-famille Fragrantica | Affinité par famille, affichage |
| `Accords` | 6 accords principaux | Embedding « notes » |
| `TopNotes`, `HeartNotes`, `BaseNotes` | Notes séparées par `;` | Embedding « notes » + notes à éviter |
| `Description` | Phrase générée + texte éditorial nettoyé | Embedding « description » |
| `Seasons`, `Moment` | Saisons et jour/nuit dominants | Embedding « occasion » |
| `Intensity` | 1–5 (quintiles tenue + sillage) | Contrainte intensité |
| `Longevity`, `Sillage`, `Rating`, `Votes` | Moyennes et votes Fragrantica | Affichage, analyse |

## `families.csv`

Une ligne par famille, avec les sous-familles Fragrantica qu'elle regroupe et une **description en langage naturel** (en français, avec quelques mots-clés en anglais). On la compare aux réponses de l'utilisateur pour calculer l'affinité par famille (diagramme radar).

# Moteur NLP (`src/nlp_engine.py`), Membre 1

Ce document est le **contrat entre le moteur NLP et le scoring (Membre 2)**. Il complète le contrat d'interface de l'équipe : `user_input` (entrée) et `results` (sortie du scoring).

## Utilisation

```python
from nlp_engine import analyze, get_catalogue

nlp = analyze(user_input)     # user_input au format du contrat d'interface
df, _ = get_catalogue()       # le CSV (200 parfums), dans le même ordre que les résultats
```

Au premier appel, le modèle et les embeddings se chargent (quelques secondes). Les appels suivants sont instantanés. Les embeddings sont mis en cache dans `data/embeddings.npz`, un fichier généré et non versionné. Ils sont recalculés automatiquement si le CSV ou la méthode change.

## Champs de `user_input` utilisés

| Champ | Utilisation |
|---|---|
| `description_libre` | Comparé au profil olfactif (`sim_profile`) et aux phrases de description (`sim_description`) |
| `ambiance_emotionnelle` | Comparé aux phrases de description (`sim_description`) |
| `moment`, `seasons` | Transformés en phrase (« Best worn during the day, in spring and summer. »), puis comparés au texte d'occasion (`sim_occasion`) |

Les noms anglais `free_description` et `emotional_mood` sont aussi acceptés. Les autres champs (`families`, `gender`, `budget`, `longevity`, `sillage`) **ne sont pas utilisés par le moteur NLP** : ce sont des critères structurés, gérés par le scoring.

## Sortie de `analyze(user_input)`

```python
{
  "perfume_similarities": DataFrame,          # 200 lignes, même ordre que le CSV
  "family_similarities": {famille: float},    # 8 familles, valeurs brutes
  "family_similarities_scaled": {famille: float},  # 8 familles, 0–1 (meilleure = 1)
  "user_sentences": [str],                    # phrases utilisateur analysées
  "occasion_query": str,                      # "" si ni moment ni saisons
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
}
```

Colonnes de `perfume_similarities` :

| Colonne | Signification | Plage observée* |
|---|---|---|
| `PerfumeID` | Identifiant, pour la jointure avec le CSV | |
| `sim_profile` | Cosinus entre la description libre et le profil olfactif (accords + notes) | ≈ 0,19 – 0,45 |
| `sim_description` | Cosinus de la meilleure paire (phrase utilisateur, phrase de description), **corrigé du hubness** | ≈ 0,20 – 0,62 |
| `sim_description_raw` | Même paire, cosinus brut, pour la transparence | |
| `sim_occasion` | Cosinus entre la requête d'occasion et l'occasion du parfum | ≈ 0,75 – 1,00 |
| `best_user_phrase` / `best_perfume_phrase` | Meilleure paire de phrases, pour les **explications** (`matched_phrases`) | |

\* Plages observées pour l'exemple du contrat (« frais, élégant, boisé, bureau »).

**Valeurs `NaN`** : elles signifient « pas d'information », et non « aucun rapport ».
- Le champ utilisateur correspondant est vide, donc le critère est à **ignorer**.
- `sim_description` vaut aussi `NaN` pour les **72 parfums sur 200 sans texte éditorial** utilisable.

Conseil : ne pas remplacer `NaN` par 0. Il vaut mieux recalculer la moyenne pondérée sur les seuls critères disponibles.

## Recommandations pour le scoring (issues des tests)

1. **Les similarités brutes ne sont pas des scores.** Leurs plages sont étroites (voir le tableau). Il vaut mieux les **remettre à l'échelle pour chaque requête** (min-max ou z-score sur les 200 parfums) avant de les pondérer. Sinon, tous les parfums ont un score final entre 0,6 et 0,8.
2. **`sim_profile` est le critère le plus fiable** pour les ingrédients. Par exemple, la requête « agrumes, citron » donne 7 « Citrus & Fresh » sur 10 dans le top. Il mérite le **poids principal**.
3. **`sim_description` capte l'ambiance et les images** (« au soleil » → « shimmers like sunbeams »). Il est plus bruité : on lui donne un **poids modéré**, et on s'en sert surtout pour l'**explication** grâce à `best_user_phrase` et `best_perfume_phrase`.
4. **`sim_occasion`** est très resserré (0,75 – 1,00). Une remise à l'échelle est indispensable, sinon il ne départage presque rien.
5. **Familles** : `family_similarities_scaled` est fait pour le **radar**. Pour le score, la préférence explicite de l'utilisateur (`families`) reste le signal le plus sûr.

## Choix techniques et difficultés résolues

| Problème constaté | Mesure | Solution |
|---|---|---|
| Le modèle ne lit que **128 tokens** : `SearchText` (≈ 230 tokens) était tronqué pour 200 parfums sur 200, et la description perdue | Comptage des tokens par colonne | Trois textes encodés séparément (profil, occasion, phrases de description), chacun lu en entier |
| Une **phrase de repli** générée pour les parfums sans description gonflait leur score : 7 à 10 parfums sur 10 dans les tops | Proportion de phrases de repli dans les tops sur 3 requêtes | Suppression : `sim_description = NaN` pour ces parfums |
| Le **nom de famille** dans le profil biaisait les résultats (« agrumes » → famille « Fruity ») | Familles du top 10 avant et après | Nom de famille retiré du profil (de 1 à 7 « Citrus & Fresh » sur 10) |
| **Hubness** : des phrases génériques (« The heart is blooming in opulent flowers… ») étaient proches de toutes les requêtes | Popularité de chaque phrase (moyenne de ses 10 plus proches voisines) | Correction **CSLS** : similarité − popularité centrée |
| **Similarité par famille** : la comparaison à une phrase rédigée par famille plaçait la bonne famille 1re pour seulement 3 requêtes sur 5 | Évaluation sur 5 requêtes types | Famille vue à travers ses parfums (moyenne des 5 meilleurs `sim_profile`) : **5 requêtes sur 5** |

## Limites connues

- Pour « frais, élégant, boisé », la famille Woody n'arrive que 6e : les mots d'ambiance (« élégant, propre ») tirent vers d'autres familles.
- 72 parfums n'ont pas de texte éditorial, donc pas de critère de description.
- L'évaluation repose sur 5 requêtes types. Un jeu de test plus large (10 à 15 requêtes avec les parfums attendus) permettrait de mesurer la qualité plus précisément.

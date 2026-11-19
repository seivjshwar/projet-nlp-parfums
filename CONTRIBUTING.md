# Workflow Git de l'équipe

## Règles de base

- **On ne pousse jamais directement sur `main`.** `main` doit toujours contenir une version qui tourne (démo possible à tout moment).
- Une tâche = une issue = une branche = une Pull Request.
- Chaque PR est relue par **au moins un autre membre** avant fusion.

## Nommage des branches

```
feature/<sujet>     # nouvelle fonctionnalité   ex : feature/questionnaire-likert
fix/<sujet>         # correction de bug         ex : fix/score-negatif
data/<sujet>        # modification du référentiel
docs/<sujet>        # documentation, notebooks
```

## Cycle d'une tâche

```bash
git checkout main
git pull                                  # partir de la dernière version
git checkout -b feature/mon-sujet

# ... travail ...
git add <fichiers>
git commit -m "Ajoute le calcul du score par bloc"
git push -u origin feature/mon-sujet
# → ouvrir une Pull Request sur GitHub, demander une relecture
```

Avant d'ouvrir la PR, se remettre à jour :

```bash
git fetch origin
git rebase origin/main      # ou : git merge origin/main
pytest
```

## Messages de commit

Impératif, en français, une ligne courte : `Ajoute…`, `Corrige…`, `Met à jour…`.

## Éviter les conflits

- Chacun travaille principalement dans « ses » dossiers (cf. tableau du README).
- Les interfaces entre modules (signatures des fonctions dans `src/`) se discutent en équipe avant d'être modifiées.
- Pour les notebooks : un notebook par personne/sujet, et **vider les sorties avant de commit** (`Kernel → Restart & Clear Output`) pour éviter des conflits illisibles.

## Ne pas versionner

Données personnelles, réponses collectées (`data/responses/`), clés d'API (`.env`), modèles téléchargés. Le `.gitignore` les exclut déjà.

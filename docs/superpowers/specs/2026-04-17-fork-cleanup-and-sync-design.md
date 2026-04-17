# Design — Nettoyage et synchronisation du fork `echo-whisperX`

**Date** : 2026-04-17
**Auteur** : Thibault (tb@e-sensia.com)
**Statut** : À valider

## Contexte

Le fork `E-Sensia/echo-whisperX` de `m-bain/whisperX` s'est dégradé au fil
des itérations :

- `origin/main` est à 32 commits derrière `upstream/main`.
- La branche de travail `whisperx_improvement_explo` contient un commit
  (`a7a7dfc`) qui ajoute ~2,3 M lignes dont 107 MB de corpus texte
  `models/corpus_*.txt`, plusieurs Markdown de scratch
  (`ANALYSE_ET_AMELIORATIONS.md`, `OPTIMISATIONS_AVANCEES.md`), 17
  prédictions KenLM et 42 JSON de résultats de benchmark.
- 7 branches sont présentes sur `origin`, dont plusieurs parasites :
  `improve-code` (vieille branche upstream leakée), `stable` (miroir figé
  et inutilisé), `bench/optimizations` et `echo-updates` (sous-ensembles
  de `whisperx_improvement_explo`).
- Deux lockfiles coexistent (`poetry.lock` tracké, `uv.lock` tracké) alors
  que le projet est passé à `uv`.
- Les tests (`tests/test_fallback.py`, `tests/test_lm_fusion.py`,
  `tests/test_transcribe_segment.py`) existent sur disque mais sont
  gitignorés.

## Objectifs

1. Adopter un modèle de branches git-flow (décision **B2** validée) :
   `main` = release echo, `develop` = intégration, `whisperx-upstream` =
   miroir local d'upstream.
2. Reconstruire un historique echo propre, sans réécriture de branches
   existantes (décision **R3** validée), en archivant l'ancien état sur
   `origin/archive/2026-04-pre-cleanup`.
3. Retirer le bruit commité (corpus, scratch Markdown, résultats de
   bench, `poetry.lock`) de l'historique cible.
4. Transitionner entièrement vers `uv` (suppression `poetry.lock`).
5. Versionner le répertoire `tests/`.
6. Documenter la procédure de synchronisation upstream pour usage
   récurrent.

## Non-objectifs

- Refactoring du code echo — on transporte les features telles quelles.
- Changement de dépendances hors retrait `poetry.lock`.
- Ajout de tests ou features nouvelles.
- Réécriture destructive (force-push) de branches publiques.

## Modèle de branches cible

```
upstream/main  ─── whisperx-upstream (miroir local, jamais poussé)
                      │
                      └─► origin/main      (release echo, default branch)
                               │
                               └─► origin/develop
                                        │
                                        └─► origin/feature/<nom>

origin/archive/2026-04-pre-cleanup       (historique pré-cleanup, gelé)
```

### Sémantique

- `main` : version echo qui tourne en production. Toute fusion s'y fait
  depuis `develop` via merge `--no-ff`. Taggée `v<x.y.z>-echo`.
- `develop` : branche d'intégration. Les features y sont mergées après
  review/test. Les mises à jour d'`upstream` y arrivent via merge de
  `whisperx-upstream`.
- `whisperx-upstream` : miroir strict local de `upstream/main`. Jamais
  poussée sur `origin`. Utilisée comme sas pour intégrer les
  nouveautés upstream.
- `feature/<nom>` : branche feature courte, tirée de `develop`, mergée
  dans `develop` via PR ou merge direct.
- `archive/<date>-<raison>` : snapshot gelé pour référence historique.
  Jamais mergée.

### Workflow sync upstream

```bash
git fetch upstream
git checkout whisperx-upstream
git merge --ff-only upstream/main

git checkout develop
git merge whisperx-upstream       # résoudre conflits
# tester (pytest + bench)
git push origin develop
```

### Workflow release

```bash
git checkout main
git merge --no-ff develop
git tag v<x.y.z>-echo
git push origin main --tags
```

## Plan d'exécution

### Phase 0 — Safety net

Poser un tag local `backup/2026-04-17-pre-cleanup` sur
`whisperx_improvement_explo`. Non poussé. Garantit un rollback complet
tant que le tag existe.

```bash
git tag backup/2026-04-17-pre-cleanup whisperx_improvement_explo
```

### Phase 1 — Archivage

```bash
git branch archive/2026-04-pre-cleanup whisperx_improvement_explo
git push origin archive/2026-04-pre-cleanup
```

L'ancien travail reste consultable via `origin/archive/2026-04-pre-cleanup`.

### Phase 2 — Création `main`, `develop`, `whisperx-upstream`

```bash
git fetch upstream --tags

# Recréer main depuis upstream/main récent
git branch -f main upstream/main   # = d00ec69 (progress_callback, v3.8.2)
git checkout main

# Créer develop depuis main
git branch develop

# Miroir local d'upstream (jamais poussé sur origin)
git branch whisperx-upstream upstream/main
```

### Phase 3 — Ré-application des features echo sur `develop`

Cherry-picks depuis `archive/2026-04-pre-cleanup`. Pour les commits qui
ajoutent des fichiers non voulus, utiliser le pattern :

```bash
git cherry-pick --no-commit <sha>
git reset HEAD -- <fichiers_non_voulus>
git checkout -- <fichiers_non_voulus>  # ou rm -f pour nouveaux fichiers
git commit -C <sha>                    # réutilise message original
```

Se placer sur `develop` : `git checkout develop`.

#### C1 — `feat(asr): expose no-speech and repetition quality metrics per segment`

- Source : `f2df4a5`
- Commande : `git cherry-pick f2df4a5`
- Fichiers : `whisperx/asr.py`, `whisperx/schema.py`

#### C2 — `chore: rename package to echo-whisperx`

- Source : `f91ce44`
- Commande : `git cherry-pick f91ce44`
- Fichiers : `pyproject.toml`, `whisperx/alignment.py`

#### C3 — `feat(asr): add transcribe_multi() for cross-call batched transcription`

- Source : `b7ea4f8`
- Retire : `poetry.lock`
- Commandes :
  ```bash
  git cherry-pick --no-commit b7ea4f8
  git reset HEAD -- poetry.lock
  rm -f poetry.lock
  git commit -C b7ea4f8
  ```
- Fichiers conservés : `.dockerignore`, `whisperx/asr.py`

#### C4 — `feat(bench): add optimization benchmark suite for parallel call transcription`

- Source : `f0e50c6`
- Retire : tous les `bench/results/*.json` (42 fichiers)
- Commandes :
  ```bash
  git cherry-pick --no-commit f0e50c6
  git reset HEAD -- 'bench/results/*.json'
  rm -f bench/results/*.json
  git commit -C f0e50c6
  ```
- Fichiers conservés : `bench/__init__.py`, `bench/compare.py`,
  `bench/config.py`, `bench/data.py`, `bench/harness.py`,
  `bench/metrics.py`, `bench/run_step.py`, `bench/simulate.py`,
  `bench/stream_compare.py`, `bench/triton/*`

#### C5 — `feat(asr): add multi-temperature fallback and KenLM shallow fusion rescoring`

- Source : `c9aad7b`
- Commande : `git cherry-pick c9aad7b`
- Fichiers : `whisperx/__main__.py`, `whisperx/asr.py`, `whisperx/transcribe.py`

#### C6 — `feat(bench): add KenLM training and evaluation scripts`

- Source : partie utile de `a7a7dfc`
- Commandes :
  ```bash
  git checkout archive/2026-04-pre-cleanup -- \
      bench/eval_whisperx.py \
      bench/extract_prod_corpus.py \
      bench/train_kenlm.py \
      whisperx/lm_fusion.py \
      CLAUDE.md
  git add bench/eval_whisperx.py bench/extract_prod_corpus.py \
          bench/train_kenlm.py whisperx/lm_fusion.py CLAUDE.md
  git commit -m "feat(bench): add KenLM training and evaluation scripts"
  ```
- Fichiers conservés : scripts bench + `whisperx/lm_fusion.py` +
  `CLAUDE.md`
- **Retirés** : `ANALYSE_ET_AMELIORATIONS.md`,
  `OPTIMISATIONS_AVANCEES.md`, `models/corpus_*.txt`,
  `bench/results/predictions_*.jsonl`
- Note : si `whisperx/lm_fusion.py` est déjà ajouté par C5 (à vérifier
  lors de l'exécution, car le fichier peut apparaître en C5 selon
  l'état du cherry-pick), le retirer du `git add` de C6.

#### C7 — `chore: update .gitignore and track tests/`

- Fichiers :
  - `.gitignore` modifié (voir Phase 4)
  - `tests/__init__.py`, `tests/test_fallback.py`,
    `tests/test_lm_fusion.py`, `tests/test_transcribe_segment.py` ajoutés
- Commandes :
  ```bash
  # éditer .gitignore
  git add .gitignore
  git add tests/
  git commit -m "chore: update .gitignore and version tests directory"
  ```

### Phase 4 — `.gitignore` final

Retrait :

```
tests/          # tests maintenant trackés
```

Ajouts :

```
# Bench artifacts
bench/results/

# Large KenLM training corpora (stored outside repo)
models/corpus_*.txt

# Scratch explorations
explorations/
last_name_*.json

# Legacy package manager (project uses uv)
poetry.lock
```

Les sections `models/*.arpa` et `models/*.bin` sont déjà présentes.

### Phase 5 — Documentation du workflow

Ajouter une section `## Development workflow` dans `README.md` (juste
après la section d'installation) :

````markdown
## Development workflow

### Branches

- `main` — current echo release (what runs in prod)
- `develop` — integration of in-progress features
- `feature/<name>` — individual features, branched from `develop`
- `archive/<date>-<reason>` — frozen history, reference only

### Syncing with upstream whisperX

This fork tracks `m-bain/whisperX` via the `upstream` remote.

```bash
git fetch upstream

# Update local mirror (never pushed on origin)
git checkout whisperx-upstream
git merge --ff-only upstream/main

# Integrate into develop
git checkout develop
git merge whisperx-upstream
# resolve conflicts, run tests
uv sync
pytest tests/
git push origin develop
```

### Cutting a release

```bash
git checkout main
git merge --no-ff develop
git tag v<x.y.z>-echo
git push origin main --tags
```
````

### Phase 6 — Vérification locale

1. `git log main..develop --oneline` doit lister exactement 7 commits
   (C1 à C7).
2. `git diff main..develop --stat` ne doit contenir aucun fichier
   `models/corpus_*.txt`, `bench/results/*.json`, `poetry.lock`,
   `ANALYSE_ET_AMELIORATIONS.md`, `OPTIMISATIONS_AVANCEES.md`,
   `bench/results/predictions_*.jsonl`.
3. `uv sync` (régénérer l'environnement).
4. `pytest tests/` — tous les tests doivent passer.
5. `python bench/eval_whisperx.py --label post-cleanup` — le WER doit
   rester aligné avec la baseline (35,4 % norm WER, tolérance ±0,5 pt).
6. Inspection visuelle de `git log main..develop --graph`.

### Phase 7 — Push et cleanup `origin`

À exécuter uniquement après validation des phases 0-6.

```bash
# Push des nouvelles branches
git push origin main
git push origin develop

# Changer la default branch sur GitHub
gh api -X PATCH repos/E-Sensia/echo-whisperX -f default_branch=main

# Suppression des branches remote obsolètes
git push origin --delete whisperx_improvement_explo
git push origin --delete bench/optimizations
git push origin --delete echo-updates
git push origin --delete improve-code
git push origin --delete stable
```

### Phase 8 — Cleanup des branches locales

```bash
git branch -D upstream-main
git branch -D bench/optimizations
git branch -D echo-updates
git branch -D whisperx_improvement_explo   # après confirmation que archive/ est bien sur origin
```

### Phase 9 — Cleanup disque local (optionnel)

```bash
rm -rf whisperx.egg-info/            # résidu, gitignoré
# models/medical_fr_4gram*.{arpa,bin} laissés en place (utile, gitignorés)
```

## Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Conflits lors des cherry-picks (upstream a bougé entre `064f737` et `d00ec69`) | Moyenne | Moyen | Résolution manuelle commit par commit ; le delta upstream est de 5 commits, principalement des fixes localisés (`blank_id`, `progress_callback`, revert wildcard alignment). |
| Tests cassés post-reconstruction | Faible | Moyen | Phase 6 exécute pytest + bench ; rollback via `git reset --hard backup/2026-04-17-pre-cleanup`. |
| Suppression accidentelle de branches | Faible | Élevé | Tag backup local + archive branch poussée **avant** toute suppression (Phases 0 et 1 précèdent tout le reste). |
| `uv sync` échoue après retrait `poetry.lock` | Faible | Faible | `uv.lock` intact ; si problème, `uv lock --upgrade` regénère. |
| `whisperx/lm_fusion.py` introduit dans `c9aad7b` ET `a7a7dfc` simultanément | Moyenne | Faible | Vérifier à l'exécution C5 si le fichier y apparaît déjà ; dans ce cas ne pas le re-ajouter en C6. |

## Critères de succès

- [ ] `origin/main` pointe sur un commit = `upstream/main` + 7 commits echo clean.
- [ ] `origin/develop` existe et est aligné avec `origin/main` à
      l'issue de l'exécution (identique, sauf si des commits ont été
      ajoutés ultérieurement).
- [ ] `origin/archive/2026-04-pre-cleanup` préservé et immutable.
- [ ] Branches `origin` présentes après cleanup : `main`, `develop`,
      `archive/2026-04-pre-cleanup`. Aucune autre.
- [ ] Default branch GitHub = `main`.
- [ ] `git diff upstream/main..main --name-only` n'inclut aucun corpus,
      résultat de bench, scratch Markdown ou `poetry.lock`.
- [ ] `pytest tests/` passe sur `develop`.
- [ ] `bench/eval_whisperx.py` reproduit baseline WER 35,4 % ±0,5 pt.
- [ ] Section "Development workflow" présente dans `README.md`.

## Ordre d'exécution et points de contrôle

Les phases 0-6 sont locales et réversibles. Pas de manipulation de
`origin` avant la phase 7. Points de contrôle :

1. Après phase 1 : confirmer que `origin/archive/2026-04-pre-cleanup`
   existe bien sur GitHub (vérification `gh api
   repos/E-Sensia/echo-whisperX/branches/archive/2026-04-pre-cleanup`).
2. Après phase 3 : inspection `git log main..develop --oneline` et
   `git diff main..develop --stat` avant de passer à Phase 4.
3. Après phase 6 : ne procéder à la phase 7 qu'après validation des
   tests et du bench.
4. Après phase 7 : confirmer via `gh api
   repos/E-Sensia/echo-whisperX/branches | jq '.[].name'` que la liste
   est bien `[main, develop, archive/2026-04-pre-cleanup]`.

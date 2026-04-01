# Audit technique de `echo-whisperX` (WhisperX)

Ce document remplace l'ancien contenu trop spéculatif (features “web”, “cloud”, “cache distribué”, etc.) par un audit basé sur **l'état réel du code présent dans ce repo**.

## 1) Périmètre et version

- Package : `whisperx` (version `3.7.4` dans `pyproject.toml`)
- Entrée CLI : `whisperx/__main__.py` → `whisperx/transcribe.py:transcribe_task`
- Pipeline : **VAD** → **ASR batched (faster-whisper / CTranslate2)** → **alignement CTC (wav2vec2)** → **diarization (pyannote-audio)** → **writers (SRT/VTT/TXT/TSV/JSON/AUD)**.

## 2) Cartographie du repo (ce que fait chaque fichier)

- `whisperx/__main__.py` : parsing CLI (arguments, defaults, logging).
- `whisperx/transcribe.py` : orchestrateur (boucles fichiers, load/unload modèles, appel alignement/diarization, écriture).
- `whisperx/asr.py` : wrapper faster-whisper + VAD + batching + (quelques métriques de qualité par chunk).
- `whisperx/alignment.py` : forced alignment wav2vec2/CTC (torchaudio ou Hugging Face) + segmentation “sentence”.
- `whisperx/diarize.py` : pipeline pyannote + assignation speaker aux segments/mots.
- `whisperx/vads/*` : VAD pyannote (modèle local `assets/pytorch_model.bin`) ou Silero (torch.hub).
- `whisperx/audio.py` : lecture audio via `ffmpeg` + log-mel spectrogram.
- `whisperx/utils.py` : writers + helpers (format timestamps, interpolation NaNs).
- `whisperx/schema.py` : `TypedDict` (API des résultats).
- `whisperx/SubtitlesProcessor.py` : splitting avancé SRT/VTT (actuellement non branché sur la CLI).

## 3) Flux d’exécution (CLI)

1. `whisperx/__main__.py` construit `args` et appelle `transcribe_task`.
2. `whisperx/transcribe.py` :
   - charge le modèle ASR via `whisperx.asr.load_model(...)` (inclut VAD),
   - transcrit chaque audio (VAD → chunks → batched decode),
   - libère le modèle ASR,
   - charge le modèle d’alignement (`load_align_model`) puis aligne segment par segment (`align`),
   - optionnel : charge le modèle de diarization et assigne les speakers,
   - écrit les sorties.

## 4) Points forts actuels

- Pipeline lisible et modulaire (ASR, alignement, diarization séparés).
- VAD avant ASR (réduction des “hallucinations” et batching plus efficace).
- Alignement CTC simple et robuste (fallback au segment original si non alignable).
- Nettoyage mémoire explicite côté CLI (déload des modèles entre étapes).

## 5) Problèmes et axes d’amélioration (priorisés)

### P0 — Bugs / incohérences à corriger en premier

1. **Import cassé dans `whisperx/asr.py`** : `from whisperx.types ...` alors que le module n’existe pas dans ce repo (les types sont dans `whisperx/schema.py`).
2. **VAD pyannote : `--vad_onset/--vad_offset` ignorés** : `whisperx/vads/pyannote.py` valide `vad_onset` mais ne le propage pas au pipeline (les seuils restent ceux par défaut).
3. **`--interpolate_method=ignore` annoncé mais non supporté** : `alignment.py` appelle `pandas.Series.interpolate(method="ignore")` → erreur.
4. **Langue écrasée à l’écriture** : `whisperx/transcribe.py` force `result["language"] = align_language` même si la langue détectée est différente.
5. **Progress** : division par zéro possible si VAD retourne 0 segment et `--print_progress True`.
6. **Arguments CLI non utilisés / trompeurs** :
   - `--segment_resolution` n’est utilisé nulle part.
   - `--best_of`, `--condition_on_previous_text`, `--fp16` ne pilotent pas réellement le comportement (au mieux partiellement via `compute_type`).

### P1 — Qualité ASR / robustesse

1. **Décodage batched ne réimplémente pas les garde-fous faster-whisper** (fallback multi-températures, `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`). Résultat : risques accrus de répétitions/hallucinations alors que des métriques (`compression_ratio`, `no_speech_prob`, tokens/second, répétitions) sont déjà calculées.
2. **Silero VAD via `torch.hub.load(..., trust_repo=True)`** :
   - dépend d’un dépôt distant (reproductibilité et sécurité),
   - conversion waveform (numpy/torch) fragile selon la version des utils silero.
3. **NLTK sentence splitter** :
   - chargé dans une boucle (coût),
   - ressource téléchargée au runtime (fragile en prod/offline),
   - modèle “english.pickle” utilisé pour toutes les langues.

### P2 — Performance (hotspots probables)

1. **`alignment.py`** :
   - inférence wav2vec2 *séquentielle* par segment (TODO de batching),
   - surcoût pandas (DataFrame, groupby) évitable,
   - recherche `list.index(...)` dans une boucle (complexité inutile).
2. **`diarize.py:assign_word_speakers`** :
   - recalcul des intersections + `groupby` à chaque segment et à chaque mot,
   - colonnes temporaires ajoutées en boucle (DataFrame muté).
3. **I/O audio** :
   - `audio.load_audio` charge tout en RAM (peut exploser sur des fichiers très longs).

### P3 — Maintenabilité / dette technique

- Mélange `print()` / `logger` (logs non homogènes, difficile à filtrer).
- Schéma de sortie non stabilisé (segments ASR enrichis de champs non décrits dans `schema.py`).
- Pas de tests (même “smoke tests”).
- Lockfiles incohérents : `uv.lock` est présent, `poetry.lock` existe mais est non tracké.

## 6) Recommandations concrètes (qu’est-ce qu’on change exactement)

### Correctifs P0 (faible risque, gros gain)

- Aligner les imports/types : `whisperx/asr.py` doit référencer `whisperx/schema.py` (ou ajouter un module `types.py` explicite).
- Rendre effectifs `--vad_onset/--vad_offset` pour pyannote.
- Gérer `--interpolate_method=ignore` (implémenter “ne pas interpoler” + ajuster le writer) **ou** retirer l’option pour éviter les crashes.
- Ne pas écraser `result["language"]` avec une valeur par défaut : conserver la langue détectée/choisie par l’ASR.
- Guard dans le progress quand `total_segments == 0`.
- Nettoyer l’interface CLI : retirer les flags non supportés, ou les implémenter réellement.

### Améliorations P1 (qualité)

- Réintroduire un équivalent de `generate_with_fallback` (faster-whisper) en version batched **ou** appliquer les seuils *post-decoding* :
  - rejet silence via `no_speech_prob`,
  - rejet répétitions via `compression_ratio` et/ou `repeat_*`,
  - fallback température/beam quand un chunk est suspect.
- Charger NLTK punkt une seule fois (et choisir le tokenizer selon `language_code` quand possible).
- Sécuriser Silero : remplacer `torch.hub` par une dépendance/pinning clair ou un modèle local, et standardiser le type de waveform (`torch.Tensor`).

### Optimisations P2 (perf)

- Batcher l’alignement wav2vec2 (pad + `attention_mask`/`lengths`) pour réduire le nombre d’appels modèle.
- Remplacer les DataFrame temporaires hot-path par des structures list/numpy.
- Optimiser l’assignation speaker (vectorisation + groupby hors boucles).
- Option “streaming decode” (ffmpeg) ou décodage par morceaux si vous traitez > 1h d’audio.

## 7) Plan d’implémentation priorisé (proposé)

### Semaine 1 — Stabiliser (P0)

- Fix imports + VAD thresholds + interpolate_method + langue + guard progress.
- Ajouter 2–3 smoke tests :
  - VAD retourne 0 segment (ne crash pas),
  - `--interpolate_method ignore` (comportement défini),
  - sortie JSON contient les clés attendues.

### Semaine 2 — Qualité (P1)

- Implémenter “fallback / filtering” inspiré de faster-whisper (même approximatif au début).
- Nettoyer les flags CLI (cohérence et docs).

### Semaine 3 — Perf (P2)

- Batching alignement wav2vec2.
- Optimiser diarization assignment.
- Bench automatique (RTF, VRAM, temps align/ASR/diarize).

## 8) Mesures recommandées (pour éviter l’optimisation “à l’aveugle”)

- **RTF (real-time factor)** global et par étape (ASR / align / diarize).
- **VRAM** max (pics) par étape.
- **Qualité** :
  - WER/CER sur un petit set de référence,
  - métriques “hallucination” (répétitions, `compression_ratio`, tokens/sec).

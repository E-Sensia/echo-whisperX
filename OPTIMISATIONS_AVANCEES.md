# Optimisations avancées (réalistes) pour `echo-whisperX`

Ce document remplace l’ancien contenu trop spéculatif (API web, cloud, cache distribué, etc.) par des optimisations **directement applicables** au pipeline actuel :

**VAD → ASR batched (faster-whisper/CTranslate2) → alignement wav2vec2/CTC → diarization pyannote → writers**.

---

## 1) Mesurer avant d’optimiser

Sans mesure, on risque d’améliorer “ce qui se voit” et pas “ce qui coûte”.

Mesures minimales recommandées :

- **Temps par étape** : VAD / ASR / align / diarize / write.
- **RTF (real-time factor)** : `temps_total / durée_audio`.
- **Pics VRAM/RAM** par étape.
- **Qualité** : WER/CER sur un petit set + indicateurs “anti-hallucinations” (répétitions, `compression_ratio`, `no_speech_prob`, tokens/sec).

---

## 2) ASR (faster-whisper / CTranslate2) : throughput vs qualité

Fichiers clés : `whisperx/asr.py`, `whisperx/transcribe.py`, `whisperx/__main__.py`.

### 2.1 Réintroduire les garde-fous de décodage (priorité qualité)

Le chemin “batched” actuel expose déjà des signaux utiles (`avg_logprob`, `compression_ratio`, `no_speech_prob`, répétitions, tokens/sec) mais **n’applique pas** les stratégies de fallback/filtrage qu’on trouve dans faster-whisper (multi-températures, seuils, skip silence).

Recommandation :

- implémenter une version batched de “fallback” (même simplifiée),
- ou appliquer un filtrage post-décodage (skip silence + rejet répétitions) et relancer un chunk suspect avec des paramètres plus robustes.

### 2.2 `compute_type`

Objectif : maximiser le débit sans sacrifier la qualité.

- `float16` : bon compromis GPU.
- `int8` : baisse VRAM (peut dégrader la qualité, surtout sur audio difficile).

Recommandation : documenter des presets par taille de modèle/VRAM, et envisager d’exposer les compute types “mixtes” si votre stack les supporte.

### 2.3 `batch_size` (et pourquoi ça plafonne parfois)

`batch_size` améliore la saturation GPU **si** :

- les chunks VAD sont assez longs et homogènes,
- l’overhead Python (préprocess/DataLoader) ne domine pas.

Recommandation :

- ajouter un mode “auto batch” (basé sur VRAM dispo),
- et un bench simple qui trace `batch_size → RTF/VRAM`.

### 2.4 Réduire l’overhead Python (si beaucoup de petits chunks)

Le batching passe par `transformers.Pipeline` + `DataLoader`. Avec beaucoup de segments courts, ce coût peut devenir visible.

Option perf : remplacer la surcouche Pipeline par une boucle explicite :

- préprocess (log-mel) → stack → `generate_segment_batched`,
- moins de conversions torch↔numpy,
- comportement batch plus simple à tester.

---

## 3) VAD : le meilleur accélérateur est de faire moins d’ASR

Fichiers clés : `whisperx/vads/pyannote.py`, `whisperx/vads/silero.py`, `whisperx/asr.py`.

### 3.1 Débloquer le tuning `vad_onset` / `vad_offset` (bug)

Aujourd’hui le chemin pyannote ne propage pas réellement ces seuils au pipeline. Corriger ça permet d’ajuster :

- recall (ne pas rater de parole),
- precision (ne pas envoyer de bruit à l’ASR),
- taille moyenne des chunks (donc vitesse).

### 3.2 `chunk_size` : contrainte explicite (≤ 30s)

Whisper est basé sur une fenêtre 30s (mels ~3000 frames). Le code pad sur `N_SAMPLES=30s`.

Recommandation :

- documenter que `chunk_size` doit rester ≤ 30,
- valider/clipper si l’utilisateur dépasse.

### 3.3 Silero : reproductibilité + sécurité

`torch.hub.load(..., trust_repo=True)` implique :

- dépendance à un dépôt externe,
- exécution de code téléchargé,
- variabilité selon cache/version.

Recommandation : pinner, vendoriser, ou remplacer par une dépendance contrôlée + standardiser le type waveform (`torch.Tensor`).

---

## 4) Alignement wav2vec2/CTC : hotspot fréquent

Fichier clé : `whisperx/alignment.py`.

### 4.1 Batching alignement (gros gain)

Actuel : 1 forward wav2vec2 par segment.

Optimisation : batcher les segments :

- pad à la longueur max du batch,
- fournir `lengths` (torchaudio) ou `attention_mask` (Hugging Face),
- traiter ensuite chaque item avec sa longueur réelle.

### 4.2 Réduire le coût pandas / Python

Le chemin critique crée des DataFrame, fait des groupby/agg, et utilise `list.index(...)` dans des boucles.

Recommandation :

- remplacer par listes/arrays (surtout pour `char_segments_arr` et la construction des mots),
- pré-calculer des maps (ex: `cdx → idx`) au lieu de `.index(...)`.

### 4.3 Sentence splitter : cache + offline

Le tokenizer NLTK est chargé dans une boucle et peut déclencher un téléchargement au runtime.

Recommandation :

- charger une seule fois hors boucle,
- prévoir un fallback simple (regex ponctuation) si la ressource n’est pas disponible,
- choisir un tokenizer par langue quand possible.

### 4.4 `--interpolate_method=ignore`

Aujourd’hui “ignore” n’est pas supporté par l’implémentation actuelle.

Recommandation : définir clairement le comportement :

- “ignore” = ne pas interpoler (laisser NaNs) et adapter writers,
- ou “ignore” = fusionner avec voisins,
- ou retirer l’option.

---

## 5) Diarization : assignation speaker (souvent coûteuse)

Fichier clé : `whisperx/diarize.py`.

### 5.1 Optimiser `assign_word_speakers`

Actuel : recalcul intersection + `groupby` à chaque segment et à chaque mot.

Optimisations :

- vectoriser via numpy (sans muter `diarize_df` en boucle),
- indexer les segments de diarization (interval tree / sweep-line) pour limiter les candidats,
- option : assignation au niveau segment uniquement si word-level n’est pas nécessaire.

### 5.2 Embeddings

Retourner des embeddings dans le JSON peut exploser la taille et la mémoire.

Recommandation : compression/downsampling, ou stockage séparé (ex: `.npy` / parquet) plutôt que dans la sortie JSON.

---

## 6) I/O audio : éviter les pics RAM

Fichier clé : `whisperx/audio.py`.

`ffmpeg` renvoie tout le PCM en mémoire. Pour des fichiers longs, c’est un point dur.

Options :

- streaming decode (lecture par chunks),
- découper tôt (si votre pipeline devient streaming),
- guard : alerter/forcer un mode “long audio” > X minutes.

---

## 7) “Quick wins” (1–2 jours)

- Corriger les P0 (imports, seuils VAD, `ignore`, langue, division par 0, flags CLI).
- Déplacer le chargement NLTK hors boucle + mode offline.
- Supprimer les lectures inutiles (ex: `open(...).read()` non utilisé).
- Uniformiser les logs (éviter `print()` en prod, tout passer par `logger`).


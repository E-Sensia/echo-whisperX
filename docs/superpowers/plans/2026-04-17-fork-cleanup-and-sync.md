# Fork Cleanup & Upstream Sync — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the `echo-whisperX` fork with a clean git-flow branching model (`main`/`develop`/`whisperx-upstream`), archive the current work, re-apply echo features as clean commits on top of recent upstream, and document the upstream sync workflow — all without destructive force-pushes.

**Architecture:** Linear cherry-pick-driven rebuild. Current tip (`whisperx_improvement_explo`) is tagged and branched as `archive/2026-04-pre-cleanup`. New `main` is reset to `upstream/main` (commit `d00ec69`). New `develop` is created from `main`, then accumulates 10 clean commits (C0 spec + C1…C9 features). All cleanup operations on `origin` happen only after full local verification.

**Tech Stack:** git, gh CLI, uv, pytest, WhisperX bench suite.

**Reference spec:** `docs/superpowers/specs/2026-04-17-fork-cleanup-and-sync-design.md`

---

## Pre-flight

### Task 1: Verify initial state

- [ ] **Step 1.1: Confirm current branch and clean tracked state**

Run:
```bash
git status
git branch --show-current
```

Expected output: current branch = `whisperx_improvement_explo`, and the only untracked files listed should be `explorations/`, `last_name_2_labelling.json`, `last_name_3_labelling.json`, and (if not yet committed) `docs/superpowers/specs/2026-04-17-fork-cleanup-and-sync-design.md` and `docs/superpowers/plans/2026-04-17-fork-cleanup-and-sync.md`. No tracked modified files.

If there are tracked modifications, stop and ask.

- [ ] **Step 1.2: Confirm remotes**

Run:
```bash
git remote -v
```

Expected output must include:
```
origin    git@github.com:E-Sensia/echo-whisperX.git (fetch)
upstream  https://github.com/m-bain/whisperX.git (fetch)
```

- [ ] **Step 1.3: Verify upstream has the expected tip**

Run:
```bash
git fetch upstream
git log --oneline upstream/main -1
```

Expected output: starts with `d00ec69 feat: add progress_callback to transcribe, align, and diarize` (or a commit newer than this — if newer, proceed, the plan still works).

---

## Safety net

### Task 2: Create backup tag

- [ ] **Step 2.1: Tag current tip**

Run:
```bash
git tag backup/2026-04-17-pre-cleanup whisperx_improvement_explo
```

- [ ] **Step 2.2: Verify tag exists**

Run:
```bash
git tag -l 'backup/*'
```

Expected output: `backup/2026-04-17-pre-cleanup`

**If anything goes wrong later**, recover with:
```bash
git checkout whisperx_improvement_explo
git reset --hard backup/2026-04-17-pre-cleanup
```

---

## Archive current work

### Task 3: Create and push archive branch

- [ ] **Step 3.1: Create archive branch from current tip**

Run:
```bash
git branch archive/2026-04-pre-cleanup whisperx_improvement_explo
```

- [ ] **Step 3.2: Push archive to origin**

Run:
```bash
git push origin archive/2026-04-pre-cleanup
```

- [ ] **Step 3.3: Verify archive is on origin**

Run:
```bash
gh api repos/E-Sensia/echo-whisperX/branches/archive/2026-04-pre-cleanup --jq '.name'
```

Expected output: `archive/2026-04-pre-cleanup`

---

## Create new branch structure

### Task 4: Reset `main` to upstream and create `develop` + `whisperx-upstream`

- [ ] **Step 4.1: Re-point local `main` to `upstream/main`**

Run:
```bash
git branch -f main upstream/main
```

This moves the local `main` ref to `d00ec69`. No checkout yet.

- [ ] **Step 4.2: Checkout `main`**

Run:
```bash
git checkout main
```

Note: untracked files (`explorations/`, `last_name_*.json`, `docs/superpowers/specs/*`, `docs/superpowers/plans/*`, `ANALYSE_ET_AMELIORATIONS.md`, `OPTIMISATIONS_AVANCEES.md`, `last_name_*.json`) persist in the working tree — this is expected.

- [ ] **Step 4.3: Create `develop` from `main`**

Run:
```bash
git branch develop
```

- [ ] **Step 4.4: Create `whisperx-upstream` local mirror**

Run:
```bash
git branch whisperx-upstream upstream/main
```

This branch must **never** be pushed to `origin`.

- [ ] **Step 4.5: Checkout `develop`**

Run:
```bash
git checkout develop
```

Expected output: `Switched to branch 'develop'`.

- [ ] **Step 4.6: Verify branch state**

Run:
```bash
git log --oneline main -1
git log --oneline develop -1
git log --oneline whisperx-upstream -1
```

Expected: all three show the same commit (starting with `d00ec69`).

---

## Commit C0 — Design spec + plan

### Task 5: Commit the design spec and implementation plan

- [ ] **Step 5.1: Verify spec and plan files exist**

Run:
```bash
ls -la docs/superpowers/specs/2026-04-17-fork-cleanup-and-sync-design.md
ls -la docs/superpowers/plans/2026-04-17-fork-cleanup-and-sync.md
```

Both must exist. If not, stop.

- [ ] **Step 5.2: Stage and commit**

Run:
```bash
git add docs/superpowers/specs/2026-04-17-fork-cleanup-and-sync-design.md
git add docs/superpowers/plans/2026-04-17-fork-cleanup-and-sync.md
git commit -m "$(cat <<'EOF'
docs: add fork cleanup and upstream sync design spec + plan

Documents the branching model (main/develop/whisperx-upstream),
the reconstruction procedure, and the upstream sync workflow.
EOF
)"
```

- [ ] **Step 5.3: Verify commit**

Run:
```bash
git log --oneline develop -2
```

Expected: tip is the new `docs: add fork cleanup...` commit, parent is `d00ec69`.

---

## Re-apply echo features as clean commits

> **Conflict handling during cherry-picks**: because the new base (`upstream/main` = `d00ec69`) is 5 commits ahead of the old base (`064f737`), some cherry-picks may conflict. When they do:
>
> ```bash
> git status              # see conflicted files
> # edit files, resolve conflicts
> git add <resolved files>
> git cherry-pick --continue
> ```
>
> If a conflict is not obviously resolvable, stop and ask.

### Task 6: C1 — `feat(asr): expose no-speech and repetition quality metrics`

**Files affected:** `whisperx/asr.py`, `whisperx/schema.py`

- [ ] **Step 6.1: Cherry-pick**

Run:
```bash
git cherry-pick f2df4a5
```

- [ ] **Step 6.2: Verify**

Run:
```bash
git log --oneline develop -3
git show --stat HEAD
```

Expected: new HEAD commit message starts with `feat(asr): expose no-speech and repetition quality metrics per segment`, touches `whisperx/asr.py` and `whisperx/schema.py`.

### Task 7: C2 — `chore: rename package to echo-whisperx`

**Files affected:** `pyproject.toml`, `whisperx/alignment.py`

- [ ] **Step 7.1: Cherry-pick**

Run:
```bash
git cherry-pick f91ce44
```

- [ ] **Step 7.2: Verify**

Run:
```bash
git show --stat HEAD
grep -E '^name|^description' pyproject.toml | head -5
```

Expected: tip commit touches only `pyproject.toml` and `whisperx/alignment.py`. `pyproject.toml` name is `echo-whisperx`.

### Task 8: C3 — `feat(asr): add transcribe_multi()` (strip `poetry.lock`)

**Files affected:** `.dockerignore`, `whisperx/asr.py` (NOT `poetry.lock`)

- [ ] **Step 8.1: Cherry-pick without committing**

Run:
```bash
git cherry-pick --no-commit b7ea4f8
```

- [ ] **Step 8.2: Verify `poetry.lock` is staged (to be stripped)**

Run:
```bash
git status --short | grep poetry.lock
```

Expected: `A  poetry.lock` (staged as addition). If not present, skip to Step 8.4.

- [ ] **Step 8.3: Strip `poetry.lock` from the cherry-pick**

Run:
```bash
git reset HEAD -- poetry.lock
rm -f poetry.lock
```

- [ ] **Step 8.4: Confirm only intended files are staged**

Run:
```bash
git status --short
```

Expected staged files (green `A`/`M`): `.dockerignore`, `whisperx/asr.py`. Nothing else.

- [ ] **Step 8.5: Commit with original message**

Run:
```bash
git commit -C b7ea4f8
```

- [ ] **Step 8.6: Verify**

Run:
```bash
git show --stat HEAD
```

Expected: commit message starts with `feat(asr): add transcribe_multi()...`, touches only `.dockerignore` and `whisperx/asr.py`.

### Task 9: C4 — `feat(bench): add optimization benchmark suite` (strip `bench/results/*.json`)

**Files affected:** `bench/__init__.py`, `bench/compare.py`, `bench/config.py`, `bench/data.py`, `bench/harness.py`, `bench/metrics.py`, `bench/run_step.py`, `bench/simulate.py`, `bench/stream_compare.py`, `bench/triton/*`

- [ ] **Step 9.1: Cherry-pick without committing**

Run:
```bash
git cherry-pick --no-commit f0e50c6
```

- [ ] **Step 9.2: Unstage and remove `bench/results/*.json`**

Run:
```bash
git reset HEAD -- 'bench/results/*.json'
rm -f bench/results/*.json
```

- [ ] **Step 9.3: Confirm only intended files are staged**

Run:
```bash
git status --short | grep -c '^A'
ls bench/results/ 2>/dev/null
```

Expected: staged additions count is around 15 (bench python files + triton assets). `bench/results/` is either empty or does not exist.

- [ ] **Step 9.4: Commit with original message**

Run:
```bash
git commit -C f0e50c6
```

- [ ] **Step 9.5: Verify no `bench/results/` file in the commit**

Run:
```bash
git show --stat HEAD | grep 'bench/results' | grep -v 'triton' | head -5
```

Expected: no output (empty).

### Task 10: C5 — `feat(asr): add multi-temperature fallback + KenLM shallow fusion`

**Files affected:** `whisperx/__main__.py`, `whisperx/asr.py`, `whisperx/transcribe.py`, `whisperx/lm_fusion.py`

> Note: `c9aad7b` modifies `asr.py`/`transcribe.py` to import `whisperx.lm_fusion`, but the `lm_fusion.py` file itself was only added in the next commit (`a7a7dfc`). We bundle both so the tree stays coherent.

- [ ] **Step 10.1: Cherry-pick `c9aad7b` without committing**

Run:
```bash
git cherry-pick --no-commit c9aad7b
```

- [ ] **Step 10.2: Add `lm_fusion.py` from archive**

Run:
```bash
git checkout archive/2026-04-pre-cleanup -- whisperx/lm_fusion.py
git add whisperx/lm_fusion.py
```

- [ ] **Step 10.3: Confirm staged files**

Run:
```bash
git status --short
```

Expected: `M whisperx/__main__.py`, `M whisperx/asr.py`, `M whisperx/transcribe.py`, `A whisperx/lm_fusion.py`.

- [ ] **Step 10.4: Commit with original message**

Run:
```bash
git commit -C c9aad7b
```

- [ ] **Step 10.5: Sanity-check the import works (no runtime, just parse)**

Run:
```bash
python -c "import ast; ast.parse(open('whisperx/lm_fusion.py').read()); print('OK')"
```

Expected output: `OK`

### Task 11: C6 — `feat(asr): add transcribe_segment() for externally-segmented audio`

**Files affected:** `whisperx/asr.py` (+99 lines, new method only)

> `a7a7dfc` added a new method `transcribe_segment()` to `asr.py` (99 lines) on top of `c9aad7b`. We extract just that delta.

- [ ] **Step 11.1: Apply only the asr.py delta from `a7a7dfc`**

Run:
```bash
git show a7a7dfc -- whisperx/asr.py | git apply
```

Expected: command succeeds silently. If you see `patch does not apply` or `hunk failed`, stop and resolve manually using `git apply --3way` or by editing `whisperx/asr.py` directly.

- [ ] **Step 11.2: Verify the new method is present**

Run:
```bash
grep -n 'def transcribe_segment' whisperx/asr.py
```

Expected output: one line, e.g. `512:    def transcribe_segment(`

- [ ] **Step 11.3: Stage and commit**

Run:
```bash
git add whisperx/asr.py
git commit -m "$(cat <<'EOF'
feat(asr): add transcribe_segment() for externally-segmented audio

New pipeline method that transcribes a pre-segmented audio chunk
without VAD — designed for callbot use cases where speech boundaries
come from an external end-of-speech detector.  Skips VAD and
alignment; pipes mel-spectrogram through encoder+decoder only.
EOF
)"
```

- [ ] **Step 11.4: Verify commit**

Run:
```bash
git show --stat HEAD
```

Expected: touches only `whisperx/asr.py` (~99 lines added).

### Task 12: C7 — `feat(bench): add KenLM training and evaluation scripts`

**Files affected:** `bench/eval_whisperx.py`, `bench/extract_prod_corpus.py`, `bench/train_kenlm.py`, `CLAUDE.md`

- [ ] **Step 12.1: Checkout the 4 files from archive**

Run:
```bash
git checkout archive/2026-04-pre-cleanup -- \
    bench/eval_whisperx.py \
    bench/extract_prod_corpus.py \
    bench/train_kenlm.py \
    CLAUDE.md
```

- [ ] **Step 12.2: Stage and commit**

Run:
```bash
git add bench/eval_whisperx.py bench/extract_prod_corpus.py bench/train_kenlm.py CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(bench): add KenLM training and evaluation scripts

Utilities for training domain-specific 4-gram KenLMs on French medical
corpora, extracting production transcripts as training data, and
running WER evaluation against the Echo reference manifest.
EOF
)"
```

- [ ] **Step 12.3: Verify**

Run:
```bash
git show --stat HEAD
```

Expected: touches 4 files (3 bench scripts + `CLAUDE.md`).

### Task 13: C8 — `chore: update .gitignore and track tests/`

**Files affected:** `.gitignore`, `tests/__init__.py`, `tests/test_fallback.py`, `tests/test_lm_fusion.py`, `tests/test_transcribe_segment.py`

- [ ] **Step 13.1: Edit `.gitignore`**

Open `.gitignore` and apply these changes:

1. **Remove** the line:
   ```
   tests/
   ```

2. **Append** the following block at the end of the file:
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

- [ ] **Step 13.2: Verify `tests/` is no longer ignored**

Run:
```bash
git check-ignore tests/test_fallback.py
```

Expected output: nothing (exit code 1 = not ignored). If the path is still reported as ignored, revisit `.gitignore`.

- [ ] **Step 13.3: Stage tests and .gitignore**

Run:
```bash
git add .gitignore
git add tests/__init__.py tests/test_fallback.py tests/test_lm_fusion.py tests/test_transcribe_segment.py
```

- [ ] **Step 13.4: Confirm staged set**

Run:
```bash
git status --short
```

Expected: `M .gitignore`, `A tests/__init__.py`, `A tests/test_fallback.py`, `A tests/test_lm_fusion.py`, `A tests/test_transcribe_segment.py`.

- [ ] **Step 13.5: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: update .gitignore and version tests directory

- Stop ignoring tests/ so the existing fallback, LM-fusion and
  transcribe_segment tests are versioned.
- Ignore bench/results/ artifacts, KenLM training corpora
  (models/corpus_*.txt), scratch explorations, and poetry.lock
  (project uses uv).
EOF
)"
```

### Task 14: C9 — `docs: document development and upstream sync workflow`

**Files affected:** `README.md`

- [ ] **Step 14.1: Locate insertion point**

Run:
```bash
grep -n '^## ' README.md | head -20
```

Note the heading list. The new `## Development workflow` section will be inserted **immediately before** the `## Setup` section if present, otherwise after the project introduction. Pick the line number for insertion.

- [ ] **Step 14.2: Add the section**

Insert the following block in `README.md` at the chosen location:

````markdown
## Development workflow

### Branches

- `main` — current echo release (what runs in prod)
- `develop` — integration of in-progress features
- `feature/<name>` — individual features, branched from `develop`
- `archive/<date>-<reason>` — frozen history, reference only

### Syncing with upstream whisperX

This fork tracks [`m-bain/whisperX`](https://github.com/m-bain/whisperX) via the `upstream` remote.

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

- [ ] **Step 14.3: Stage and commit**

Run:
```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: document development and upstream sync workflow

Add a "Development workflow" section describing the branching model
(main/develop/feature/archive) and the recurring upstream-sync
procedure using the local whisperx-upstream mirror branch.
EOF
)"
```

---

## Local verification

### Task 15: Structural checks

- [ ] **Step 15.1: Verify commit count on develop**

Run:
```bash
git log main..develop --oneline | wc -l
```

Expected output: `10` (C0 spec/plan + C1…C9).

- [ ] **Step 15.2: Inspect the history**

Run:
```bash
git log --oneline main..develop
```

Expected: 10 commits, in order:
```
docs: document development and upstream sync workflow
chore: update .gitignore and version tests directory
feat(bench): add KenLM training and evaluation scripts
feat(asr): add transcribe_segment() for externally-segmented audio
feat(asr): add multi-temperature fallback and LM fusion rescoring
feat(bench): add optimization benchmark suite for parallel call transcription
feat(asr): add transcribe_multi() for cross-call batched transcription
chore: rename package to echo-whisperx and fix missing newline
feat(asr): expose no-speech and repetition quality metrics per segment
docs: add fork cleanup and upstream sync design spec + plan
```

(Most-recent first.)

- [ ] **Step 15.3: Verify no noise files are introduced**

Run:
```bash
git diff --name-only main..develop | grep -E '(corpus_.*\.txt|bench/results/|poetry\.lock|ANALYSE_ET_AMELIORATIONS|OPTIMISATIONS_AVANCEES|\.arpa$|\.bin$)' | head
```

Expected: no output (empty). If anything prints, stop and investigate.

- [ ] **Step 15.4: Diff size sanity**

Run:
```bash
git diff main..develop --shortstat
```

Expected: insertions in the low thousands of lines (not millions). Ballpark: 2000–5000 lines inserted.

### Task 16: Environment and tests

- [ ] **Step 16.1: Regenerate virtualenv**

Run:
```bash
uv sync
```

Expected: no errors. If `uv sync` fails, capture the error and stop.

- [ ] **Step 16.2: Run tests**

Run:
```bash
uv run pytest tests/ -v
```

Expected: all tests pass. If any fail, capture the failure and stop.

- [ ] **Step 16.3: Baseline WER regression check (optional but recommended)**

Run:
```bash
uv run python bench/eval_whisperx.py --label post-cleanup
```

Expected: Norm WER ≈ 35.4 % (baseline from `CLAUDE.md`), within ±0.5 pt. If the WER drifts substantially, stop and investigate before pushing.

---

## Push and cleanup `origin`

> **Point of no return from here** — only proceed after verification passes.

### Task 17: Push `main` and `develop`

- [ ] **Step 17.1: Push both branches**

Run:
```bash
git push origin main
git push origin develop
```

- [ ] **Step 17.2: Verify on GitHub**

Run:
```bash
gh api repos/E-Sensia/echo-whisperX/branches --jq '.[].name' | sort
```

Expected: the output contains at minimum `main`, `develop`, `archive/2026-04-pre-cleanup`, plus the old branches (to be deleted in Task 19).

### Task 18: Change default branch to `main`

- [ ] **Step 18.1: Patch the repo default branch**

Run:
```bash
gh api -X PATCH repos/E-Sensia/echo-whisperX -f default_branch=main
```

Expected: JSON response with `"default_branch": "main"`.

- [ ] **Step 18.2: Verify**

Run:
```bash
gh api repos/E-Sensia/echo-whisperX --jq '.default_branch'
```

Expected output: `main`

### Task 19: Delete obsolete remote branches

- [ ] **Step 19.1: Delete branches on `origin`**

Run each line separately and confirm it succeeds before the next:

```bash
git push origin --delete whisperx_improvement_explo
git push origin --delete bench/optimizations
git push origin --delete echo-updates
git push origin --delete improve-code
git push origin --delete stable
```

Expected: each command prints `- [deleted] <branch>`.

- [ ] **Step 19.2: Verify final origin branch list**

Run:
```bash
gh api repos/E-Sensia/echo-whisperX/branches --jq '.[].name' | sort
```

Expected output (exactly):
```
archive/2026-04-pre-cleanup
develop
main
```

### Task 20: Delete obsolete local branches

- [ ] **Step 20.1: Verify archive is safely on origin before deleting locals**

Run:
```bash
git ls-remote origin archive/2026-04-pre-cleanup
```

Expected: one line with a commit SHA. If empty, stop — archive is not on origin.

- [ ] **Step 20.2: Delete local branches**

Run:
```bash
git branch -D upstream-main
git branch -D bench/optimizations
git branch -D echo-updates
git branch -D whisperx_improvement_explo
```

Expected: each line prints `Deleted branch <name> (was <sha>)`.

- [ ] **Step 20.3: Verify final local branch list**

Run:
```bash
git branch
```

Expected output (exactly):
```
  archive/2026-04-pre-cleanup
* develop
  main
  whisperx-upstream
```

### Task 21: Disk cleanup (optional)

- [ ] **Step 21.1: Remove stale egg-info**

Run:
```bash
rm -rf whisperx.egg-info/
```

- [ ] **Step 21.2: Verify both egg-infos are gitignored**

Run:
```bash
git check-ignore whisperx.egg-info/ echo_whisperx.egg-info/ 2>/dev/null || true
git status --short | grep egg-info
```

Expected: second command produces no output.

- [ ] **Step 21.3: Confirm working tree is clean on tracked files**

Run:
```bash
git status
```

Expected: only untracked (gitignored) paths should remain: `explorations/`, `last_name_*.json`, etc. No tracked modifications.

---

## Done

At this point:

- `origin` has exactly 3 branches: `main` (= `upstream/main + 10 echo commits`), `develop` (= `main`), `archive/2026-04-pre-cleanup` (historical reference).
- Default branch on GitHub is `main`.
- Local working copy is on `develop`, ready for new `feature/<nom>` branches.
- The upstream sync workflow is documented in `README.md`.
- All tests pass; WER baseline is preserved.
- Safety tag `backup/2026-04-17-pre-cleanup` is still in place locally (can be deleted once you're confident: `git tag -d backup/2026-04-17-pre-cleanup`).

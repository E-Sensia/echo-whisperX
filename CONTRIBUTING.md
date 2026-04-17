# Contributing to echo-whisperX

This project is a fork of [`m-bain/whisperX`](https://github.com/m-bain/whisperX)
maintained by E-Sensia for Echo's telephony ASR pipeline (French medical
emergency calls, PCMU 8 kHz upsampled to 16 kHz).

The goal of the fork is to track upstream closely while accumulating
Echo-specific features (multi-temperature fallback, KenLM shallow fusion,
cross-call batched transcription, externally-segmented transcription,
domain benchmarks). This document describes the conventions that keep the
fork maintainable.

---

## 1. Branch model

We use a git-flow layout adapted for a fork.

| Branch | Role | Who updates it |
|---|---|---|
| `main` | Current echo release — what runs in production. Only tagged release commits land here. | Release merges from `develop`, tagged `vX.Y.Z-echo.N`. |
| `develop` | Integration of in-progress features. Default landing branch for feature work. | Merges from `feature/*` and `whisperx-upstream`. |
| `feature/<name>` | Individual feature or fix. Short-lived. | Opened from `develop`, merged back into `develop`. |
| `archive/<date>-<reason>` | Frozen snapshot of historical work. Never merged. | Created once, never updated. |
| `whisperx-upstream` (local only) | Mirror of `upstream/main`. **Never pushed to origin.** | Fast-forwarded from `upstream/main` during sync. |

### Remotes

```
origin    git@github.com:E-Sensia/echo-whisperX.git   # our fork
upstream  https://github.com/m-bain/whisperX.git      # upstream (fetch-only)
```

`upstream` is configured with push disabled so a stray `git push` cannot
reach m-bain's repo.

---

## 2. Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <summary in imperative mood, max ~72 chars>

<optional body: motivation, context, tradeoffs>

<optional footer: Co-Authored-By, Refs, etc.>
```

### Types

| Type | Use for |
|---|---|
| `feat` | New user-visible feature |
| `fix` | Bug fix |
| `perf` | Performance improvement with no behavior change |
| `refactor` | Code restructuring with no behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation only (README, CONTRIBUTING, comments, docstrings) |
| `chore` | Tooling, config, deps, release mechanics |
| `ci` | CI pipeline changes |

### Scopes (common)

`asr`, `align`, `diarize`, `bench`, `vad`, `transcribe`, `schema`, `lm`.
Add new scopes as they become relevant. Use lowercase.

### Examples

```
feat(asr): expose no-speech and repetition quality metrics per segment
feat(bench): add KenLM training and evaluation scripts
fix(align): restore word-level timestamps for unalignable characters
chore: rename package to echo-whisperx
docs: document development and upstream sync workflow
```

Keep each commit atomic — one logical change per commit, tests passing
at every commit, file set coherent with the summary.

---

## 3. Versioning

We tag releases as:

```
vX.Y.Z-echo.N
```

- `X.Y.Z` — the underlying upstream `m-bain/whisperX` version (e.g. `3.8.5`).
- `N` — echo iteration on top of that upstream base. Starts at `0`, bumped
  at each new echo release.

### Examples

| Tag | Meaning |
|---|---|
| `v3.8.5-echo.0` | First echo release on top of upstream `v3.8.5`. |
| `v3.8.5-echo.1` | Second echo release, still on upstream `v3.8.5`. |
| `v3.9.0-echo.0` | Resync to upstream `v3.9.0`, first echo release after the resync. |

`pyproject.toml`'s `version` field tracks the **underlying upstream
version** only (e.g. `3.8.5`). The `-echo.N` suffix is carried by the git
tag. This keeps `pip` / `uv` resolvers happy and makes it obvious which
upstream we're based on.

---

## 4. Upstream sync workflow

Upstream ships patches regularly. We integrate them through a local
`whisperx-upstream` mirror branch that never leaves the developer's
machine. This keeps origin clean and gives us a sas to verify upstream
before touching `develop`.

```bash
# Fetch everything new from m-bain/whisperX
git fetch upstream --tags

# Advance the local mirror (fast-forward only)
git checkout whisperx-upstream
git merge --ff-only upstream/main

# Integrate into develop
git checkout develop
git merge whisperx-upstream

# Resolve any conflicts, then verify
uv sync
uv run python -m pytest tests/

# If everything is green, publish
git push origin develop
```

### Conflict resolution

Most upstream conflicts happen in `whisperx/asr.py`, `whisperx/alignment.py`
and `pyproject.toml`. Rules of thumb:

- **`pyproject.toml`** — keep `name = "echo-whisperx"` and the E-Sensia
  repository URL, prefer the upstream version field.
- **`whisperx/asr.py`** — echo additions (multi-temperature fallback, LM
  fusion, `transcribe_multi`, `transcribe_segment`) must be preserved;
  if a signature shifts upstream, adapt the echo additions, don't drop
  them.
- **`whisperx/alignment.py`** — prefer upstream fixes; our delta here is
  minimal.

If in doubt, pause and open a PR so the conflict resolution gets
reviewed before landing on `develop`.

---

## 5. Release workflow

A release is a merge from `develop` into `main` plus a version tag.

```bash
# Sanity check
git checkout develop
uv run python -m pytest tests/

git checkout main
git merge --ff-only develop          # fast-forward when possible

# Pick the next tag per the versioning rules
git tag -a vX.Y.Z-echo.N -m "Release notes…"

git push origin main
git push origin vX.Y.Z-echo.N
```

`--ff-only` is the default so `main` stays linear. Fall back to
`--no-ff` (or a merge PR via GitHub) if you need an explicit merge
commit for audit.

---

## 6. Feature workflow

```bash
git checkout develop
git pull --ff-only origin develop
git checkout -b feature/<short-descriptive-name>

# …code, commit…
uv run python -m pytest tests/

git push -u origin feature/<short-descriptive-name>
# Open a PR feature/<name> → develop on GitHub
```

Squash or rebase-merge at your discretion — keep `develop` readable.

---

## 7. Tests and quality gates

- **Required before merge to `develop`**: `uv run python -m pytest tests/`
  passes locally. Add new tests for new behavior.
- **Recommended after touching ASR code**: run
  `uv run python bench/eval_whisperx.py --label <change-name>` and
  compare WER to the documented baseline in `CLAUDE.md` (currently
  35.4 % norm WER on 160-segment Echo reference set).
- **Tests live in `tests/`** and are versioned. Do not re-ignore the
  directory.
- Run `pytest` via `uv run python -m pytest …` (not bare `pytest`) to
  avoid conflicts with `pyenv` shims.

---

## 8. What never goes into the repo

These are enforced by `.gitignore` and should not be unignored:

| Path | Reason |
|---|---|
| `models/corpus_*.txt` | Training corpora (often tens of MB). Kept outside the repo. |
| `models/*.arpa`, `models/*.bin` | Compiled KenLM models (hundreds of MB). |
| `bench/results/` | Per-run benchmark artefacts. |
| `explorations/`, `last_name_*.json` | Developer-local scratch. |
| `poetry.lock` | Project standardises on `uv`; no dual lockfiles. |
| `*.egg-info/` | Build metadata. |
| `.venv/` | Virtual environments. |

If you need to share a corpus or a KenLM blob, put it in object storage
and document the path; do not commit it.

---

## 9. GitHub repository settings (for maintainers)

- **Default branch**: `main`.
- **Organization rulesets** apply to `main` (pull request required,
  previously verified signatures required). Upstream commits are not
  signed by echo maintainers, so the "verified signatures" rule must be
  off or bypass-able for the maintainers performing the upstream sync.
  Configure a **Bypass list** on the ruleset rather than disabling it
  globally.
- **Dependabot alerts** are enabled; triage on the security tab.

---

## 10. Further reading

- `README.md` — project overview, installation, basic usage.
- `CLAUDE.md` — how to run the ASR quality benchmark.
- `docs/superpowers/specs/` — design specs (e.g. the April 2026 cleanup spec).
- `docs/superpowers/plans/` — implementation plans.
- Upstream docs: <https://github.com/m-bain/whisperX>.

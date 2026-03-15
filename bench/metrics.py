"""WER computation and quality metrics aggregation."""

import re
import string
from typing import Optional


# ── Text normalization ──────────────────────────────────────────────────────

# Keep apostrophes (important for French: l'homme, j'ai, n'a, etc.)
_PUNCT_TO_REMOVE = re.compile(
    r"[" + re.escape(string.punctuation.replace("'", "").replace("'", "")) + r"]"
)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison: lowercase, strip punctuation, collapse spaces."""
    text = text.lower()
    text = _PUNCT_TO_REMOVE.sub(" ", text)
    text = text.replace("\u2019", "'")  # right single quotation mark → apostrophe
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ── WER (Levenshtein on word lists) ────────────────────────────────────────

def _levenshtein(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int]:
    """Compute edit distance and return (substitutions, deletions, insertions)."""
    n, m = len(ref_words), len(hyp_words)
    # dp[i][j] = (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                delete = dp[i - 1][j]
                insert = dp[i][j - 1]

                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                del_cost = (delete[0] + 1, delete[1], delete[2] + 1, delete[3])
                ins_cost = (insert[0] + 1, insert[1], insert[2], insert[3] + 1)

                dp[i][j] = min(sub_cost, del_cost, ins_cost, key=lambda x: x[0])

    return dp[n][m][1], dp[n][m][2], dp[n][m][3]


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between two texts.

    Returns dict with: wer, substitutions, deletions, insertions, ref_words.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return {
            "wer": 0.0 if not hyp_words else 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "ref_words": 0,
        }

    subs, dels, ins = _levenshtein(ref_words, hyp_words)
    wer = (subs + dels + ins) / len(ref_words)

    return {
        "wer": wer,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_words": len(ref_words),
    }


# ── Quality metrics aggregation ────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: float) -> float:
    """Simple percentile on a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def aggregate_quality_metrics(segments: list[dict]) -> dict:
    """Compute mean/median/p10/p90 for quality metrics across segments."""
    metric_names = [
        "avg_logprob", "no_speech_prob", "compression_ratio",
        "tokens_per_sec", "repeat_adjacent_frac", "repeat_unique_frac",
    ]
    result = {}
    for name in metric_names:
        values = [s[name] for s in segments if name in s and s[name] is not None]
        if not values:
            result[name] = {"mean": 0, "median": 0, "p10": 0, "p90": 0}
            continue
        values.sort()
        n = len(values)
        result[name] = {
            "mean": sum(values) / n,
            "median": values[n // 2],
            "p10": _percentile(values, 0.1),
            "p90": _percentile(values, 0.9),
        }
    return result

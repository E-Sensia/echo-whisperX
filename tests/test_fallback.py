"""Tests for multi-temperature fallback in generate_segment_batched."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

from whisperx.asr import (
    WhisperModel,
    _needs_fallback,
    _select_best_hypothesis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeGenResult:
    """Mimics ctranslate2.models.WhisperGenerationResult."""

    def __init__(self, sequences_ids, scores, no_speech_prob=0.0):
        self.sequences_ids = sequences_ids
        self.scores = scores
        self.no_speech_prob = no_speech_prob


class FakeOptions:
    """Minimal stand-in for TranscriptionOptions (only fields used by helpers)."""

    def __init__(self, **overrides):
        defaults = dict(
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            beam_size=5,
            best_of=5,
            patience=1,
            length_penalty=1,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            suppress_blank=True,
            suppress_tokens=[-1],
            no_repeat_ngram_size=0,
            repetition_penalty=1,
            without_timestamps=True,
            prefix=None,
            hotwords=None,
            initial_prompt=None,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


def _make_tokenizer(eot=50257, decode_fn=None):
    """Return a mock tokenizer."""
    tok = MagicMock()
    tok.eot = eot
    tok.encode.return_value = []
    if decode_fn is not None:
        tok.tokenizer.decode_batch.side_effect = decode_fn
    else:
        tok.tokenizer.decode_batch.side_effect = lambda batch: [
            "".join(str(t) for t in tokens) for tokens in batch
        ]
    return tok


# ---------------------------------------------------------------------------
# _needs_fallback
# ---------------------------------------------------------------------------

class TestNeedsFallback:
    def test_good_sample(self):
        opts = FakeOptions()
        assert not _needs_fallback(1.5, -0.5, 0.1, opts)

    def test_high_compression_ratio(self):
        opts = FakeOptions()
        assert _needs_fallback(3.0, -0.5, 0.1, opts)

    def test_low_avg_logprob(self):
        opts = FakeOptions()
        assert _needs_fallback(1.5, -1.5, 0.1, opts)

    def test_both_bad(self):
        opts = FakeOptions()
        assert _needs_fallback(3.0, -1.5, 0.1, opts)

    def test_silence_no_fallback(self):
        opts = FakeOptions()
        # High no_speech_prob + low avg_logprob = silence
        assert not _needs_fallback(1.5, -1.5, 0.8, opts)

    def test_thresholds_disabled(self):
        opts = FakeOptions(
            compression_ratio_threshold=None,
            log_prob_threshold=None,
        )
        assert not _needs_fallback(100.0, -100.0, 0.0, opts)

    def test_only_compression_threshold(self):
        opts = FakeOptions(log_prob_threshold=None)
        assert _needs_fallback(3.0, -100.0, 0.0, opts)
        assert not _needs_fallback(2.0, -100.0, 0.0, opts)

    def test_only_logprob_threshold(self):
        opts = FakeOptions(compression_ratio_threshold=None)
        assert _needs_fallback(100.0, -1.5, 0.1, opts)
        assert not _needs_fallback(100.0, -0.5, 0.1, opts)


# ---------------------------------------------------------------------------
# _select_best_hypothesis
# ---------------------------------------------------------------------------

class TestSelectBestHypothesis:
    def test_single_hypothesis_no_lm(self):
        res = FakeGenResult(
            sequences_ids=[[10, 20, 30]],
            scores=[-0.3],
        )
        opts = FakeOptions(length_penalty=1)
        tok = _make_tokenizer()

        tokens, avg_lp = _select_best_hypothesis(res, tok, opts, lm_fusion=None)
        assert tokens == [10, 20, 30]
        # cum = -0.3 * 3^1 = -0.9, avg = -0.9 / 4 = -0.225
        assert pytest.approx(avg_lp) == -0.225

    def test_multiple_hypotheses_no_lm_picks_best(self):
        res = FakeGenResult(
            sequences_ids=[[10, 20], [30, 40, 50]],
            scores=[-0.5, -0.2],  # second has better score
        )
        opts = FakeOptions(length_penalty=1)
        tok = _make_tokenizer()

        tokens, avg_lp = _select_best_hypothesis(res, tok, opts, lm_fusion=None)
        # hyp 0: cum=-0.5*2=-1.0, avg=-1.0/3=-0.333
        # hyp 1: cum=-0.2*3=-0.6, avg=-0.6/4=-0.15  (better)
        assert tokens == [30, 40, 50]
        assert pytest.approx(avg_lp) == -0.15

    def test_with_lm_fusion_rescoring(self):
        res = FakeGenResult(
            sequences_ids=[[10, 20], [30, 40]],
            scores=[-0.5, -0.8],  # first has better acoustic
        )
        opts = FakeOptions(length_penalty=1)
        tok = _make_tokenizer()

        lm = MagicMock()
        lm.rescore.return_value = 1  # LM prefers the second hypothesis
        lm.num_hypotheses = 5

        tokens, avg_lp = _select_best_hypothesis(res, tok, opts, lm_fusion=lm)
        assert tokens == [30, 40]
        lm.rescore.assert_called_once()


# ---------------------------------------------------------------------------
# generate_segment_batched integration
# ---------------------------------------------------------------------------

class TestGenerateSegmentBatchedFallback:
    """Test the full fallback path by calling the real method on a mock self."""

    def _make_model_mock(self):
        model = MagicMock()
        model.max_length = 448
        model.lm_fusion = None
        model.get_prompt.return_value = [50258, 50259, 50359]
        model.encode.return_value = MagicMock(name="encoder_output")
        return model

    def test_no_fallback_single_temperature(self):
        """Single temperature → no fallback logic, behaves like original."""
        model = self._make_model_mock()
        opts = FakeOptions(temperatures=[0.0])

        good = FakeGenResult(
            sequences_ids=[[100, 200]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )
        model.model.generate.return_value = [good]

        features = np.zeros((1, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        assert len(result["text"]) == 1
        assert len(result["avg_logprob"]) == 1
        assert model.model.generate.call_count == 1  # no fallback call

    def test_fallback_triggered_and_settles(self):
        """Bad sample triggers fallback, second temperature settles."""
        model = self._make_model_mock()
        opts = FakeOptions(
            temperatures=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        # Initial pass: bad result (avg_logprob too low)
        bad_initial = FakeGenResult(
            sequences_ids=[[400, 400, 400]],
            scores=[-2.0],
            no_speech_prob=0.1,
        )
        # Fallback at temp 0.2: good result
        good_fallback = FakeGenResult(
            sequences_ids=[[500, 600]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )

        model.model.generate.side_effect = [
            [bad_initial],    # batch pass
            [good_fallback],  # single-sample fallback
        ]

        features = np.zeros((1, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        assert model.model.generate.call_count == 2
        # The fallback result's tokens should be used
        assert result["tokens"][0] == [500, 600]

    def test_silence_not_retried(self):
        """High no_speech_prob + low avg_logprob = silence → no fallback."""
        model = self._make_model_mock()
        opts = FakeOptions(
            temperatures=[0.0, 0.2],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        silence = FakeGenResult(
            sequences_ids=[[100]],
            scores=[-2.0],
            no_speech_prob=0.9,  # very likely silence
        )
        model.model.generate.return_value = [silence]

        features = np.zeros((1, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        # Only the initial call — no fallback for silence
        assert model.model.generate.call_count == 1

    def test_good_sample_in_batch_not_retried(self):
        """In a 2-sample batch, only the bad sample triggers fallback."""
        model = self._make_model_mock()
        opts = FakeOptions(
            temperatures=[0.0, 0.2],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        good = FakeGenResult(
            sequences_ids=[[100, 200]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )
        bad = FakeGenResult(
            sequences_ids=[[400, 400, 400]],
            scores=[-2.0],
            no_speech_prob=0.1,
        )
        fixed = FakeGenResult(
            sequences_ids=[[500, 600]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )

        model.model.generate.side_effect = [
            [good, bad],   # batch pass
            [fixed],       # fallback for sample 1 only
        ]

        features = np.zeros((2, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        assert model.model.generate.call_count == 2
        assert result["tokens"][0] == [100, 200]  # good unchanged
        assert result["tokens"][1] == [500, 600]   # bad → fixed

    def test_all_temps_exhausted_picks_best(self):
        """When all temperatures fail, pick the best avg_logprob."""
        model = self._make_model_mock()
        opts = FakeOptions(
            temperatures=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        bad_initial = FakeGenResult(
            sequences_ids=[[400, 400, 400]],
            scores=[-2.0],
            no_speech_prob=0.1,
        )
        # Both fallbacks still bad, but second is slightly better
        still_bad_1 = FakeGenResult(
            sequences_ids=[[410, 410]],
            scores=[-1.8],
            no_speech_prob=0.1,
        )
        still_bad_2 = FakeGenResult(
            sequences_ids=[[420, 420]],
            scores=[-1.5],  # best of the lot
            no_speech_prob=0.1,
        )

        model.model.generate.side_effect = [
            [bad_initial],
            [still_bad_1],
            [still_bad_2],
        ]

        features = np.zeros((1, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        # All 3 calls: initial + 2 fallbacks
        assert model.model.generate.call_count == 3
        # Should pick the best among the three
        assert result["tokens"][0] in (
            [400, 400, 400], [410, 410], [420, 420]
        )

    def test_no_regression_without_features(self):
        """When thresholds are None, behavior is identical to old code."""
        model = self._make_model_mock()
        opts = FakeOptions(
            temperatures=[0.0, 0.2, 0.4],
            compression_ratio_threshold=None,
            log_prob_threshold=None,
        )

        res = FakeGenResult(
            sequences_ids=[[10, 20, 30]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )
        model.model.generate.return_value = [res]

        features = np.zeros((1, 80, 3000))
        tok = _make_tokenizer()

        result = WhisperModel.generate_segment_batched(
            model, features, tok, opts,
        )

        # No fallback even with multiple temps — thresholds disabled
        assert model.model.generate.call_count == 1
        assert result["tokens"][0] == [10, 20, 30]

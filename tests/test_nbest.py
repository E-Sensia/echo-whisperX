"""Tests for N-best hypothesis exposure in generate_segment_batched.

Exercises the ``n_best`` parameter added to WhisperModel.generate_segment_batched
and its propagation through FasterWhisperPipeline.transcribe_segment. Uses
mocks (same pattern as test_fallback.py) to keep the suite fast and
deterministic; no GPU or real model required.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from whisperx.asr import WhisperModel


class FakeGenResult:
    """Mimics ctranslate2.models.WhisperGenerationResult."""

    def __init__(self, sequences_ids, scores, no_speech_prob=0.0):
        self.sequences_ids = sequences_ids
        self.scores = scores
        self.no_speech_prob = no_speech_prob


class FakeOptions:
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


def _make_tokenizer(eot=50257):
    tok = MagicMock()
    tok.eot = eot
    tok.encode.return_value = []
    # Decode each token id to its string form so we can assert contents.
    tok.tokenizer.decode_batch.side_effect = lambda batch: [
        "".join(chr(65 + (t % 26)) for t in tokens) for tokens in batch
    ]
    tok.tokenizer.decode.side_effect = lambda tokens: "".join(
        chr(65 + (t % 26)) for t in tokens
    )
    return tok


def _stub_model(fake_result):
    """Build a bare WhisperModel stub exposing the fields used by
    generate_segment_batched, without touching CTranslate2."""
    stub = WhisperModel.__new__(WhisperModel)
    stub.model = MagicMock()
    stub.model.generate.return_value = fake_result
    stub.max_length = 448
    stub.encode = MagicMock(return_value="fake_encoder_output")
    # get_prompt is inherited from faster_whisper.WhisperModel — stub it too.
    stub.get_prompt = MagicMock(return_value=[1, 2, 3])
    stub.lm_fusion = None
    return stub


def _fake_features(batch_size=1):
    # Shape [batch, n_mels, n_frames]
    return np.zeros((batch_size, 80, 3000), dtype=np.float32)


class TestNBestBackwardCompat:
    """Default behaviour (n_best=1) must not change."""

    def test_default_no_hypotheses_key(self):
        result = [FakeGenResult(
            sequences_ids=[[10, 20, 30]],
            scores=[-0.3],
            no_speech_prob=0.05,
        )]
        model = _stub_model(result)
        out = model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
        )
        assert "hypotheses" not in out
        assert out["text"][0] == "KUE"  # ids 10,20,30 → 'K','U','E'

    def test_default_requests_num_hypotheses_1(self):
        result = [FakeGenResult([[10]], [-0.5])]
        model = _stub_model(result)
        model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
        )
        call_kwargs = model.model.generate.call_args.kwargs
        assert call_kwargs["num_hypotheses"] == 1


class TestNBestExposed:
    """When n_best > 1, hypotheses are returned and sorted by avg_logprob."""

    def test_returns_hypotheses_key(self):
        # Three hypotheses with distinct scores
        result = [FakeGenResult(
            sequences_ids=[[10, 20], [11, 21], [12, 22]],
            scores=[-0.1, -0.3, -0.5],
        )]
        model = _stub_model(result)
        out = model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
            n_best=3,
        )
        assert "hypotheses" in out
        hyps = out["hypotheses"][0]
        assert len(hyps) == 3
        # Scores are length-normalised; the ranking by avg_logprob should
        # still put the best cum_score first for equal-length sequences.
        for h in hyps:
            assert "text" in h
            assert "avg_logprob" in h
            assert isinstance(h["avg_logprob"], float)

    def test_hypotheses_preserve_ctranslate2_order(self):
        # In real usage CTranslate2 returns sequences_ids/scores already in
        # descending score order. We just pass them through, so the output
        # order tracks the input order of the FakeGenResult.
        result = [FakeGenResult(
            sequences_ids=[[5, 6], [1, 2], [3, 4]],  # caller-provided order
            scores=[-0.05, -0.2, -0.5],              # matches: desc
        )]
        model = _stub_model(result)
        out = model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
            n_best=3,
        )
        lps = [h["avg_logprob"] for h in out["hypotheses"][0]]
        # For equal-length sequences the avg_lp order equals the score order.
        assert lps == sorted(lps, reverse=True)

    def test_bumps_beam_size_and_num_hypotheses(self):
        result = [FakeGenResult(
            sequences_ids=[[i] for i in range(8)],
            scores=[-0.1 * (i + 1) for i in range(8)],
        )]
        model = _stub_model(result)
        model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(),
            FakeOptions(beam_size=5),  # lower than n_best
            n_best=8,
        )
        call_kwargs = model.model.generate.call_args.kwargs
        # beam_size bumped up so CTranslate2 has enough beams.
        assert call_kwargs["beam_size"] >= 8
        assert call_kwargs["num_hypotheses"] >= 8

    def test_respects_n_best_cap(self):
        # Model returns 8 hypotheses but caller requested 3: we only keep 3.
        result = [FakeGenResult(
            sequences_ids=[[i] for i in range(8)],
            scores=[-0.1 * (i + 1) for i in range(8)],
        )]
        model = _stub_model(result)
        out = model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(),
            FakeOptions(beam_size=8),
            n_best=3,
        )
        assert len(out["hypotheses"][0]) == 3


class TestNBestSkipsFallback:
    """When n_best > 1, the multi-temperature fallback must be skipped."""

    def test_fallback_skipped(self):
        # Craft a result that WOULD trigger fallback (low avg_logprob).
        # cum_score -5.0 on 2 tokens → avg_lp = -5.0*2 / 3 = -3.33, below -1.0.
        bad_result = [FakeGenResult(
            sequences_ids=[[10, 20], [11, 21]],
            scores=[-5.0, -4.0],
            no_speech_prob=0.05,
        )]
        model = _stub_model(bad_result)

        # With n_best=1 the fallback should fire (model.generate called more
        # than once as temperatures are tried).
        model.model.generate.reset_mock()
        model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
            n_best=1,
        )
        default_calls = model.model.generate.call_count

        # With n_best=2 the fallback should be skipped entirely — generate
        # is called exactly once (the initial beam search).
        model.model.generate.reset_mock()
        model.generate_segment_batched(
            _fake_features(1), _make_tokenizer(), FakeOptions(),
            n_best=2,
        )
        assert model.model.generate.call_count == 1, (
            f"Expected a single generate call with n_best=2 "
            f"(default had {default_calls} calls for the same bad input), "
            f"got {model.model.generate.call_count}"
        )

"""Tests for KenLM N-best rescoring (shallow fusion)."""

import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from whisperx.asr import WhisperModel, _select_best_hypothesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeGenResult:
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
            temperatures=[0.0],
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
    tok.tokenizer.decode_batch.side_effect = lambda batch: [
        f"text_{i}" for i in range(len(batch))
    ]
    return tok


# ---------------------------------------------------------------------------
# KenLMFusion unit tests (mocked kenlm)
# ---------------------------------------------------------------------------

class TestKenLMFusion:
    @patch.dict("sys.modules", {"kenlm": MagicMock()})
    def _make_fusion(self, scores_map=None, lm_weight=0.5):
        """Create a KenLMFusion with a mocked kenlm backend."""
        import sys
        mock_kenlm = sys.modules["kenlm"]

        mock_model = MagicMock()
        mock_model.order = 3
        # Default: return 0.0 for all texts
        if scores_map is not None:
            mock_model.score.side_effect = lambda text, **kw: scores_map.get(
                text.strip(), 0.0
            )
        else:
            mock_model.score.return_value = 0.0
        mock_kenlm.Model.return_value = mock_model

        from whisperx.lm_fusion import KenLMFusion
        fusion = KenLMFusion.__new__(KenLMFusion)
        fusion.model = mock_model
        fusion.lm_weight = lm_weight
        fusion.num_hypotheses = 5
        return fusion

    def test_single_hypothesis(self):
        fusion = self._make_fusion()
        assert fusion.rescore(["hello"], [-0.5]) == 0

    def test_reranking_by_lm(self):
        # LM strongly prefers "la tension" over "l'attention"
        fusion = self._make_fusion(
            scores_map={
                "la tension": -1.0,     # good LM score (log10)
                "l'attention": -5.0,    # bad LM score (log10)
            },
            lm_weight=0.5,
        )
        # Acoustic: "l'attention" slightly better
        idx = fusion.rescore(
            ["l'attention", "la tension"],
            [-0.4, -0.5],  # acoustic scores (ln)
        )
        # LM boost should flip the ranking
        assert idx == 1  # "la tension" wins

    def test_zero_weight_preserves_acoustic(self):
        fusion = self._make_fusion(
            scores_map={"bad": -10.0, "good": 0.0},
            lm_weight=0.0,
        )
        idx = fusion.rescore(
            ["good", "bad"],
            [-0.3, -0.8],
        )
        assert idx == 0  # acoustic ranking preserved

    def test_score_text_converts_log10_to_ln_per_word(self):
        fusion = self._make_fusion()
        fusion.model.score.return_value = -6.0  # log10 total for 3 words
        score = fusion.score_text("one two three")
        # -6.0 / 3 words * ln(10)
        expected = (-6.0 / 3) * math.log(10)
        assert pytest.approx(score) == expected

    def test_empty_hypotheses(self):
        fusion = self._make_fusion()
        assert fusion.rescore([], []) == 0


# ---------------------------------------------------------------------------
# Integration: _select_best_hypothesis with LM fusion
# ---------------------------------------------------------------------------

class TestSelectBestWithLM:
    def _make_lm(self, preferred_idx=0):
        lm = MagicMock()
        lm.rescore.return_value = preferred_idx
        lm.num_hypotheses = 3
        return lm

    def test_lm_overrides_acoustic_ranking(self):
        res = FakeGenResult(
            sequences_ids=[[10, 20], [30, 40], [50, 60]],
            scores=[-0.3, -0.5, -0.8],
        )
        opts = FakeOptions(length_penalty=1)
        tok = _make_tokenizer()
        lm = self._make_lm(preferred_idx=2)

        tokens, avg_lp = _select_best_hypothesis(res, tok, opts, lm_fusion=lm)
        assert tokens == [50, 60]
        lm.rescore.assert_called_once()

    def test_no_lm_picks_best_acoustic(self):
        res = FakeGenResult(
            sequences_ids=[[10, 20], [30, 40]],
            scores=[-0.3, -0.1],  # second is better
        )
        opts = FakeOptions(length_penalty=1)
        tok = _make_tokenizer()

        tokens, avg_lp = _select_best_hypothesis(res, tok, opts, lm_fusion=None)
        assert tokens == [30, 40]


# ---------------------------------------------------------------------------
# Integration: generate_segment_batched with LM fusion
# ---------------------------------------------------------------------------

class TestGenerateBatchedWithLM:
    def test_lm_fusion_passes_num_hypotheses(self):
        """When lm_fusion is set, num_hypotheses is forwarded to generate()."""
        model = MagicMock()
        model.max_length = 448
        model.get_prompt.return_value = [50258]
        model.encode.return_value = MagicMock()

        lm = MagicMock()
        lm.num_hypotheses = 3
        lm.rescore.return_value = 0
        model.lm_fusion = lm

        res = FakeGenResult(
            sequences_ids=[[10, 20], [30, 40], [50, 60]],
            scores=[-0.3, -0.5, -0.8],
            no_speech_prob=0.1,
        )
        model.model.generate.return_value = [res]

        features = np.zeros((1, 80, 3000))
        opts = FakeOptions(beam_size=5)
        tok = _make_tokenizer()

        WhisperModel.generate_segment_batched(model, features, tok, opts)

        # Check that num_hypotheses=3 was passed
        call_kwargs = model.model.generate.call_args
        assert call_kwargs.kwargs.get("num_hypotheses") == 3

    def test_no_lm_fusion_num_hypotheses_is_1(self):
        """Without lm_fusion, num_hypotheses defaults to 1."""
        model = MagicMock()
        model.max_length = 448
        model.lm_fusion = None
        model.get_prompt.return_value = [50258]
        model.encode.return_value = MagicMock()

        res = FakeGenResult(
            sequences_ids=[[10, 20]],
            scores=[-0.3],
            no_speech_prob=0.1,
        )
        model.model.generate.return_value = [res]

        features = np.zeros((1, 80, 3000))
        opts = FakeOptions()
        tok = _make_tokenizer()

        WhisperModel.generate_segment_batched(model, features, tok, opts)

        call_kwargs = model.model.generate.call_args
        assert call_kwargs.kwargs.get("num_hypotheses") == 1

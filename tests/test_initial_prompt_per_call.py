"""Tests for per-call initial_prompt override in transcribe() / transcribe_multi().

Ensures the contract we want from the engine:
- initial_prompt can be overridden per call without mutating self.options
- suppress_numerals doesn't mutate self.options during the call
- the HF Pipeline forward_params path carries a call-scoped options tuple
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from faster_whisper.transcribe import TranscriptionOptions
from whisperx.asr import FasterWhisperPipeline
from whisperx.vads.vad import Vad


def _default_options(**overrides):
    defaults = dict(
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[0.0],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=[-1],
        without_timestamps=True,
        max_initial_timestamp=1.0,
        word_timestamps=False,
        prepend_punctuations="\"'",
        append_punctuations="\"'.,",
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
    )
    defaults.update(overrides)
    return TranscriptionOptions(**defaults)


class _FakeVad(Vad):
    """Minimal Vad subclass returning a fixed list of segments."""

    def __init__(self, segments):
        self._segments = segments

    def __call__(self, inp):
        return MagicMock()

    @staticmethod
    def preprocess_audio(audio):
        if isinstance(audio, np.ndarray):
            return torch.from_numpy(audio).unsqueeze(0)
        return audio

    def merge_chunks(self, segments, chunk_size, onset, offset):  # noqa: ARG002
        return list(self._segments)


def _make_pipeline(vad_segments=None):
    pipeline = object.__new__(FasterWhisperPipeline)

    pipeline.model = MagicMock()
    pipeline.model.feat_kwargs = {"feature_size": 80}
    pipeline.model.lm_fusion = None
    pipeline.model.max_length = 448
    pipeline.model.hf_tokenizer = MagicMock()
    pipeline.model.model = MagicMock()
    pipeline.model.model.is_multilingual = False
    pipeline.model.get_prompt.return_value = [50258]
    pipeline.model.encode.return_value = MagicMock()
    pipeline.model.generate_segment_batched.return_value = {
        "text": ["bonjour"],
        "avg_logprob": [-0.3],
        "no_speech_prob": [0.05],
        "tokens": [[100, 200]],
        "compression_ratio": [1.5],
    }

    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.eot = 50257
    pipeline.tokenizer.language_code = "fr"
    pipeline.tokenizer.task = "transcribe"

    pipeline.options = _default_options()
    pipeline.preset_language = "fr"
    pipeline.suppress_numerals = False
    pipeline._batch_size = 1
    pipeline._num_workers = 0
    pipeline._vad_params = {"vad_onset": 0.5, "vad_offset": 0.363}
    pipeline._preprocess_params = {}
    pipeline._forward_params = {}
    pipeline._postprocess_params = {}
    pipeline.call_count = 0
    pipeline.framework = "pt"
    pipeline.device = torch.device("cpu")
    pipeline.binary_output = False

    segments = vad_segments or [{"start": 0.0, "end": 2.0, "segments": [(0.0, 2.0)]}]
    pipeline.vad_model = _FakeVad(segments)

    return pipeline


# ---------------------------------------------------------------------------
# _forward contract: accept a call-scoped `options` kwarg
# ---------------------------------------------------------------------------

class TestForwardOptionsOverride:
    def test_forward_uses_self_options_by_default(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt="default_prompt")

        pipeline._forward({"inputs": torch.zeros(1, 80, 3000)})

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used.initial_prompt == "default_prompt"

    def test_forward_uses_call_scoped_options_when_provided(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt="default")
        custom = _default_options(initial_prompt="custom_call_prompt")

        pipeline._forward({"inputs": torch.zeros(1, 80, 3000)}, options=custom)

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used.initial_prompt == "custom_call_prompt"

    def test_forward_does_not_mutate_self_options(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt=None)
        custom = _default_options(initial_prompt="override")

        pipeline._forward({"inputs": torch.zeros(1, 80, 3000)}, options=custom)

        assert pipeline.options.initial_prompt is None


# ---------------------------------------------------------------------------
# _sanitize_parameters routes `options` kwarg into forward_params
# ---------------------------------------------------------------------------

class TestSanitizeParametersRoutesOptions:
    def test_options_kwarg_goes_to_forward_params(self):
        pipeline = _make_pipeline()
        custom = _default_options(initial_prompt="x")
        _pre, fwd, _post = pipeline._sanitize_parameters(options=custom)
        assert fwd.get("options") is custom


# ---------------------------------------------------------------------------
# transcribe(): initial_prompt as a call-scoped parameter
# ---------------------------------------------------------------------------

class TestTranscribeInitialPrompt:
    def test_accepts_initial_prompt_kwarg(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000 * 3, dtype=np.float32)
        pipeline.transcribe(audio, initial_prompt="le patient dit son nom")

    def test_initial_prompt_propagated_to_generate(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000 * 3, dtype=np.float32)

        pipeline.transcribe(audio, initial_prompt="Nom de famille : DUPONT")

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used.initial_prompt == "Nom de famille : DUPONT"

    def test_initial_prompt_does_not_mutate_self_options(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt=None)
        audio = np.zeros(16000 * 3, dtype=np.float32)

        pipeline.transcribe(audio, initial_prompt="override")

        assert pipeline.options.initial_prompt is None

    def test_no_initial_prompt_uses_default(self):
        """Regression: without override, self.options is passed by identity.

        The ``is`` check locks in byte-equivalence with the pre-patch path:
        no unnecessary ``replace()`` allocations when the caller provides
        no override.
        """
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt="default_stays")
        audio = np.zeros(16000 * 3, dtype=np.float32)

        pipeline.transcribe(audio)

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used is pipeline.options

    def test_state_leak_between_two_calls(self):
        """Call A with prompt, call B without prompt → B must see the default, not A's prompt."""
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt=None)
        audio = np.zeros(16000 * 3, dtype=np.float32)

        pipeline.transcribe(audio, initial_prompt="Nom: DUPONT")
        pipeline.transcribe(audio)  # no override

        last_options = pipeline.model.generate_segment_batched.call_args[0][2]
        assert last_options.initial_prompt is None, (
            "State leak: the second call inherited the first call's initial_prompt. "
            "This is the exact Triton bug we're eliminating at the engine level."
        )


# ---------------------------------------------------------------------------
# suppress_numerals must not mutate self.options during the call
# ---------------------------------------------------------------------------

class TestTranscribeSuppressNumeralsNoMutation:
    def test_self_options_untouched_during_generate(self):
        from whisperx import asr as asr_mod

        pipeline = _make_pipeline()
        pipeline.suppress_numerals = True
        original_find = asr_mod.find_numeral_symbol_tokens
        asr_mod.find_numeral_symbol_tokens = lambda _t: [10, 11, 12]

        original_suppress = list(pipeline.options.suppress_tokens)
        captured = {}

        def capture(features, tokenizer, options, **_kw):
            captured["self_suppress_at_call"] = list(pipeline.options.suppress_tokens)
            captured["call_suppress"] = list(options.suppress_tokens)
            return {
                "text": ["x"],
                "avg_logprob": [-0.3],
                "no_speech_prob": [0.05],
                "tokens": [[100]],
                "compression_ratio": [1.5],
            }

        pipeline.model.generate_segment_batched.side_effect = capture
        audio = np.zeros(16000 * 3, dtype=np.float32)

        try:
            pipeline.transcribe(audio)
        finally:
            asr_mod.find_numeral_symbol_tokens = original_find

        assert captured["self_suppress_at_call"] == original_suppress, (
            "self.options.suppress_tokens was mutated DURING the call — "
            "the race-prone pattern we want to eliminate."
        )
        assert 10 in captured["call_suppress"]
        assert 11 in captured["call_suppress"]
        assert 12 in captured["call_suppress"]


# ---------------------------------------------------------------------------
# transcribe_multi(): same contract
# ---------------------------------------------------------------------------

class TestTranscribeMultiInitialPrompt:
    def test_accepts_initial_prompt_kwarg(self):
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]
        pipeline.transcribe_multi(audios, initial_prompt="x")

    def test_initial_prompt_propagated(self):
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(audios, initial_prompt="context")

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used.initial_prompt == "context"

    def test_initial_prompt_does_not_mutate_self_options(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt=None)
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(audios, initial_prompt="override")

        assert pipeline.options.initial_prompt is None

    def test_state_leak_between_two_calls(self):
        pipeline = _make_pipeline()
        pipeline.options = _default_options(initial_prompt=None)
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(audios, initial_prompt="LEAK")
        pipeline.transcribe_multi(audios)

        last_options = pipeline.model.generate_segment_batched.call_args[0][2]
        assert last_options.initial_prompt is None

    def test_initial_prompt_propagated_precompute_features(self):
        """precompute_features path calls generate_segment_batched directly,
        bypassing the HF Pipeline __call__ — the per-call options must reach
        that branch too.
        """
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(
            audios, initial_prompt="precomp_ctx", precompute_features=True,
        )

        options_used = pipeline.model.generate_segment_batched.call_args[0][2]
        assert options_used.initial_prompt == "precomp_ctx"


# ---------------------------------------------------------------------------
# n_best must also be call-scoped — no mutation of self._forward_params
# ---------------------------------------------------------------------------

class TestTranscribeMultiNBest:
    """n_best must be wired through transcribe_multi() like it is for transcribe()."""

    def _generate_with_hypotheses(self):
        """Mock of generate_segment_batched that emits hypotheses when n_best>1."""
        def generate(features, tokenizer, options, n_best=1, **_kw):
            out = {
                "text": ["top"],
                "avg_logprob": [-0.3],
                "no_speech_prob": [0.05],
                "tokens": [[100]],
                "compression_ratio": [1.5],
            }
            if n_best > 1:
                out["hypotheses"] = [[
                    {"text": "top", "avg_logprob": -0.3, "tokens": [100]},
                    {"text": "alt", "avg_logprob": -0.5, "tokens": [101]},
                ]]
            return out
        return generate

    def test_accepts_n_best_kwarg(self):
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]
        pipeline.transcribe_multi(audios, n_best=3)

    def test_n_best_propagated_to_generate(self):
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(audios, n_best=4)

        call_kwargs = pipeline.model.generate_segment_batched.call_args.kwargs
        assert call_kwargs.get("n_best") == 4

    def test_n_best_propagated_precompute_features(self):
        """precompute_features bypasses __call__ and hits generate_segment_batched
        directly — n_best must reach that branch too."""
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        pipeline.transcribe_multi(
            audios, n_best=2, precompute_features=True,
        )

        call_kwargs = pipeline.model.generate_segment_batched.call_args.kwargs
        assert call_kwargs.get("n_best") == 2

    def test_hypotheses_attached_when_n_best_gt_1(self):
        pipeline = _make_pipeline()
        pipeline.model.generate_segment_batched.side_effect = (
            self._generate_with_hypotheses()
        )
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        results = pipeline.transcribe_multi(audios, n_best=2)

        seg = results[0]["segments"][0]
        assert "hypotheses" in seg
        assert len(seg["hypotheses"]) == 2
        assert seg["hypotheses"][0]["text"] == "top"

    def test_hypotheses_attached_precompute_features(self):
        pipeline = _make_pipeline()
        pipeline.model.generate_segment_batched.side_effect = (
            self._generate_with_hypotheses()
        )
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        results = pipeline.transcribe_multi(
            audios, n_best=2, precompute_features=True,
        )

        seg = results[0]["segments"][0]
        assert "hypotheses" in seg
        assert len(seg["hypotheses"]) == 2

    def test_default_no_hypotheses_key(self):
        """Backward compat: n_best=1 (default) must not add a hypotheses key."""
        pipeline = _make_pipeline()
        audios = [np.zeros(16000 * 3, dtype=np.float32)]

        results = pipeline.transcribe_multi(audios)

        assert "hypotheses" not in results[0]["segments"][0]


class TestNBestCallScoped:
    def test_forward_params_not_mutated_during_call(self):
        """n_best must reach generate_segment_batched WITHOUT going through
        a transient self._forward_params mutation — otherwise a concurrent
        caller could observe the wrong n_best mid-flight.
        """
        pipeline = _make_pipeline()
        assert "n_best" not in pipeline._forward_params
        captured = {}

        def capture(features, tokenizer, options, n_best=1, **_kw):
            captured["forward_params_keys_at_call"] = set(pipeline._forward_params.keys())
            captured["n_best_seen"] = n_best
            return {
                "text": ["x"],
                "avg_logprob": [-0.3],
                "no_speech_prob": [0.05],
                "tokens": [[100]],
                "compression_ratio": [1.5],
            }

        pipeline.model.generate_segment_batched.side_effect = capture
        audio = np.zeros(16000 * 3, dtype=np.float32)

        pipeline.transcribe(audio, n_best=5)

        assert "n_best" not in captured["forward_params_keys_at_call"], (
            "self._forward_params was mutated with n_best during the call — "
            "race-prone pattern identical to the initial_prompt bug, just with "
            "a different field."
        )
        assert captured["n_best_seen"] == 5

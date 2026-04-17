"""Tests for transcribe_segment() — VAD-free transcription path."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from faster_whisper.transcribe import TranscriptionOptions
from whisperx.asr import FasterWhisperPipeline


def _default_options(**overrides):
    """Create a real TranscriptionOptions dataclass with sensible defaults."""
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
        prepend_punctuations="\"'\u2018\u00bf([{-",
        append_punctuations="\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]\u300b\u3001",
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
    )
    defaults.update(overrides)
    return TranscriptionOptions(**defaults)


def _make_pipeline():
    """Build a FasterWhisperPipeline with mocked internals."""
    pipeline = object.__new__(FasterWhisperPipeline)

    # Mock the WhisperModel
    pipeline.model = MagicMock()
    pipeline.model.feat_kwargs = {"feature_size": 80}
    pipeline.model.lm_fusion = None
    pipeline.model.max_length = 448
    pipeline.model.get_prompt.return_value = [50258, 50259, 50359]
    pipeline.model.encode.return_value = MagicMock()

    # Standard generate result
    pipeline.model.generate_segment_batched.return_value = {
        "text": ["bonjour madame"],
        "avg_logprob": [-0.3],
        "no_speech_prob": [0.05],
        "tokens": [[100, 200, 300]],
        "compression_ratio": [1.5],
    }

    # Mock tokenizer
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.eot = 50257
    pipeline.tokenizer.language_code = "fr"
    pipeline.tokenizer.task = "transcribe"

    # Pipeline state
    pipeline.options = _default_options()
    pipeline.preset_language = "fr"
    pipeline.suppress_numerals = False

    return pipeline


class TestTranscribeSegment:
    def test_returns_single_segment(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds

        result = pipeline.transcribe_segment(audio)

        assert result["language"] == "fr"
        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["text"] == "bonjour madame"
        assert seg["start"] == 0.0
        assert seg["end"] == 3.0

    def test_no_vad_called(self):
        pipeline = _make_pipeline()
        pipeline.vad_model = MagicMock()
        audio = np.zeros(16000, dtype=np.float32)

        pipeline.transcribe_segment(audio)

        # VAD model should never be called
        pipeline.vad_model.assert_not_called()

    def test_generate_called_with_features(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000 * 2, dtype=np.float32)

        pipeline.transcribe_segment(audio)

        pipeline.model.generate_segment_batched.assert_called_once()
        call_args = pipeline.model.generate_segment_batched.call_args
        features = call_args[0][0]
        # Should be (1, n_mels, time) — batch dim added
        assert len(features.shape) == 3
        assert features.shape[0] == 1

    def test_initial_prompt_override(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        pipeline.transcribe_segment(
            audio,
            initial_prompt="Le patient épelle son nom :",
        )

        call_args = pipeline.model.generate_segment_batched.call_args
        options_used = call_args[0][2]
        assert options_used.initial_prompt == "Le patient épelle son nom :"

    def test_initial_prompt_does_not_mutate_model_options(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000, dtype=np.float32)

        pipeline.transcribe_segment(
            audio, initial_prompt="override"
        )

        # Original options should be unchanged
        assert pipeline.options.initial_prompt is None

    def test_quality_metrics_present(self):
        pipeline = _make_pipeline()
        audio = np.zeros(16000 * 5, dtype=np.float32)

        result = pipeline.transcribe_segment(audio)
        seg = result["segments"][0]

        assert "avg_logprob" in seg
        assert "no_speech_prob" in seg
        assert "compression_ratio" in seg
        assert "tokens_per_sec" in seg
        assert "repeat_adjacent_frac" in seg

    def test_short_audio_padded(self):
        """Audio shorter than 30s should still work (padded)."""
        pipeline = _make_pipeline()
        audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds

        result = pipeline.transcribe_segment(audio)

        assert len(result["segments"]) == 1
        assert result["segments"][0]["end"] == 0.5

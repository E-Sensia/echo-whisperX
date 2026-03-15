"""Triton Python backend for WhisperX transcription.

Replica of the production model.py from inference-server, adapted to use the
local echo-whisperX fork. Supports both single-request and cross-call batched
inference via transcribe_multi().
"""

import json
import traceback
from dataclasses import replace

import numpy as np
import triton_python_backend_utils as pb_utils
import whisperx
from faster_whisper.tokenizer import Tokenizer

PUNCTS = [",", ".", "!", "?", ":", ";", "\u2026", "..."]


def rms_normalize(
    waveform: np.ndarray,
    target_dbfs: float = -20.0,
    eps: float = 1e-8,
) -> np.ndarray:
    rms = np.sqrt(np.mean(waveform**2)).clip(min=eps)
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    return np.clip(waveform * gain, -1.0, 1.0)


def log(*msgs):
    print(f"[WHISPERX] {' '.join(str(m) for m in msgs)}", flush=True)


def get_token_ids_for_strings(tokenizer: Tokenizer, strings):
    ids = set()
    for s in strings:
        for v in (s, " " + s):
            enc = tokenizer.encode(v)
            if len(enc) == 1 and enc[0] < tokenizer.eot:
                ids.add(enc[0])
    return sorted(ids)


def add_punctuation_suppression_to_pipeline(tokenizer, pipeline, puncts=PUNCTS):
    punct_ids = get_token_ids_for_strings(tokenizer, puncts)
    current = pipeline.options.suppress_tokens
    merged = sorted(set((current if isinstance(current, list) else [-1]) + punct_ids))
    pipeline.options = replace(pipeline.options, suppress_tokens=merged)


class TritonPythonModel:
    def initialize(self, _):
        self.device = "cuda"
        self.compute_type = "float16"

        log("Loading large-v3 model...")
        self.model = whisperx.load_model(
            "large-v3",
            self.device,
            compute_type=self.compute_type,
            language=None,
        )
        self.default_suppress_tokens = self.model.options.suppress_tokens

        self.alignment_models = {
            "fr": self._load_alignment("fr"),
            "en": self._load_alignment("en"),
        }
        self.batch_size = 16
        log("Initialization complete.")

    def _load_alignment(self, lang):
        log(f"Loading alignment for {lang}...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=lang, device=self.device
            )
            log(f"Alignment loaded for {lang}.")
            return (model_a, metadata)
        except Exception as e:
            log(f"Warning: alignment {lang} failed: {e}")
            return (None, None)

    def _parse_params(self, request):
        """Extract optional parameters from request."""
        params = {
            "initial_prompt": None,
            "enable_alignment": True,
            "delete_punctuation": False,
            "language": None,
            "use_multi": False,
        }
        if len(request.inputs()) > 1:
            try:
                raw = request.inputs()[1].as_numpy().flatten()
                if raw.size > 0:
                    parsed = json.loads(raw[0].decode("utf-8"))
                    params.update({k: parsed[k] for k in params if k in parsed})
            except (IndexError, json.JSONDecodeError, UnicodeDecodeError) as e:
                log(f"Warning: Could not parse parameters: {e}")
        return params

    def _transcribe_single(self, audio, params):
        """Transcribe a single audio array (production path)."""
        model = self.model
        forced_language = params["language"]

        if params["initial_prompt"]:
            model.options = replace(model.options, initial_prompt=params["initial_prompt"])

        if params["delete_punctuation"] and forced_language:
            tokenizer = Tokenizer(
                model.model.hf_tokenizer,
                model.model.model.is_multilingual,
                task="transcribe",
                language=forced_language,
            )
            add_punctuation_suppression_to_pipeline(tokenizer, model, PUNCTS)
        else:
            model.options = replace(
                model.options, suppress_tokens=self.default_suppress_tokens
            )

        result = model.transcribe(
            audio, language=forced_language, batch_size=self.batch_size
        )
        detected_language = result.get("language", "fr")
        final_segments = result["segments"]

        if params["enable_alignment"]:
            final_segments = self._align(final_segments, audio, detected_language)

        return {"segments": final_segments, "language": detected_language}

    def _align(self, segments, audio, language):
        """Apply word-level alignment if model available."""
        if language not in self.alignment_models:
            log(f"Loading alignment for {language}...")
            self.alignment_models[language] = self._load_alignment(language)

        model_a, metadata = self.alignment_models.get(language, (None, None))
        if model_a and metadata:
            try:
                aligned = whisperx.align(
                    segments, model_a, metadata, audio, self.device
                )
                return aligned["segments"]
            except Exception as e:
                log(f"Alignment failed: {e}")
        return segments

    def execute(self, requests):
        """Handle batch of Triton requests.

        If multiple requests arrive in the same batch (via dynamic_batching),
        use transcribe_multi() to process all audio arrays in one GPU pass.
        """
        try:
            # Parse all requests
            audios = []
            params_list = []
            for request in requests:
                audio = request.inputs()[0].as_numpy()
                # Flatten batch dim if present (max_batch_size > 0 adds it)
                audio = audio.flatten()
                audio = rms_normalize(audio)
                audios.append(audio)
                params_list.append(self._parse_params(request))

            # Check if we can use cross-call batching
            # Requirements: all same language, no punctuation suppression,
            # and more than 1 request in the batch
            can_multi = (
                len(requests) > 1
                and all(not p["delete_punctuation"] for p in params_list)
                and all(not p["initial_prompt"] for p in params_list)
                and len(set(p["language"] for p in params_list)) <= 1
            )

            if can_multi and hasattr(self.model, "transcribe_multi"):
                log(f"Cross-call batching {len(requests)} requests")
                forced_language = params_list[0]["language"]
                multi_results = self.model.transcribe_multi(
                    audios,
                    language=forced_language,
                    batch_size=self.batch_size,
                )

                responses = []
                for i, result in enumerate(multi_results):
                    detected_language = result.get("language", "fr")
                    final_segments = result["segments"]

                    if params_list[i]["enable_alignment"]:
                        final_segments = self._align(
                            final_segments, audios[i], detected_language
                        )

                    response_data = {
                        "segments": final_segments,
                        "language": detected_language,
                    }
                    segments_np = np.array(
                        [json.dumps(response_data, ensure_ascii=False)],
                        dtype=np.object_,
                    )
                    responses.append(
                        pb_utils.InferenceResponse(
                            [pb_utils.Tensor("segments", segments_np)]
                        )
                    )
                return responses

            # Fallback: process each request independently
            responses = []
            for audio, params in zip(audios, params_list):
                response_data = self._transcribe_single(audio, params)
                segments_np = np.array(
                    [json.dumps(response_data, ensure_ascii=False)],
                    dtype=np.object_,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        [pb_utils.Tensor("segments", segments_np)]
                    )
                )
            return responses

        except Exception as e:
            log("Error during inference:", str(e))
            log(traceback.format_exc())
            raise e

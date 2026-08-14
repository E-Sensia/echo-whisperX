import os
from typing import List, Optional, Union
from dataclasses import replace

import zlib
import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from whisperx.schema import SingleSegment, TranscriptionResult, ProgressCallback
from whisperx.vads import Vad, Silero, Pyannote
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


def _compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def _token_repetition_metrics(text_tokens: List[int], duration: float) -> tuple[float, float, float]:
    tokens_per_sec = len(text_tokens) / max(1e-6, duration)
    adjacent_repeats = sum(
        1 for i in range(1, len(text_tokens)) if text_tokens[i] == text_tokens[i - 1]
    )
    repeat_adjacent_frac = adjacent_repeats / max(1, len(text_tokens) - 1)
    repeat_unique_frac = 1.0 - (len(set(text_tokens)) / max(1, len(text_tokens)))
    return tokens_per_sec, repeat_adjacent_frac, repeat_unique_frac


def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


def _needs_fallback(compression_ratio, avg_logprob, no_speech_prob, options):
    """Check whether a decoded sample should be re-decoded at a higher temperature."""
    needs = False

    if (options.compression_ratio_threshold is not None
            and compression_ratio > options.compression_ratio_threshold):
        needs = True

    if (options.log_prob_threshold is not None
            and avg_logprob < options.log_prob_threshold):
        needs = True

    # Silence: high no-speech probability AND low avg logprob → not an error.
    if (options.no_speech_threshold is not None
            and no_speech_prob > options.no_speech_threshold
            and options.log_prob_threshold is not None
            and avg_logprob < options.log_prob_threshold):
        needs = False

    return needs


def _select_best_hypothesis(gen_result, tokenizer, options, lm_fusion=None):
    """Pick the best hypothesis from a CTranslate2 generation result.

    Returns ``(tokens, avg_logprob)`` for the winning hypothesis.
    """
    n_hyp = len(gen_result.sequences_ids)

    # Fast path: single hypothesis, no LM rescoring
    if n_hyp == 1 and lm_fusion is None:
        tokens = gen_result.sequences_ids[0]
        seq_len = len(tokens)
        cum_logprob = gen_result.scores[0] * (seq_len ** options.length_penalty)
        return tokens, cum_logprob / (seq_len + 1)

    # Compute acoustic score for every hypothesis
    candidates_tokens = []
    candidates_avg_lp = []
    for h in range(n_hyp):
        tokens = gen_result.sequences_ids[h]
        seq_len = len(tokens)
        cum_logprob = gen_result.scores[h] * (seq_len ** options.length_penalty)
        candidates_tokens.append(tokens)
        candidates_avg_lp.append(cum_logprob / (seq_len + 1))

    if lm_fusion is not None:
        # Decode texts for LM scoring
        filtered = [[t for t in tk if t < tokenizer.eot] for tk in candidates_tokens]
        texts = tokenizer.tokenizer.decode_batch(filtered)
        best_idx = lm_fusion.rescore(texts, candidates_avg_lp)
    else:
        # Multiple hypotheses from sampling → pick highest avg logprob
        best_idx = max(range(n_hyp), key=lambda j: candidates_avg_lp[j])

    return candidates_tokens[best_idx], candidates_avg_lp[best_idx]


class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output=None,
        n_best: int = 1,
    ):
        """Transcribe a batch of pre-computed mel features.

        Parameters
        ----------
        n_best : int, default 1
            When > 1, the method returns the top-``n_best`` beam-search
            hypotheses for each sample under the ``"hypotheses"`` key of the
            output dict. Each entry is a list of ``{"text": str,
            "avg_logprob": float}`` sorted by descending acoustic score. The
            ``beam_size`` is automatically bumped to ``max(beam_size, n_best)``
            so CTranslate2 has enough beams.

            **When ``n_best > 1``, the multi-temperature fallback is
            deliberately SKIPPED**. Callers asking for N-best have their own
            reranker downstream — mixing T>0 sampling hypotheses into the
            N-best output would break rerankers that assume comparable T=0
            beam scores. Setting ``n_best=1`` (default) preserves the exact
            prior behaviour (fallback runs, no ``"hypotheses"`` key).
        """
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
            hotwords=options.hotwords
        )

        encoder_output = self.encode(features)

        lm_fusion = getattr(self, 'lm_fusion', None)
        num_hypotheses = (
            min(lm_fusion.num_hypotheses, options.beam_size)
            if lm_fusion is not None else 1
        )
        # If caller asked for N-best, make sure we actually generate that many.
        if n_best > 1:
            num_hypotheses = max(num_hypotheses, n_best)
        effective_beam_size = max(options.beam_size, num_hypotheses)

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                return_scores=True,
                return_no_speech_prob=True,
                beam_size=effective_beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                repetition_penalty=options.repetition_penalty,
                num_hypotheses=num_hypotheses,
            )

        # Snapshot the full N-best NOW, before the multi-temperature fallback
        # below can swap out the "winning" hypothesis for a sampling-based one.
        # Downstream rerankers need the initial beam-search output.
        hypotheses_batch = None
        if n_best > 1:
            hypotheses_batch = []
            for res in result:
                per_sample = []
                for seq_ids, cum_score in zip(
                    res.sequences_ids[:n_best], res.scores[:n_best]
                ):
                    seq_len = len(seq_ids)
                    cum_lp = cum_score * (seq_len ** options.length_penalty)
                    avg_lp = cum_lp / (seq_len + 1)
                    filtered = [t for t in seq_ids if t < tokenizer.eot]
                    txt = tokenizer.tokenizer.decode(filtered).strip()
                    per_sample.append({
                        "text": txt,
                        "avg_logprob": float(avg_lp),
                    })
                hypotheses_batch.append(per_sample)

        # Select best hypothesis per sample (with optional LM rescoring)
        tokens_batch = []
        avg_logprobs = []
        for res in result:
            best_tokens, best_avg_lp = _select_best_hypothesis(
                res, tokenizer, options, lm_fusion
            )
            tokens_batch.append(best_tokens)
            avg_logprobs.append(best_avg_lp)

        def decode_batch(tokens: List[List[int]]) -> List[str]:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)
        no_speech_probs = [r.no_speech_prob for r in result]
        compression_ratios = [_compression_ratio(t) for t in text]

        # --- Multi-temperature fallback for failed samples ---
        # Skipped when the caller requested N-best: callers asking for
        # multiple hypotheses have their own reranking downstream, and mixing
        # T>0 sampling hypotheses into the N-best output would break the
        # rerankers that assume comparable T=0 beam scores (e.g. consensus-
        # weighted majority). The fallback targets a different failure mode
        # (hallucinations on a default top-1) that N-best rerankers handle
        # through other signals.
        has_fallback_temps = len(options.temperatures) > 1
        has_thresholds = (
            options.compression_ratio_threshold is not None
            or options.log_prob_threshold is not None
        )
        skip_fallback = n_best > 1

        if has_fallback_temps and has_thresholds and not skip_fallback:
            for i in range(batch_size):
                if not _needs_fallback(
                    compression_ratios[i], avg_logprobs[i],
                    no_speech_probs[i], options,
                ):
                    continue

                logger.debug(
                    "Sample %d/%d: fallback triggered "
                    "(cr=%.2f, avg_lp=%.3f, nsp=%.3f)",
                    i + 1, batch_size,
                    compression_ratios[i], avg_logprobs[i], no_speech_probs[i],
                )

                single_encoder = self.encode(features[i:i+1])

                # Track all attempts (including the initial beam-search one)
                initial_attempt = (
                    text[i], avg_logprobs[i], tokens_batch[i],
                    compression_ratios[i],
                )
                all_attempts = [initial_attempt]
                below_cr_attempts = []
                if (options.compression_ratio_threshold is None
                        or compression_ratios[i]
                        <= options.compression_ratio_threshold):
                    below_cr_attempts.append(initial_attempt)

                settled = False
                for temperature in options.temperatures[1:]:
                    fb_result = self.model.generate(
                        single_encoder,
                        [prompt],
                        return_scores=True,
                        return_no_speech_prob=True,
                        beam_size=1,
                        num_hypotheses=options.best_of,
                        sampling_topk=0,
                        sampling_temperature=temperature,
                        length_penalty=options.length_penalty,
                        max_length=self.max_length,
                        suppress_blank=options.suppress_blank,
                        suppress_tokens=options.suppress_tokens,
                        no_repeat_ngram_size=options.no_repeat_ngram_size,
                        repetition_penalty=options.repetition_penalty,
                    )[0]

                    fb_tokens, fb_avg_lp = _select_best_hypothesis(
                        fb_result, tokenizer, options, lm_fusion,
                    )
                    fb_text = decode_batch([fb_tokens])[0]
                    fb_cr = _compression_ratio(fb_text)

                    attempt = (fb_text, fb_avg_lp, fb_tokens, fb_cr)
                    all_attempts.append(attempt)
                    if (options.compression_ratio_threshold is None
                            or fb_cr <= options.compression_ratio_threshold):
                        below_cr_attempts.append(attempt)

                    if not _needs_fallback(
                        fb_cr, fb_avg_lp,
                        fb_result.no_speech_prob, options,
                    ):
                        text[i] = fb_text
                        avg_logprobs[i] = fb_avg_lp
                        tokens_batch[i] = fb_tokens
                        compression_ratios[i] = fb_cr
                        settled = True
                        logger.debug(
                            "Sample %d: settled at temperature %.1f",
                            i + 1, temperature,
                        )
                        break

                if not settled:
                    # All temperatures exhausted — pick the best result
                    pool = below_cr_attempts or all_attempts
                    best = max(pool, key=lambda x: x[1])
                    text[i] = best[0]
                    avg_logprobs[i] = best[1]
                    tokens_batch[i] = best[2]
                    compression_ratios[i] = best[3]
                    logger.debug(
                        "Sample %d: all temperatures exhausted, "
                        "best avg_logprob=%.3f",
                        i + 1, best[1],
                    )

        output = {
            "text": text,
            "avg_logprob": avg_logprobs,
            "no_speech_prob": no_speech_probs,
            "tokens": tokens_batch,
            "compression_ratio": compression_ratios,
        }
        if hypotheses_batch is not None:
            output["hypotheses"] = hypotheses_batch
        return output

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        if "options" in kwargs:
            forward_kwargs["options"] = kwargs["options"]
        if "n_best" in kwargs:
            forward_kwargs["n_best"] = kwargs["n_best"]
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        padding = max(0, N_SAMPLES - audio.shape[0])
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=padding,
        )
        return {'inputs': features}

    def _forward(self, model_inputs, **forward_params):
        # HuggingFace Pipeline unpacks ``self._forward_params`` into **kwargs
        # when invoking _forward, so we read n_best from either the kwargs
        # or fall back to the stored dict (both paths must work).
        n_best = forward_params.get(
            "n_best", self._forward_params.get("n_best", 1)
        )
        # Per-call options override (e.g. caller-scoped initial_prompt) — falls
        # back to self.options when no override is supplied. This is the
        # mechanism that keeps self.options immutable across concurrent calls.
        options = forward_params.get("options", self.options)
        outputs = self.model.generate_segment_batched(
            model_inputs['inputs'],
            self.tokenizer,
            options,
            n_best=n_best,
        )
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
        initial_prompt: Optional[str] = None,
        n_best: int = 1,
    ) -> TranscriptionResult:
        """Transcribe with VAD segmentation.

        ``initial_prompt`` (default None): per-call override of the decoder
        prompt.  When provided, it does **not** mutate ``self.options`` —
        the override lives only for this call, so concurrent callers never
        leak prompts into each other.

        ``n_best`` (default 1, backward compatible): when > 1, each returned
        segment has a ``"hypotheses"`` key with the top-``n_best`` beam-search
        candidates captured before any multi-temperature fallback.  Like
        ``initial_prompt``, ``n_best`` is call-scoped — it never touches
        ``self._forward_params`` and is safe under concurrent callers.
        """
        return self._do_transcribe(
            audio, batch_size, num_workers, language, task, chunk_size,
            print_progress, combined_progress, verbose, progress_callback,
            initial_prompt=initial_prompt,
            n_best=n_best,
        )

    def _do_transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
        initial_prompt: Optional[str] = None,
        n_best: int = 1,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        # Pre-process audio and merge chunks as defined by the respective VAD child class 
        # In case vad_model is manually assigned (see 'load_model') follow the functionality of pyannote toolkit
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks =  self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        # Build a call-scoped options instance — never mutate self.options so
        # that concurrent callers cannot leak prompts/suppress_tokens into
        # each other.
        call_options = self.options
        if initial_prompt is not None:
            call_options = replace(call_options, initial_prompt=initial_prompt)
        if self.suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            logger.info("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = list(set(
                numeral_symbol_tokens + call_options.suppress_tokens
            ))
            call_options = replace(call_options, suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(
            data(audio, vad_segments),
            batch_size=batch_size,
            num_workers=num_workers,
            options=call_options,
            n_best=n_best,
        )):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)
            text = out['text']
            avg_logprob = out['avg_logprob']
            tokens = out['tokens']
            no_speech_prob = out['no_speech_prob']
            compression_ratio = out['compression_ratio']
            hypotheses = out.get('hypotheses')  # present only when n_best > 1

            if batch_size in [0, 1, None]:
                text = text[0]
                avg_logprob = avg_logprob[0]
                tokens = tokens[0]
                no_speech_prob = no_speech_prob[0]
                compression_ratio = compression_ratio[0]
                if hypotheses is not None:
                    hypotheses = hypotheses[0]
            if verbose:
                print(f"Transcript: [{round(vad_segments[idx]['start'], 3)} --> {round(vad_segments[idx]['end'], 3)}] {text}")

            start_t = round(vad_segments[idx]['start'], 3)
            end_t = round(vad_segments[idx]['end'], 3)
            duration = end_t - start_t

            text_tokens = [t for t in tokens if t < self.tokenizer.eot]
            tokens_per_sec, repeat_adjacent_frac, repeat_unique_frac = _token_repetition_metrics(
                text_tokens, duration
            )

            seg_dict = {
                "text": text,
                "start": start_t,
                "end": end_t,
                "avg_logprob": float(avg_logprob),
                "no_speech_prob": float(no_speech_prob),
                "compression_ratio": float(compression_ratio),
                "tokens_len": int(len(text_tokens)),
                "tokens_per_sec": float(tokens_per_sec),
                "repeat_adjacent_frac": float(repeat_adjacent_frac),
                "repeat_unique_frac": float(repeat_unique_frac),
            }
            if hypotheses is not None:
                seg_dict["hypotheses"] = hypotheses
            segments.append(seg_dict)

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        return {"segments": segments, "language": language}

    def transcribe_segment(
        self,
        audio: Union[str, np.ndarray],
        initial_prompt: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        n_best: int = 1,
    ) -> TranscriptionResult:
        """Transcribe a single pre-segmented audio chunk without VAD.

        Designed for cases where speech boundaries are already known
        (e.g. callbot with external end-of-speech detection).  Skips
        VAD and alignment — just mel-spectrogram → encoder → decoder.

        Args:
            audio: Audio array (16 kHz float32) or path. Already
                segmented by the caller; no VAD will be applied.
            initial_prompt: Optional prompt to condition the decoder
                (overrides the model-level initial_prompt for this call).
            language: Language code (default: model language).
            task: "transcribe" or "translate".
            n_best: When > 1, attach the top-``n_best`` beam-search hypotheses
                to the returned segment under the ``"hypotheses"`` key. Useful
                for downstream rerankers that need the full N-best. Default 1
                preserves the prior behaviour (no hypotheses key).
        """
        if isinstance(audio, str):
            audio = load_audio(audio)

        # Ensure tokenizer
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        # Build options — override initial_prompt if provided
        options = self.options
        if initial_prompt is not None:
            options = replace(options, initial_prompt=initial_prompt)

        # Mel spectrogram (same as preprocess())
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        padding = max(0, N_SAMPLES - audio.shape[0])
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=padding,
        )
        features = features.unsqueeze(0)  # batch dim

        # Direct decode — no VAD
        out = self.model.generate_segment_batched(
            features, self.tokenizer, options, n_best=n_best,
        )

        text = out["text"][0]
        tokens = out["tokens"][0]
        avg_logprob = out["avg_logprob"][0]
        no_speech_prob = out["no_speech_prob"][0]
        compression_ratio = out["compression_ratio"][0]

        duration = len(audio) / SAMPLE_RATE
        text_tokens = [t for t in tokens if t < self.tokenizer.eot]
        tokens_per_sec, repeat_adjacent_frac, repeat_unique_frac = (
            _token_repetition_metrics(text_tokens, duration)
        )

        segment: SingleSegment = {
            "text": text,
            "start": 0.0,
            "end": round(duration, 3),
            "avg_logprob": float(avg_logprob),
            "no_speech_prob": float(no_speech_prob),
            "compression_ratio": float(compression_ratio),
            "tokens_len": int(len(text_tokens)),
            "tokens_per_sec": float(tokens_per_sec),
            "repeat_adjacent_frac": float(repeat_adjacent_frac),
            "repeat_unique_frac": float(repeat_unique_frac),
        }
        if "hypotheses" in out:
            segment["hypotheses"] = out["hypotheses"][0]

        # Revert tokenizer if multilingual
        if self.preset_language is None:
            self.tokenizer = None

        return {"segments": [segment], "language": language}

    def transcribe_multi(
        self,
        audios: List[np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        precompute_features: bool = False,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        initial_prompt: Optional[str] = None,
        n_best: int = 1,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio arrays in a single batched pass.

        Runs VAD on each audio, pools all VAD segments together, processes them
        in combined batches through the encoder/decoder, then demultiplexes
        results back to per-audio outputs.  This is significantly more efficient
        than calling transcribe() N times when processing many short audio
        chunks concurrently.

        ``initial_prompt`` is applied per call without mutating ``self.options``
        so concurrent callers cannot leak prompts into each other.

        ``n_best`` (default 1, backward compatible): when > 1, each returned
        segment has a ``"hypotheses"`` key with the top-``n_best`` beam-search
        candidates captured before any multi-temperature fallback.  Like
        ``initial_prompt``, ``n_best`` is call-scoped — it never touches
        ``self._forward_params`` and is safe under concurrent callers.
        """
        if issubclass(type(self.vad_model), Vad):
            preprocess_fn = self.vad_model.preprocess_audio
            merge_fn = self.vad_model.merge_chunks
        else:
            preprocess_fn = Pyannote.preprocess_audio
            merge_fn = Pyannote.merge_chunks

        # 1. Run VAD on each audio, collect segments with source mapping
        all_vad_segments = []  # (audio_idx, vad_segment, audio_ref)
        per_audio_vad = []     # per-audio list of vad_segments

        for i, audio in enumerate(audios):
            waveform = preprocess_fn(audio)
            vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
            vad_segments = merge_fn(
                vad_segments,
                chunk_size,
                onset=self._vad_params["vad_onset"],
                offset=self._vad_params["vad_offset"],
            )
            per_audio_vad.append(vad_segments)
            for seg in vad_segments:
                all_vad_segments.append((i, seg, audio))

        if not all_vad_segments:
            return [{"segments": [], "language": language or "fr"} for _ in audios]

        # 2. Ensure tokenizer is set
        if self.tokenizer is None:
            language = language or self.detect_language(audios[0])
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        # Call-scoped options instance — never mutate self.options.
        call_options = self.options
        if initial_prompt is not None:
            call_options = replace(call_options, initial_prompt=initial_prompt)
        if self.suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            new_suppressed_tokens = list(set(
                numeral_symbol_tokens + call_options.suppress_tokens
            ))
            call_options = replace(call_options, suppress_tokens=new_suppressed_tokens)

        # 3. Combined data generator across all audios
        batch_size = batch_size or self._batch_size

        if precompute_features:
            # Pre-compute mel spectrograms on CPU threads, then batch to GPU
            from concurrent.futures import ThreadPoolExecutor

            model_n_mels = self.model.feat_kwargs.get("feature_size")
            n_mels = model_n_mels if model_n_mels is not None else 80

            def _compute_mel(item):
                audio_idx, seg, audio = item
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                chunk = audio[f1:f2]
                padding = max(0, N_SAMPLES - chunk.shape[0])
                features = log_mel_spectrogram(
                    chunk,
                    n_mels=n_mels,
                    padding=padding,
                )
                return features

            n_threads = max(2, min(num_workers or 4, 8))
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                precomputed = list(pool.map(_compute_mel, all_vad_segments))

            # Batch features and call generate_segment_batched directly
            raw_outputs = []
            for i in range(0, len(precomputed), batch_size):
                batch_features = torch.stack(precomputed[i:i + batch_size])
                batch_out = self.model.generate_segment_batched(
                    batch_features, self.tokenizer, call_options,
                    n_best=n_best,
                )
                # Unbatch: convert dict-of-lists to list-of-dicts
                batch_hypotheses = batch_out.get("hypotheses")
                n = batch_features.shape[0]
                for j in range(n):
                    item = {
                        "text": batch_out["text"][j],
                        "avg_logprob": batch_out["avg_logprob"][j],
                        "no_speech_prob": batch_out["no_speech_prob"][j],
                        "tokens": batch_out["tokens"][j],
                        "compression_ratio": batch_out["compression_ratio"][j],
                    }
                    if batch_hypotheses is not None:
                        item["hypotheses"] = batch_hypotheses[j]
                    raw_outputs.append(item)
        else:
            def data_multi():
                for audio_idx, seg, audio in all_vad_segments:
                    f1 = int(seg['start'] * SAMPLE_RATE)
                    f2 = int(seg['end'] * SAMPLE_RATE)
                    yield {'inputs': audio[f1:f2]}

            # 4. Process all segments in combined batches
            raw_outputs = []
            for out in self.__call__(
                data_multi(),
                batch_size=batch_size,
                num_workers=num_workers,
                options=call_options,
                n_best=n_best,
            ):
                raw_outputs.append(out)

        # 5. Demux results back to per-audio
        results: List[TranscriptionResult] = [{"segments": [], "language": language} for _ in audios]

        for seg_idx, out in enumerate(raw_outputs):
            audio_idx, vad_seg, _ = all_vad_segments[seg_idx]

            text = out['text']
            avg_logprob = out['avg_logprob']
            tokens = out['tokens']
            no_speech_prob = out['no_speech_prob']
            compression_ratio = out['compression_ratio']
            hypotheses = out.get('hypotheses')  # present only when n_best > 1

            # Pipeline iterator path wraps single items in lists when batch_size <= 1
            if not precompute_features and batch_size in [0, 1, None]:
                text = text[0]
                avg_logprob = avg_logprob[0]
                tokens = tokens[0]
                no_speech_prob = no_speech_prob[0]
                compression_ratio = compression_ratio[0]
                if hypotheses is not None:
                    hypotheses = hypotheses[0]

            start_t = round(vad_seg['start'], 3)
            end_t = round(vad_seg['end'], 3)
            duration = end_t - start_t

            text_tokens = [t for t in tokens if t < self.tokenizer.eot]
            tokens_per_sec, repeat_adjacent_frac, repeat_unique_frac = _token_repetition_metrics(
                text_tokens, duration
            )

            seg_dict = {
                "text": text,
                "start": start_t,
                "end": end_t,
                "avg_logprob": float(avg_logprob),
                "no_speech_prob": float(no_speech_prob),
                "compression_ratio": float(compression_ratio),
                "tokens_len": int(len(text_tokens)),
                "tokens_per_sec": float(tokens_per_sec),
                "repeat_adjacent_frac": float(repeat_adjacent_frac),
                "repeat_unique_frac": float(repeat_unique_frac),
            }
            if hypotheses is not None:
                seg_dict["hypotheses"] = hypotheses
            results[audio_idx]["segments"].append(seg_dict)

        # Revert state
        if self.preset_language is None:
            self.tokenizer = None

        return results

    def detect_language(self, audio: np.ndarray) -> str:
        if audio.shape[0] < N_SAMPLES:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        logger.info(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio")
        return language


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad]= None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    use_auth_token: Optional[Union[str, bool]] = None,
    lm_model_path: Optional[str] = None,
    lm_weight: float = 0.1,
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
            Use "default" to automatically select based on device (float16 for GPU, float32 for CPU).
        vad_model - The vad model to manually assign.
        vad_method - The vad method to use. vad_model has a higher priority if it is not None.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
        logger.info(f"Compute type not specified, defaulting to {compute_type} for device {device}")

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads,
                         use_auth_token=use_auth_token)

    if lm_model_path is not None:
        from whisperx.lm_fusion import KenLMFusion
        model.lm_fusion = KenLMFusion(
            lm_model_path, lm_weight=lm_weight,
        )
    else:
        model.lm_fusion = None

    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        logger.info("No language specified, language will be detected for each audio file (increases inference time)")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    # Note: manually assigned vad_model has higher priority than vad_method!
    if vad_model is not None:
        print("Use manually assigned vad_model. vad_method is ignored.")
        vad_model = vad_model
    else:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            if device == 'cuda':
                device_vad = f'cuda:{device_index}'
            else:
                device_vad = device
            vad_model = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )

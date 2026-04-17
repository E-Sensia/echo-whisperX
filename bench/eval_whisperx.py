"""Evaluate WhisperX pipeline on hand-transcribed echo-asr test set.

Runs the full WhisperX pipeline (VAD + ASR + optional alignment) on the
echo-asr benchmark segments and computes WER against human references.

Usage:
    python bench/eval_whisperx.py
    python bench/eval_whisperx.py --whisper_arch large-v3-turbo --compute_type float16
    python bench/eval_whisperx.py --label "my_optimization"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import List

import jiwer
import numpy as np

ECHO_ASR_ROOT = Path("/home/thibault/Documents/projets/echo-asr")
DEFAULT_MANIFEST = ECHO_ASR_ROOT / "data" / "test_manifest.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate WhisperX on echo-asr test set")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--whisper_arch", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--language", type=str, default="fr")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--vad_method", type=str, default="pyannote")
    parser.add_argument("--skip_alignment", action="store_true")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions JSONL for later comparison")
    parser.add_argument("--no_fallback", action="store_true",
                        help="Disable multi-temperature fallback (for A/B comparison)")
    parser.add_argument("--lm_model_path", type=str, default=None,
                        help="Path to KenLM .arpa/.bin for shallow fusion")
    parser.add_argument("--lm_weight", type=float, default=0.1)
    args = parser.parse_args()

    import whisperx

    entries = load_manifest(args.manifest)
    print(f"Loaded {len(entries)} test segments")

    # Group entries by audio file for efficient loading
    by_file: dict[str, list[dict]] = {}
    for e in entries:
        by_file.setdefault(e["audio_filepath"], []).append(e)

    print(f"  {len(by_file)} unique audio files")
    print(f"\nLoading model: {args.whisper_arch} ({args.compute_type}, {args.vad_method})...")

    asr_options = {"beam_size": args.beam_size}
    if args.no_fallback:
        asr_options["temperatures"] = [0.0]
        asr_options["compression_ratio_threshold"] = None
        asr_options["log_prob_threshold"] = None

    pipeline = whisperx.load_model(
        whisper_arch=args.whisper_arch,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        language=args.language,
        vad_method=args.vad_method,
        asr_options=asr_options,
        lm_model_path=args.lm_model_path,
        lm_weight=args.lm_weight,
    )

    align_model = None
    align_metadata = None
    device_str = f"{args.device}:{args.device_index}" if args.device == "cuda" else args.device
    if not args.skip_alignment:
        align_model, align_metadata = whisperx.load_align_model(args.language, device_str)

    # Process each segment
    pred_strs: List[str] = []
    ref_strs: List[str] = []
    predictions: List[dict] = []
    total_t = 0.0

    for file_idx, (audio_path, file_entries) in enumerate(by_file.items()):
        audio = whisperx.load_audio(audio_path)
        sr = 16000

        for entry in file_entries:
            # Extract segment if start/end present
            if "start" in entry and "end" in entry:
                start_frame = int(math.floor(entry["start"] * sr))
                end_frame = int(math.ceil(entry["end"] * sr))
                segment_audio = audio[start_frame:end_frame]
            else:
                segment_audio = audio

            t0 = time.perf_counter()
            result = pipeline.transcribe(
                segment_audio,
                batch_size=args.batch_size,
                language=args.language,
            )
            total_t += time.perf_counter() - t0

            # Optional alignment
            if align_model is not None:
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    segment_audio,
                    device_str,
                )

            segments = result.get("segments", [])
            hyp_text = " ".join(s.get("text", "").strip() for s in segments)

            pred_strs.append(hyp_text)
            ref_strs.append(entry["text"].strip())
            predictions.append({
                "audio_filepath": entry["audio_filepath"],
                **({"start": entry["start"], "end": entry["end"]}
                   if "start" in entry else {}),
                "text": hyp_text,
            })

        if (file_idx + 1) % 10 == 0:
            print(f"  [{file_idx+1}/{len(by_file)} files]")

    # Compute WER
    norm_refs = [normalize_text(r) for r in ref_strs]
    norm_preds = [normalize_text(p) for p in pred_strs]
    wer = jiwer.wer(norm_refs, norm_preds)
    measures = jiwer.process_words(norm_refs, norm_preds)

    label = args.label or f"{args.whisper_arch}_{args.compute_type}"

    print(f"\n===== Results: {label} =====")
    print(f"  Samples:     {len(ref_strs)}")
    print(f"  Norm WER:    {wer:.4f} ({wer*100:.1f}%)")
    print(f"  S/I/D/H:     {measures.substitutions}/{measures.insertions}/{measures.deletions}/{measures.hits}")
    print(f"  Total time:  {total_t:.1f}s")
    print(f"  RTF:         {total_t / (sum(len(p) for p in pred_strs) / 16000 + 1e-9):.3f}")

    # Save predictions
    if args.save_predictions:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pred_path = RESULTS_DIR / f"predictions_{label}.jsonl"
        with open(pred_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()

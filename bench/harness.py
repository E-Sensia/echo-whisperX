"""Core benchmark runner: load model, run transcriptions, time stages, collect metrics."""

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from bench.config import BenchConfig
from bench.data import (
    AudioSample,
    discover_all_files,
    extract_reference_text,
    load_reference,
    select_sample,
)
from bench.metrics import aggregate_quality_metrics, compute_wer

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class FileResult:
    file_id: str
    audio_duration: float
    t_load: float
    t_asr: float
    t_align: float
    gpu_peak_mb: float
    wer: float
    wer_detail: dict
    n_segments: int
    quality_metrics: dict


@dataclass
class BenchResult:
    config_name: str
    config_description: str
    timestamp: str
    total_wall_clock: float
    total_audio_duration: float
    rtf: float
    rtf_asr_only: float
    gpu_peak_mb: float
    mean_wer: float
    median_wer: float
    n_files: int
    per_file: list[dict]
    quality_summary: dict
    config: dict


def _gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def run_single_file(
    pipeline,
    sample: AudioSample,
    config: BenchConfig,
    align_model=None,
    align_metadata=None,
) -> FileResult:
    """Benchmark a single audio file through the pipeline."""
    import whisperx

    # 1. Load audio
    t0 = time.perf_counter()
    audio = whisperx.load_audio(sample.wav_path)
    t_load = time.perf_counter() - t0
    audio_duration = len(audio) / 16000

    # 2. ASR (VAD + transcription)
    _reset_gpu_stats()
    t0 = time.perf_counter()
    result = pipeline.transcribe(
        audio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        language=config.language,
    )
    t_asr = time.perf_counter() - t0
    gpu_after_asr = _gpu_peak_mb()

    # 3. Alignment (optional)
    t_align = 0.0
    if not config.skip_alignment and align_model is not None:
        t0 = time.perf_counter()
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            config.device,
        )
        t_align = time.perf_counter() - t0

    gpu_peak = max(gpu_after_asr, _gpu_peak_mb())

    # 4. Quality metrics from transcription segments
    segments = result.get("segments", [])
    quality = aggregate_quality_metrics(segments)

    # 5. WER against reference
    ref = load_reference(sample)
    ref_text = extract_reference_text(ref)
    hyp_text = " ".join(s.get("text", "").strip() for s in segments)
    wer_result = compute_wer(ref_text, hyp_text)

    return FileResult(
        file_id=sample.call_id,
        audio_duration=audio_duration,
        t_load=t_load,
        t_asr=t_asr,
        t_align=t_align,
        gpu_peak_mb=gpu_peak,
        wer=wer_result["wer"],
        wer_detail=wer_result,
        n_segments=len(segments),
        quality_metrics=quality,
    )


def run_benchmark(config: BenchConfig) -> BenchResult:
    """Run the full benchmark for a given configuration."""
    import whisperx

    print(f"\n{'='*60}")
    print(f"  Step: {config.name}")
    print(f"  {config.description}")
    print(f"{'='*60}")

    # Discover and select sample files
    print(f"Discovering audio files...")
    all_files = discover_all_files()
    print(f"  Found {len(all_files)} files total")
    samples = select_sample(all_files, n=config.n_files, seed=config.sample_seed)
    print(f"  Selected {len(samples)} files (seed={config.sample_seed})")

    total_duration = sum(s.duration for s in samples)
    print(f"  Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Load model
    print(f"\nLoading model: {config.whisper_arch} ({config.compute_type}, {config.vad_method})...")
    t0 = time.perf_counter()
    load_kwargs = dict(
        whisper_arch=config.whisper_arch,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        vad_method=config.vad_method,
        asr_options=config.asr_options(),
    )
    pipeline = whisperx.load_model(**load_kwargs)
    t_model_load = time.perf_counter() - t0
    print(f"  Model loaded in {t_model_load:.1f}s")

    # Load alignment model if needed
    align_model = None
    align_metadata = None
    if not config.skip_alignment:
        print(f"Loading alignment model...")
        t0 = time.perf_counter()
        align_model, align_metadata = whisperx.load_align_model(
            config.language, config.device
        )
        print(f"  Alignment model loaded in {time.perf_counter() - t0:.1f}s")

    # Warmup: run one file to prime GPU kernels
    print(f"\nWarmup run...")
    _ = run_single_file(pipeline, samples[0], config, align_model, align_metadata)

    # Benchmark each file
    print(f"\nBenchmarking {len(samples)} files...")
    _reset_gpu_stats()
    file_results: list[FileResult] = []
    bench_start = time.perf_counter()

    for i, sample in enumerate(samples):
        fr = run_single_file(pipeline, sample, config, align_model, align_metadata)
        file_results.append(fr)
        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            avg_rtf = sum(r.t_asr for r in file_results) / sum(r.audio_duration for r in file_results)
            print(f"  [{i+1}/{len(samples)}] RTF(ASR)={avg_rtf:.3f} WER={fr.wer:.3f} ({fr.audio_duration:.1f}s audio in {fr.t_asr:.2f}s)")

    total_wall = time.perf_counter() - bench_start
    gpu_peak = _gpu_peak_mb()

    # Aggregate results
    total_audio = sum(r.audio_duration for r in file_results)
    total_asr = sum(r.t_asr for r in file_results)
    total_align = sum(r.t_align for r in file_results)
    wers = [r.wer for r in file_results]
    wers_sorted = sorted(wers)

    result = BenchResult(
        config_name=config.name,
        config_description=config.description,
        timestamp=datetime.now().isoformat(),
        total_wall_clock=total_wall,
        total_audio_duration=total_audio,
        rtf=total_wall / total_audio if total_audio > 0 else 0,
        rtf_asr_only=total_asr / total_audio if total_audio > 0 else 0,
        gpu_peak_mb=gpu_peak,
        mean_wer=sum(wers) / len(wers) if wers else 0,
        median_wer=wers_sorted[len(wers_sorted) // 2] if wers_sorted else 0,
        n_files=len(file_results),
        per_file=[asdict(r) for r in file_results],
        quality_summary=aggregate_quality_metrics(
            [s for r in file_results for s in _get_segments_from_result(r)]
        ),
        config=asdict(config),
    )

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  RTF (total):    {result.rtf:.4f}")
    print(f"  RTF (ASR only): {result.rtf_asr_only:.4f}")
    print(f"  ASR time:       {total_asr:.1f}s")
    print(f"  Align time:     {total_align:.1f}s")
    print(f"  GPU peak:       {result.gpu_peak_mb:.0f} MB")
    print(f"  Mean WER:       {result.mean_wer:.4f}")
    print(f"  Median WER:     {result.median_wer:.4f}")
    print(f"{'─'*60}")

    # Save results
    _save_result(result)

    # Cleanup
    del pipeline
    if align_model is not None:
        del align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _get_segments_from_result(fr: FileResult) -> list[dict]:
    """Build pseudo-segment dicts from quality_metrics for aggregation."""
    # quality_metrics is already aggregated per-file, we need raw segments
    # Since we don't store raw segments, return the per-file quality as a single entry
    qm = fr.quality_metrics
    seg = {}
    for key, stats in qm.items():
        if isinstance(stats, dict) and "mean" in stats:
            seg[key] = stats["mean"]
    return [seg] if seg else []


def _save_result(result: BenchResult):
    """Save benchmark result as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{result.config_name}_{result.timestamp.replace(':', '-')}.json"
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved to {path}")

"""Live simulation: N concurrent calls sending 15s chunks via ThreadPoolExecutor.

Supports two modes:
- ThreadPool (default): each call runs transcribe() in its own thread
- Cross-call batching: collects chunks from all calls, batches them via transcribe_multi()
"""

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from bench.config import BenchConfig
from bench.data import AudioSample, discover_all_files, select_sample

RESULTS_DIR = Path(__file__).parent / "results"
SAMPLE_RATE = 16000


@dataclass
class ChunkResult:
    call_id: int
    chunk_idx: int
    chunk_duration: float
    latency: float
    over_budget: bool


@dataclass
class SimulationResult:
    config_name: str
    timestamp: str
    n_calls: int
    chunk_duration: float
    total_chunks: int
    chunks_over_budget: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    max_latency: float
    throughput_chunks_per_sec: float
    total_wall_clock: float
    total_audio_processed: float
    effective_rtf: float
    per_chunk: list[dict]
    config: dict


def _slice_into_chunks(audio: np.ndarray, chunk_duration: float) -> list[np.ndarray]:
    """Slice audio into fixed-length chunks."""
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) > SAMPLE_RATE:  # skip chunks shorter than 1s
            chunks.append(chunk)
    return chunks


def _call_worker(
    call_id: int,
    chunks: list[np.ndarray],
    pipeline,
    config: BenchConfig,
) -> list[ChunkResult]:
    """Simulate one call: transcribe each chunk sequentially."""
    results = []
    for i, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        _ = pipeline.transcribe(
            chunk,
            batch_size=config.batch_size,
            language=config.language,
        )
        latency = time.perf_counter() - t0
        results.append(ChunkResult(
            call_id=call_id,
            chunk_idx=i,
            chunk_duration=len(chunk) / SAMPLE_RATE,
            latency=latency,
            over_budget=latency > config.chunk_duration,
        ))
    return results


def run_simulation(
    config: BenchConfig,
    n_calls_list: list[int],
) -> list[SimulationResult]:
    """Run live simulation with varying concurrency levels."""
    import whisperx

    print(f"\n{'='*60}")
    print(f"  Live Simulation: {config.name}")
    print(f"  Concurrency levels: {n_calls_list}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model...")
    pipeline = whisperx.load_model(
        config.whisper_arch,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        vad_method=config.vad_method,
        asr_options=config.asr_options(),
    )

    # Prepare audio chunks from sample files
    print(f"Preparing audio chunks...")
    all_files = discover_all_files()
    samples = select_sample(all_files, n=max(max(n_calls_list), config.n_files), seed=config.sample_seed)

    all_audios = []
    for s in samples:
        audio = whisperx.load_audio(s.wav_path)
        all_audios.append(audio)
    print(f"  Loaded {len(all_audios)} audio files")

    # Slice into chunks
    all_chunks = [_slice_into_chunks(a, config.chunk_duration) for a in all_audios]
    total_chunks_available = sum(len(c) for c in all_chunks)
    print(f"  Total chunks available: {total_chunks_available}")

    # Warmup
    print(f"Warmup...")
    _ = pipeline.transcribe(all_chunks[0][0], batch_size=config.batch_size, language=config.language)

    results = []
    for n_calls in n_calls_list:
        print(f"\n--- Simulating {n_calls} concurrent calls ---")

        # Assign chunks to calls (round-robin from available audios)
        call_chunks = []
        for i in range(n_calls):
            idx = i % len(all_chunks)
            call_chunks.append(all_chunks[idx])

        total_chunks = sum(len(c) for c in call_chunks)
        print(f"  Total chunks to process: {total_chunks}")

        chunk_results: list[ChunkResult] = []
        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_calls) as executor:
            futures = {
                executor.submit(_call_worker, i, call_chunks[i], pipeline, config): i
                for i in range(n_calls)
            }
            for future in as_completed(futures):
                call_id = futures[future]
                try:
                    call_results = future.result()
                    chunk_results.extend(call_results)
                except Exception as e:
                    print(f"  Call {call_id} failed: {e}")

        total_wall = time.perf_counter() - t_start

        # Aggregate
        latencies = sorted(r.latency for r in chunk_results)
        n_over = sum(1 for r in chunk_results if r.over_budget)
        total_audio = sum(r.chunk_duration for r in chunk_results)

        sim_result = SimulationResult(
            config_name=f"{config.name}_sim{n_calls}",
            timestamp=datetime.now().isoformat(),
            n_calls=n_calls,
            chunk_duration=config.chunk_duration,
            total_chunks=len(chunk_results),
            chunks_over_budget=n_over,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency=latencies[len(latencies) // 2] if latencies else 0,
            p95_latency=latencies[int(len(latencies) * 0.95)] if latencies else 0,
            max_latency=latencies[-1] if latencies else 0,
            throughput_chunks_per_sec=len(chunk_results) / total_wall if total_wall > 0 else 0,
            total_wall_clock=total_wall,
            total_audio_processed=total_audio,
            effective_rtf=total_wall / total_audio if total_audio > 0 else 0,
            per_chunk=[asdict(r) for r in chunk_results],
            config=asdict(config),
        )

        print(f"  Wall clock:      {total_wall:.1f}s")
        print(f"  Avg latency:     {sim_result.avg_latency:.3f}s")
        print(f"  P95 latency:     {sim_result.p95_latency:.3f}s")
        print(f"  Chunks over 15s: {n_over}/{len(chunk_results)} ({100*n_over/max(1,len(chunk_results)):.1f}%)")
        print(f"  Throughput:      {sim_result.throughput_chunks_per_sec:.2f} chunks/s")
        print(f"  Effective RTF:   {sim_result.effective_rtf:.4f}")

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{sim_result.config_name}_{sim_result.timestamp.replace(':', '-')}.json"
        path = RESULTS_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(sim_result), f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved to {path}")

        results.append(sim_result)

    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def run_cross_call_simulation(
    config: BenchConfig,
    n_calls_list: list[int],
) -> list[SimulationResult]:
    """Run cross-call batching simulation: batch chunks from N calls in one GPU pass."""
    import whisperx

    print(f"\n{'='*60}")
    print(f"  Cross-Call Batching Simulation: {config.name}")
    print(f"  Concurrency levels: {n_calls_list}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model...")
    pipeline = whisperx.load_model(
        config.whisper_arch,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        vad_method=config.vad_method,
        asr_options=config.asr_options(),
    )

    # Prepare audio chunks
    print(f"Preparing audio chunks...")
    all_files = discover_all_files()
    samples = select_sample(all_files, n=max(max(n_calls_list), config.n_files), seed=config.sample_seed)

    all_audios = []
    for s in samples:
        audio = whisperx.load_audio(s.wav_path)
        all_audios.append(audio)
    print(f"  Loaded {len(all_audios)} audio files")

    all_chunks = [_slice_into_chunks(a, config.chunk_duration) for a in all_audios]

    # Warmup
    print(f"Warmup...")
    _ = pipeline.transcribe(all_chunks[0][0], batch_size=config.batch_size, language=config.language)

    results = []
    for n_calls in n_calls_list:
        print(f"\n--- Cross-call batching: {n_calls} calls ---")

        # Collect one chunk per call (simulating the "every 15s" window)
        batch_chunks = []
        for i in range(n_calls):
            idx = i % len(all_chunks)
            if all_chunks[idx]:
                batch_chunks.append(all_chunks[idx][0])  # first chunk of each call

        print(f"  Batching {len(batch_chunks)} chunks in one transcribe_multi() call")

        t_start = time.perf_counter()
        multi_results = pipeline.transcribe_multi(
            batch_chunks,
            batch_size=config.batch_size,
            precompute_features=config.precompute_features,
            language=config.language,
        )
        total_wall = time.perf_counter() - t_start

        # Compute per-chunk latency (shared across the batch)
        per_chunk_latency = total_wall / len(batch_chunks) if batch_chunks else 0
        total_audio = sum(len(c) / SAMPLE_RATE for c in batch_chunks)

        chunk_results = [
            ChunkResult(
                call_id=i,
                chunk_idx=0,
                chunk_duration=len(batch_chunks[i]) / SAMPLE_RATE,
                latency=total_wall,  # all chunks share the same wall clock
                over_budget=total_wall > config.chunk_duration,
            )
            for i in range(len(batch_chunks))
        ]

        n_over = 1 if total_wall > config.chunk_duration else 0

        sim_result = SimulationResult(
            config_name=f"{config.name}_xcall{n_calls}",
            timestamp=datetime.now().isoformat(),
            n_calls=n_calls,
            chunk_duration=config.chunk_duration,
            total_chunks=len(batch_chunks),
            chunks_over_budget=n_over,
            avg_latency=per_chunk_latency,
            p50_latency=per_chunk_latency,
            p95_latency=total_wall,
            max_latency=total_wall,
            throughput_chunks_per_sec=len(batch_chunks) / total_wall if total_wall > 0 else 0,
            total_wall_clock=total_wall,
            total_audio_processed=total_audio,
            effective_rtf=total_wall / total_audio if total_audio > 0 else 0,
            per_chunk=[asdict(r) for r in chunk_results],
            config=asdict(config),
        )

        print(f"  Wall clock:         {total_wall:.3f}s (for all {n_calls} calls)")
        print(f"  Per-chunk latency:  {per_chunk_latency:.3f}s")
        print(f"  Over 15s budget:    {'YES' if n_over else 'NO'}")
        print(f"  Throughput:         {sim_result.throughput_chunks_per_sec:.2f} chunks/s")
        print(f"  Effective RTF:      {sim_result.effective_rtf:.4f}")

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{sim_result.config_name}_{sim_result.timestamp.replace(':', '-')}.json"
        path = RESULTS_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(sim_result), f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved to {path}")

        results.append(sim_result)

    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results

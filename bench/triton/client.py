"""Triton client for benchmarking the WhisperX model in real conditions.

Usage:
    # Single request
    python client.py --file /path/to/audio.wav

    # Concurrent calls simulation (N calls, each sending 15s chunks)
    python client.py --simulate --n-calls 10 20 50

    # Quick health check
    python client.py --health
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRITON_URL = "localhost:9201"
SAMPLE_RATE = 16000


def _set_triton_url(url):
    global TRITON_URL
    TRITON_URL = url
DATA_ROOT = Path(
    "/mnt/nas_echo_2/prod_data/dump_prod_20251202/splitted_test/20250601"
)


def rms_normalize(waveform, target_dbfs=-20.0, eps=1e-8):
    rms = np.sqrt(np.mean(waveform**2)).clip(min=eps)
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    return np.clip(waveform * gain, -1.0, 1.0)


def load_audio(path: str) -> np.ndarray:
    """Load audio via ffmpeg, same as whisperx.load_audio."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_triton_client():
    """Get a Triton gRPC client."""
    import tritonclient.grpc as grpcclient
    return grpcclient.InferenceServerClient(url=TRITON_URL)


def infer_single(client, audio: np.ndarray, language: str = "fr") -> dict:
    """Send a single transcription request to Triton."""
    import tritonclient.grpc as grpcclient

    audio_normalized = rms_normalize(audio)

    # Build inputs (batch dim first when max_batch_size > 0)
    audio_input = grpcclient.InferInput("audio", [1, len(audio_normalized)], "FP32")
    audio_input.set_data_from_numpy(audio_normalized.reshape(1, -1))

    params = json.dumps({"language": language, "enable_alignment": False})
    params_input = grpcclient.InferInput("parameters", [1, 1], "BYTES")
    params_input.set_data_from_numpy(
        np.array([[params.encode("utf-8")]], dtype=np.object_)
    )

    # Infer
    result = client.infer(
        model_name="whisperx",
        inputs=[audio_input, params_input],
        outputs=[grpcclient.InferRequestedOutput("segments")],
    )

    # Parse response (flatten to handle batch dim)
    segments_raw = result.as_numpy("segments").flatten()[0]
    if isinstance(segments_raw, bytes):
        segments_raw = segments_raw.decode("utf-8")
    return json.loads(segments_raw)


def health_check():
    """Check if Triton server is ready."""
    client = get_triton_client()
    live = client.is_server_live()
    ready = client.is_server_ready()
    print(f"Server live: {live}, ready: {ready}")
    if ready:
        model_ready = client.is_model_ready("whisperx")
        print(f"Model whisperx ready: {model_ready}")
    return ready


def run_single_file(path: str):
    """Transcribe a single file and print results."""
    client = get_triton_client()
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {path}")
    print(f"Duration: {duration:.1f}s")

    t0 = time.perf_counter()
    result = infer_single(client, audio)
    elapsed = time.perf_counter() - t0

    print(f"Latency: {elapsed:.3f}s (RTF: {elapsed/duration:.4f})")
    print(f"Language: {result.get('language')}")
    for seg in result.get("segments", []):
        print(f"  [{seg.get('start', 0):.1f}-{seg.get('end', 0):.1f}] {seg.get('text', '')}")


def discover_samples(n=30, seed=42):
    """Discover audio samples for simulation."""
    import random
    wavs = sorted(DATA_ROOT.glob("*/merged_audio_*/merged_audio_*.wav"))
    rng = random.Random(seed)
    return rng.sample(wavs, min(n, len(wavs)))


def slice_chunks(audio, chunk_sec=15.0):
    """Slice audio into fixed-length chunks."""
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) > SAMPLE_RATE:
            chunks.append(chunk)
    return chunks


def call_worker(call_id, chunks, language="fr"):
    """Worker: transcribe chunks sequentially for one call."""
    client = get_triton_client()
    results = []
    for i, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        resp = infer_single(client, chunk, language)
        latency = time.perf_counter() - t0
        results.append({
            "call_id": call_id,
            "chunk_idx": i,
            "chunk_duration": len(chunk) / SAMPLE_RATE,
            "latency": latency,
            "over_budget": latency > 15.0,
            "n_segments": len(resp.get("segments", [])),
        })
    return results


def run_simulation(n_calls_list, n_files=50):
    """Simulate concurrent calls against Triton server."""
    print(f"Discovering audio samples...")
    sample_paths = discover_samples(n=n_files)
    print(f"Loading {len(sample_paths)} audio files...")

    audios = [load_audio(str(p)) for p in sample_paths]
    all_chunks = [slice_chunks(a) for a in audios]
    print(f"Total chunks: {sum(len(c) for c in all_chunks)}")

    # Warmup
    print("Warmup...")
    client = get_triton_client()
    _ = infer_single(client, all_chunks[0][0])

    for n_calls in n_calls_list:
        print(f"\n--- {n_calls} concurrent calls ---")

        # Assign chunks round-robin
        call_chunks = [all_chunks[i % len(all_chunks)] for i in range(n_calls)]
        total = sum(len(c) for c in call_chunks)

        all_results = []
        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_calls) as executor:
            futures = {
                executor.submit(call_worker, i, call_chunks[i]): i
                for i in range(n_calls)
            }
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    print(f"  Call {futures[future]} failed: {e}")

        total_wall = time.perf_counter() - t_start
        latencies = sorted(r["latency"] for r in all_results)
        n_over = sum(1 for r in all_results if r["over_budget"])
        total_audio = sum(r["chunk_duration"] for r in all_results)

        print(f"  Chunks:     {len(all_results)}")
        print(f"  Wall clock: {total_wall:.1f}s")
        print(f"  Avg lat:    {sum(latencies)/len(latencies):.3f}s")
        print(f"  P95 lat:    {latencies[int(len(latencies)*0.95)]:.3f}s")
        print(f"  Max lat:    {latencies[-1]:.3f}s")
        print(f"  Over 15s:   {n_over}/{len(all_results)} ({100*n_over/max(1,len(all_results)):.1f}%)")
        print(f"  Throughput: {len(all_results)/total_wall:.2f} chunks/s")
        print(f"  Eff RTF:    {total_wall/total_audio:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Triton WhisperX client")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--file", type=str, help="Transcribe a single file")
    parser.add_argument(
        "--simulate", nargs="*", type=int, default=None,
        help="Simulate N concurrent calls (e.g., --simulate 5 10 20 50)",
    )
    parser.add_argument("--url", default=TRITON_URL, help="Triton gRPC URL")
    args = parser.parse_args()

    _set_triton_url(args.url)

    if args.health:
        health_check()
    elif args.file:
        run_single_file(args.file)
    elif args.simulate is not None:
        n_calls = args.simulate if args.simulate else [1, 5, 10, 20, 50]
        run_simulation(n_calls)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Discover audio samples and load reference transcripts."""

import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(
    "/mnt/nas_echo_2/prod_data/dump_prod_20251202/splitted_test/20250601"
)


@dataclass
class AudioSample:
    wav_path: str
    json_path: str
    call_id: str
    chunk_id: str
    duration: float  # seconds, estimated from file size then refined after load


def discover_all_files(base_dir: Path = DATA_ROOT) -> list[AudioSample]:
    """Walk the data directory and find all merged_audio WAV + JSON pairs."""
    samples = []
    for call_dir in sorted(base_dir.iterdir()):
        if not call_dir.is_dir():
            continue
        call_id = call_dir.name
        for sub in sorted(call_dir.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("merged_audio_"):
                continue
            wav = sub / f"{sub.name}.wav"
            ref_json = sub / f"{sub.name}.json"
            if not wav.exists() or not ref_json.exists():
                continue
            # Estimate duration from file size (16kHz, 16-bit mono PCM = 32000 bytes/s)
            # WAV header is 44 bytes; PCMU-encoded files may differ, so we'll refine later
            file_size = wav.stat().st_size
            estimated_duration = max(0.1, (file_size - 44) / 32000)
            samples.append(AudioSample(
                wav_path=str(wav),
                json_path=str(ref_json),
                call_id=call_id,
                chunk_id=sub.name.rsplit("_N", 1)[-1] if "_N" in sub.name else "1",
                duration=estimated_duration,
            ))
    return samples


def get_accurate_duration(wav_path: str) -> float:
    """Get accurate audio duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", wav_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def select_sample(
    all_files: list[AudioSample],
    n: int = 30,
    seed: int = 42,
) -> list[AudioSample]:
    """Select a stratified sample: short (<15s), medium (15-60s), long (>60s)."""
    rng = random.Random(seed)

    short = [s for s in all_files if s.duration < 15]
    medium = [s for s in all_files if 15 <= s.duration < 60]
    long = [s for s in all_files if s.duration >= 60]

    # Aim for ~1/3 each, fill remainder from largest bucket
    n_short = min(len(short), n // 3)
    n_medium = min(len(medium), n // 3)
    n_long = min(len(long), n - n_short - n_medium)

    # If a bucket has fewer items, redistribute
    remaining = n - n_short - n_medium - n_long
    for bucket, count_ref in [(medium, n_medium), (short, n_short), (long, n_long)]:
        if remaining <= 0:
            break
        extra = min(remaining, len(bucket) - count_ref)
        if bucket is short:
            n_short += extra
        elif bucket is medium:
            n_medium += extra
        else:
            n_long += extra
        remaining -= extra

    selected = (
        rng.sample(short, min(n_short, len(short)))
        + rng.sample(medium, min(n_medium, len(medium)))
        + rng.sample(long, min(n_long, len(long)))
    )

    # Refine durations with ffprobe
    for s in selected:
        accurate = get_accurate_duration(s.wav_path)
        if accurate > 0:
            s.duration = accurate

    return selected


def load_reference(sample: AudioSample) -> dict:
    """Load the reference JSON transcript."""
    with open(sample.json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_reference_text(ref: dict) -> str:
    """Concatenate all segment texts from the reference JSON."""
    texts = []
    for seg in ref.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)

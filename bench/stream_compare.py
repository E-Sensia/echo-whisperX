"""Compare fixed 15s chunking vs VAD-triggered chunking.

Mode A (current):  Fixed 15s chunks → transcribe_multi
Mode B (proposed): VAD pre-segments full audio into natural utterances (≤30s) → transcribe_multi

Both modes deliver accumulated text to downstream every 15s.
"""

import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SAMPLE_RATE = 16000
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class StreamModeResult:
    mode: str
    n_calls: int
    total_audio_s: float
    total_wall_s: float
    throughput_chunks_per_s: float
    audio_throughput_x_realtime: float
    n_encoder_passes: int
    mean_wer: float
    median_wer: float
    per_call_wer: list[float]
    encoder_passes_per_call: list[int]
    downstream_texts_at_15s: dict  # call_id → list of (tick_s, text)


def _vad_segment_audio(pipeline, audio: np.ndarray) -> list[tuple[float, float]]:
    """Run VAD on full audio and return natural speech boundaries."""
    waveform = torch.from_numpy(audio).unsqueeze(0)
    raw_segs = pipeline.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    merged = pipeline.vad_model.merge_chunks(
        raw_segs, chunk_size=30, onset=0.5, offset=0.363
    )
    return [(seg["start"], seg["end"]) for seg in merged]


def run_stream_comparison(
    n_calls: int = 50,
    seed: int = 42,
    batch_size: int = 16,
):
    """Run full comparison: fixed 15s vs VAD-triggered."""
    import whisperx

    from bench.data import (
        discover_all_files,
        extract_reference_text,
        load_reference,
        select_sample,
    )
    from bench.metrics import compute_wer

    print(f"\n{'='*70}")
    print(f"  Stream Mode Comparison: Fixed 15s vs VAD-triggered")
    print(f"  {n_calls} concurrent calls, batch_size={batch_size}")
    print(f"{'='*70}")

    # Load model
    print("\nLoading model...")
    pipeline = whisperx.load_model(
        "large-v3",
        device="cuda",
        compute_type="float16",
        language="fr",
        vad_method="pyannote",
        asr_options={"beam_size": 5},
    )

    # Load audio files
    print("Loading audio files...")
    all_files = discover_all_files()
    samples = select_sample(all_files, n=n_calls, seed=seed)

    call_data = []  # (call_id, full_audio, ref_text)
    for s in samples:
        audio = whisperx.load_audio(s.wav_path)
        ref = extract_reference_text(load_reference(s))
        call_data.append((s.call_id, audio, ref))

    total_audio = sum(len(cd[1]) / SAMPLE_RATE for cd in call_data)
    print(f"  {len(call_data)} calls, {total_audio:.0f}s total audio")

    # Warmup
    print("Warmup...")
    _ = pipeline.transcribe(call_data[0][1][:15 * SAMPLE_RATE], batch_size=batch_size, language="fr")

    # ═══════════════════════════════════════════════════════════════
    # MODE A: Fixed 15s chunks
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  MODE A: Fixed 15s chunks")
    print(f"{'─'*70}")

    mode_a_texts = {}  # call_id → full transcription
    mode_a_downstream = {}  # call_id → [(tick_s, cumulative_text)]
    mode_a_enc_passes = {}
    total_chunks_a = 0

    torch.cuda.synchronize()
    t_start_a = time.perf_counter()

    # Find max duration to know how many 15s windows we need
    max_dur = max(len(cd[1]) / SAMPLE_RATE for cd in call_data)
    n_windows = int(np.ceil(max_dur / 15.0))

    for win_idx in range(n_windows):
        t_win_start = win_idx * 15.0
        t_win_end = t_win_start + 15.0

        # Collect 15s chunks from all calls that have audio in this window
        batch_chunks = []
        batch_call_ids = []
        for call_id, audio, _ in call_data:
            f1 = int(t_win_start * SAMPLE_RATE)
            f2 = int(t_win_end * SAMPLE_RATE)
            if f1 >= len(audio):
                continue
            chunk = audio[f1:min(f2, len(audio))]
            if len(chunk) < SAMPLE_RATE:  # skip < 1s
                continue
            batch_chunks.append(chunk)
            batch_call_ids.append(call_id)

        if not batch_chunks:
            continue

        total_chunks_a += len(batch_chunks)

        # Cross-call batch transcription
        results = pipeline.transcribe_multi(
            batch_chunks, batch_size=batch_size, language="fr"
        )

        # Accumulate text per call
        for i, call_id in enumerate(batch_call_ids):
            segs = results[i].get("segments", [])
            chunk_text = " ".join(s.get("text", "").strip() for s in segs)

            if call_id not in mode_a_texts:
                mode_a_texts[call_id] = []
                mode_a_enc_passes[call_id] = 0
                mode_a_downstream[call_id] = []

            mode_a_texts[call_id].append(chunk_text)
            mode_a_enc_passes[call_id] += max(1, len(segs))

            # Downstream gets cumulative text at this tick
            cumulative = " ".join(mode_a_texts[call_id])
            mode_a_downstream[call_id].append((t_win_end, cumulative))

        if (win_idx + 1) % 5 == 0 or win_idx == n_windows - 1:
            print(f"  Window [{win_idx+1}/{n_windows}]: {len(batch_chunks)} calls active")

    torch.cuda.synchronize()
    t_total_a = time.perf_counter() - t_start_a

    # ═══════════════════════════════════════════════════════════════
    # MODE B: VAD-triggered (natural utterances up to 30s)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  MODE B: VAD-triggered (natural utterances ≤30s)")
    print(f"{'─'*70}")

    # Pre-compute VAD boundaries for all calls
    print("  Running VAD on all calls...")
    call_utterances = {}  # call_id → [(start, end)]
    for call_id, audio, _ in call_data:
        boundaries = _vad_segment_audio(pipeline, audio)
        call_utterances[call_id] = boundaries

    total_utterances = sum(len(u) for u in call_utterances.values())
    avg_utt_dur = np.mean([e - s for utts in call_utterances.values() for s, e in utts]) if total_utterances > 0 else 0
    print(f"  {total_utterances} utterances total, avg duration: {avg_utt_dur:.1f}s")

    mode_b_texts = {}
    mode_b_downstream = {}
    mode_b_enc_passes = {}
    total_chunks_b = 0

    torch.cuda.synchronize()
    t_start_b = time.perf_counter()

    # Process utterances in time order, batching those that end in the same window
    # Simulate: at each 15s tick, all utterances that completed get transcribed
    # An utterance is "ready" when its end time has passed
    for win_idx in range(n_windows):
        tick = (win_idx + 1) * 15.0

        # Collect utterances that end before this tick and haven't been processed
        batch_audios = []
        batch_call_ids = []

        for call_id, audio, _ in call_data:
            if call_id not in mode_b_texts:
                mode_b_texts[call_id] = []
                mode_b_enc_passes[call_id] = 0
                mode_b_downstream[call_id] = []

            utts = call_utterances.get(call_id, [])
            # Find utterances that end within this 15s window
            prev_tick = win_idx * 15.0
            for utt_start, utt_end in utts:
                # Utterance ends in this window (after prev_tick, before or at tick)
                if prev_tick < utt_end <= tick:
                    f1 = int(utt_start * SAMPLE_RATE)
                    f2 = int(utt_end * SAMPLE_RATE)
                    utt_audio = audio[f1:f2]
                    if len(utt_audio) >= SAMPLE_RATE:
                        batch_audios.append(utt_audio)
                        batch_call_ids.append(call_id)

        if batch_audios:
            total_chunks_b += len(batch_audios)

            # Cross-call batch transcription
            results = pipeline.transcribe_multi(
                batch_audios, batch_size=batch_size, language="fr"
            )

            for i, call_id in enumerate(batch_call_ids):
                segs = results[i].get("segments", [])
                utt_text = " ".join(s.get("text", "").strip() for s in segs)
                mode_b_texts[call_id].append(utt_text)
                mode_b_enc_passes[call_id] += max(1, len(segs))

        # Record downstream state at this tick for all active calls
        for call_id, _, _ in call_data:
            if call_id in mode_b_texts and mode_b_texts[call_id]:
                cumulative = " ".join(mode_b_texts[call_id])
                if call_id not in mode_b_downstream:
                    mode_b_downstream[call_id] = []
                mode_b_downstream[call_id].append((tick, cumulative))

        if (win_idx + 1) % 5 == 0 or win_idx == n_windows - 1:
            print(f"  Tick [{win_idx+1}/{n_windows}]: {len(batch_audios)} utterances processed")

    torch.cuda.synchronize()
    t_total_b = time.perf_counter() - t_start_b

    # ═══════════════════════════════════════════════════════════════
    # COMPUTE WER FOR BOTH MODES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("  Computing WER...")
    print(f"{'─'*70}")

    wers_a = []
    wers_b = []
    for call_id, _, ref_text in call_data:
        hyp_a = " ".join(mode_a_texts.get(call_id, []))
        hyp_b = " ".join(mode_b_texts.get(call_id, []))

        w_a = compute_wer(ref_text, hyp_a)["wer"] if ref_text and hyp_a else 0
        w_b = compute_wer(ref_text, hyp_b)["wer"] if ref_text and hyp_b else 0
        wers_a.append(w_a)
        wers_b.append(w_b)

    enc_a = sum(mode_a_enc_passes.values())
    enc_b = sum(mode_b_enc_passes.values())

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30} {'15s fixed':>14} {'VAD-triggered':>14} {'Delta':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Total ASR time':<30} {t_total_a:>13.1f}s {t_total_b:>13.1f}s {(t_total_b-t_total_a)/t_total_a*100:>+9.1f}%")
    print(f"  {'Chunks/utterances processed':<30} {total_chunks_a:>14} {total_chunks_b:>14} {(total_chunks_b-total_chunks_a)/total_chunks_a*100:>+9.1f}%")
    print(f"  {'Encoder passes':<30} {enc_a:>14} {enc_b:>14} {(enc_b-enc_a)/enc_a*100:>+9.1f}%")
    print(f"  {'Audio throughput (x RT)':<30} {total_audio/t_total_a:>13.0f}x {total_audio/t_total_b:>13.0f}x")
    print(f"  {'Mean WER':<30} {np.mean(wers_a):>14.4f} {np.mean(wers_b):>14.4f}")
    print(f"  {'Median WER':<30} {np.median(wers_a):>14.4f} {np.median(wers_b):>14.4f}")

    # Per-call WER comparison
    better = sum(1 for a, b in zip(wers_a, wers_b) if b < a)
    same = sum(1 for a, b in zip(wers_a, wers_b) if b == a)
    worse = sum(1 for a, b in zip(wers_a, wers_b) if b > a)
    print(f"\n  Per-call WER: {better} better, {same} same, {worse} worse with VAD-triggered")

    # Show worst degradations
    deltas = [(wers_b[i] - wers_a[i], call_data[i][0][:12], wers_a[i], wers_b[i])
              for i in range(len(call_data))]
    deltas.sort(key=lambda x: -x[0])
    print(f"\n  Top WER changes (VAD - Fixed):")
    for delta, cid, wa, wb in deltas[:5]:
        print(f"    {cid}: {wa:.4f} → {wb:.4f} ({delta:+.4f})")
    print(f"  ...")
    for delta, cid, wa, wb in deltas[-3:]:
        print(f"    {cid}: {wa:.4f} → {wb:.4f} ({delta:+.4f})")

    # Save results
    result = {
        "timestamp": datetime.now().isoformat(),
        "n_calls": n_calls,
        "batch_size": batch_size,
        "total_audio_s": total_audio,
        "mode_a_fixed_15s": {
            "total_wall_s": t_total_a,
            "total_chunks": total_chunks_a,
            "encoder_passes": enc_a,
            "audio_throughput_x_rt": total_audio / t_total_a,
            "mean_wer": float(np.mean(wers_a)),
            "median_wer": float(np.median(wers_a)),
            "per_call_wer": wers_a,
        },
        "mode_b_vad_triggered": {
            "total_wall_s": t_total_b,
            "total_utterances": total_chunks_b,
            "encoder_passes": enc_b,
            "audio_throughput_x_rt": total_audio / t_total_b,
            "mean_wer": float(np.mean(wers_b)),
            "median_wer": float(np.median(wers_b)),
            "per_call_wer": wers_b,
            "avg_utterance_duration": avg_utt_dur,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"stream_compare_{datetime.now().isoformat().replace(':', '-')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {path}")

    # Cleanup
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare fixed 15s vs VAD-triggered streaming")
    parser.add_argument("--n-calls", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stream_comparison(
        n_calls=args.n_calls,
        batch_size=args.batch_size,
        seed=args.seed,
    )

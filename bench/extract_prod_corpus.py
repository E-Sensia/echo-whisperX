"""Extract text corpus from prod Whisper transcripts for KenLM training.

Filters segments by quality metrics to avoid training on hallucinations.

Usage:
    python bench/extract_prod_corpus.py --days 5
    python bench/extract_prod_corpus.py --days 50 --output models/corpus_prod_50d.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

PROD_BASE = "/mnt/nas_echo_2/prod_data/dump_prod_20251202/splitted_test"


def normalize_for_lm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def is_good_segment(seg: dict) -> bool:
    """Filter out low-quality segments."""
    cr = seg.get("compression_ratio", 0)
    alp = seg.get("avg_logprob", 0)
    nsp = seg.get("no_speech_prob", 0)
    tps = seg.get("tokens_per_sec", 0)
    raf = seg.get("repeat_adjacent_frac", 0)

    if cr > 2.0:
        return False
    if alp < -0.7:
        return False
    if nsp > 0.5:
        return False
    if tps > 8.0:  # suspiciously fast → likely hallucination
        return False
    if raf > 0.3:  # too many repeated adjacent tokens
        return False

    text = seg.get("text", "").strip()
    if len(text) < 5:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5,
                        help="Number of days to process")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prod_base", type=str, default=PROD_BASE)
    args = parser.parse_args()

    days = sorted(os.listdir(args.prod_base))[:args.days]
    print(f"Processing {len(days)} days: {days[0]} - {days[-1]}")

    sentences = []
    total_segs = 0
    kept_segs = 0

    for day_idx, day in enumerate(days):
        day_path = os.path.join(args.prod_base, day)
        jsons = glob.glob(os.path.join(day_path, "**/merged_audio_*.json"), recursive=True)

        for jpath in jsons:
            with open(jpath) as f:
                data = json.load(f)

            for seg in data.get("segments", []):
                total_segs += 1
                if not is_good_segment(seg):
                    continue
                kept_segs += 1

                text = seg["text"].strip()
                for sent in split_sentences(text):
                    norm = normalize_for_lm(sent)
                    if len(norm.split()) >= 3:
                        sentences.append(norm)

        print(f"  [{day_idx+1}/{len(days)}] {day}: "
              f"{kept_segs}/{total_segs} segs, {len(sentences)} sentences")

    # Deduplicate exact matches (reduces noise from repeated phrases)
    unique = list(dict.fromkeys(sentences))
    total_words = sum(len(s.split()) for s in unique)

    output = args.output or f"models/corpus_prod_{args.days}d.txt"
    Path(output).parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        for s in unique:
            f.write(s + "\n")

    print(f"\nDone: {len(unique)} unique sentences ({total_words} words)")
    print(f"  from {kept_segs}/{total_segs} segments "
          f"({kept_segs/max(1,total_segs)*100:.0f}% kept)")
    print(f"  saved to {output}")


if __name__ == "__main__":
    main()

"""Prepare corpus and train a KenLM model from ground-truth transcripts.

Usage:
    python bench/train_kenlm.py
    python bench/train_kenlm.py --order 5 --output models/medical_fr_5gram.arpa
    python bench/train_kenlm.py --groundtruth /path/to/data.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ECHO_ASR_ROOT = Path("/home/thibault/Documents/projets/echo-asr")
DEFAULT_GT = ECHO_ASR_ROOT / "data" / "groundtruth_clean.jsonl"
LMPLZ = "/tmp/kenlm_build/build/bin/lmplz"
BUILD_BINARY = "/tmp/kenlm_build/build/bin/build_binary"


def normalize_for_lm(text: str) -> str:
    """Normalize text for LM training: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Keep apostrophes within words (l'attention, j'ai) but remove other punctuation
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split transcript into sentence-like chunks on sentence-ending punctuation."""
    # Split on . ! ? followed by space or end-of-string
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train KenLM from ground-truth transcripts")
    parser.add_argument("--groundtruth", type=str, default=str(DEFAULT_GT),
                        help="Path to JSONL with 'text' field")
    parser.add_argument("--order", type=int, default=4,
                        help="N-gram order (default: 4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for .arpa file (default: models/medical_fr_{order}gram.arpa)")
    parser.add_argument("--binary", action="store_true",
                        help="Also produce a .bin file (faster loading)")
    parser.add_argument("--lmplz", type=str, default=LMPLZ)
    parser.add_argument("--build_binary", type=str, default=BUILD_BINARY)
    args = parser.parse_args()

    # Load and prepare corpus
    entries = [json.loads(line) for line in open(args.groundtruth) if line.strip()]
    print(f"Loaded {len(entries)} entries from {args.groundtruth}")

    # Split into sentences then normalize — gives KenLM better n-gram boundaries
    sentences = []
    for e in entries:
        for sent in split_sentences(e["text"]):
            norm = normalize_for_lm(sent)
            if len(norm.split()) >= 2:  # skip single-word fragments
                sentences.append(norm)

    total_words = sum(len(s.split()) for s in sentences)
    print(f"  {len(sentences)} sentences, {total_words} words")

    # Write corpus
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    corpus_path = models_dir / "corpus_medical_fr.txt"
    with open(corpus_path, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"  Corpus written to {corpus_path}")

    # Train with lmplz
    output_arpa = args.output or str(models_dir / f"medical_fr_{args.order}gram.arpa")
    print(f"\nTraining {args.order}-gram KenLM model...")

    cmd = [
        args.lmplz,
        "-o", str(args.order),
        "--discount_fallback",  # robust smoothing for small corpora
    ]
    with open(corpus_path) as corpus_f, open(output_arpa, "w") as arpa_f:
        result = subprocess.run(cmd, stdin=corpus_f, stdout=arpa_f, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"ERROR: lmplz failed:\n{result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    arpa_size = Path(output_arpa).stat().st_size / 1024 / 1024
    print(f"  ARPA model: {output_arpa} ({arpa_size:.1f} MB)")

    # Optional: convert to binary for faster loading
    if args.binary:
        output_bin = output_arpa.replace(".arpa", ".bin")
        print(f"  Converting to binary: {output_bin}")
        result = subprocess.run(
            [args.build_binary, output_arpa, output_bin],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"WARNING: build_binary failed:\n{result.stderr.decode()}", file=sys.stderr)
        else:
            bin_size = Path(output_bin).stat().st_size / 1024 / 1024
            print(f"  Binary model: {output_bin} ({bin_size:.1f} MB)")

    # Quick sanity check
    print("\nSanity check:")
    try:
        import kenlm
        model = kenlm.Model(output_arpa)
        test_phrases = [
            "la tension artérielle",
            "l'attention artérielle",
            "douleur thoracique",
            "douleur toracique",
            "j'ai de la toux",
            "j'ai de la tout",
        ]
        for phrase in test_phrases:
            score = model.score(normalize_for_lm(phrase), bos=True, eos=True)
            print(f"  {score:8.3f}  {phrase}")
    except ImportError:
        print("  (kenlm Python not available, skipping)")

    print(f"\nDone. Use with: --lm_model_path {output_arpa}")


if __name__ == "__main__":
    main()

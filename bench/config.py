from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchConfig:
    name: str
    description: str = ""

    # Model
    whisper_arch: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "fr"

    # ASR
    beam_size: int = 5
    no_repeat_ngram_size: int = 0

    # VAD
    vad_method: str = "pyannote"

    # Pipeline
    batch_size: int = 8
    num_workers: int = 0
    precompute_features: bool = False
    skip_alignment: bool = False

    # Sample selection
    n_files: int = 30
    sample_seed: int = 42

    # Live simulation
    concurrent_calls: int = 1
    chunk_duration: float = 15.0

    def asr_options(self) -> dict:
        opts = {"beam_size": self.beam_size}
        if self.no_repeat_ngram_size > 0:
            opts["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return opts


# ── Predefined step configurations ──────────────────────────────────────────

STEPS: dict[str, BenchConfig] = {
    "0": BenchConfig(
        name="step0_baseline",
        description="Baseline: beam=5, float16, pyannote, batch=8, align=on",
    ),
    "1a": BenchConfig(
        name="step1a_beam2",
        description="beam_size=2",
        beam_size=2,
    ),
    "1b": BenchConfig(
        name="step1b_beam1",
        description="beam_size=1 (greedy)",
        beam_size=1,
    ),
    "2": BenchConfig(
        name="step2_int8fp16",
        description="compute_type=int8_float16",
        compute_type="int8_float16",
    ),
    "3": BenchConfig(
        name="step3_silero",
        description="VAD silero (CPU) instead of pyannote (GPU)",
        vad_method="silero",
    ),
    "4a": BenchConfig(
        name="step4a_batch16",
        description="batch_size=16",
        batch_size=16,
    ),
    "4b": BenchConfig(
        name="step4b_batch32",
        description="batch_size=32",
        batch_size=32,
    ),
    "5": BenchConfig(
        name="step5_no_align",
        description="Skip alignment (no word-level timestamps)",
        skip_alignment=True,
    ),
    "6": BenchConfig(
        name="step6_combined",
        description="Best combo: beam=5, float16, pyannote, batch=16, no align",
        batch_size=16,
        skip_alignment=True,
    ),
    "8": BenchConfig(
        name="step8_precompute",
        description="Precompute mel spectrograms on CPU threads before GPU inference",
        batch_size=16,
        skip_alignment=True,
        precompute_features=True,
    ),
    "10a": BenchConfig(
        name="step10a_batch24",
        description="batch_size=24 (better GPU saturation for decoder)",
        batch_size=24,
        skip_alignment=True,
    ),
    "10b": BenchConfig(
        name="step10b_batch48",
        description="batch_size=48 (max GPU saturation)",
        batch_size=48,
        skip_alignment=True,
    ),
    "11": BenchConfig(
        name="step11_no_repeat_ngram",
        description="no_repeat_ngram_size=2 (blocks bigram repeats, decoder exits earlier)",
        batch_size=16,
        skip_alignment=True,
        no_repeat_ngram_size=2,
    ),
    "12": BenchConfig(
        name="step12_combined_v2",
        description="Best combo v2: batch=24, no_repeat_ngram=2, no align",
        batch_size=24,
        skip_alignment=True,
        no_repeat_ngram_size=2,
    ),
}

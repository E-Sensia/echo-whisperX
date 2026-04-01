# Echo-WhisperX

Fork of WhisperX optimised for Echo's telephony ASR pipeline (French medical emergency calls, PCMU 8kHz).

## ASR Quality Benchmark

160 hand-transcribed segments are maintained in the sibling project:
- **Reference manifest**: `/home/thibault/Documents/projets/echo-asr/data/test_manifest.jsonl`
- Format: JSONL with `audio_filepath`, `text`, optional `start`/`end` segment offsets
- Audio: 16kHz WAV (upsampled from PCMU 8kHz)

### Running the benchmark

```bash
# From this project root:
python bench/eval_whisperx.py                    # default: large-v3, baseline
python bench/eval_whisperx.py --compute_type int8_float16  # test quantization
python bench/eval_whisperx.py --label "my_change" # label for comparison
```

### Baseline WER (whisper-large-v3-turbo, no fine-tuning, generation)
- **Norm WER: 35.4%** (S=827 I=959 D=2601 H=8960)
- With hp80+nr audio preprocessing: **33.4%** (S=802 I=617 D=2724 H=8862)

### Comparing results
```bash
# Compare a predictions JSONL against reference
python /home/thibault/Documents/projets/echo-asr/scripts/bench_wer.py \
    --predictions results/my_predictions.jsonl \
    --reference /home/thibault/Documents/projets/echo-asr/data/test_manifest.jsonl
```

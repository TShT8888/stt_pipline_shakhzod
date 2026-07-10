# STT Data Preparation Pipeline

Reproducible data preparation pipeline for Uzbek/Russian speech recognition experiments.

Current implemented stages:

- JSONL streaming IO utilities.
- Text normalization for Uzbek Latin, Uzbek Cyrillic, and Russian/Cyrillic text.
- VAD over unlabeled audio with Silero VAD.

Planned next stage:

- Materialize VAD segments into real audio clips for pseudo-labeling and fine-tuning.

## Repository Layout

```text
data/
  raw/
    labeled/
      metadata.jsonl
      audio/*.wav
    unlabeled/
      manifest.jsonl
      audio/*.wav
  interim/
    vad/
      unlabeled_segments.jsonl

scripts/
  run_vad_unlabeled.py

src/
  data/
    jsonl.py
    text_normalization.py
    vad.py

tests/
```

`data/` is intentionally ignored by git. Keep code in git, keep datasets/artifacts outside git.

## Local Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
```

Run checks:

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check src tests scripts
```

## VAD

VAD detects speech regions inside long/unlabeled audio files.

Input:

```text
data/raw/unlabeled/manifest.jsonl
data/raw/unlabeled/audio/*.wav
```

Default output:

```text
data/interim/vad/unlabeled_segments.jsonl
```

Each output row is a speech segment with provenance:

```json
{
  "segment_id": "unlabeled_xxx_vad_00001_00",
  "source_audio_id": "unlabeled_xxx",
  "source_manifest": "data/raw/unlabeled/manifest.jsonl",
  "source_line_number": 1,
  "source_audio_path": "data/raw/unlabeled/audio/unlabeled_xxx.wav",
  "vad_start": 1.5,
  "vad_end": 6.756,
  "vad_duration": 5.256,
  "vad_model": "silero-vad"
}
```

The VAD stage does **not** physically cut audio files yet. It stores offsets:

```text
source_audio_path + vad_start + vad_end
```

This is fast and keeps the first stage lightweight. Physical audio clips should be created in the next stage before pseudo-labeling/fine-tuning.

### Run VAD Locally

CPU is the default because Silero VAD is small and often faster/more stable on CPU than GPU for one file at a time.

```bash
.venv/bin/python scripts/run_vad_unlabeled.py
```

Equivalent explicit command:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --device cpu \
  --max-threads 1
```

Optional audit files:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --output-summary data/interim/vad/unlabeled_audio_summary.jsonl \
  --output-metadata data/interim/vad/unlabeled_run_metadata.json
```

`output-summary` contains one row per source audio.

`output-metadata` contains run-level config and benchmark metrics such as:

- `processing_seconds`
- `audio_duration`
- `real_time_factor`
- `peak_rss_mb`
- `num_errors`

### Sharded VAD

For large datasets, run independent shards instead of one huge process:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --num-shards 4 \
  --shard-index 0 \
  --output-segments data/interim/vad/unlabeled_segments_0.jsonl
```

Repeat with `--shard-index 1`, `2`, and `3`.

Use different output filenames per shard.

## Colab GPU Check

The goal in Colab is not to prove GPU is faster. The goal is to verify that the code can run on CUDA and compare CPU vs GPU with `real_time_factor`.

### 1. Enable GPU

In Colab:

```text
Runtime -> Change runtime type -> GPU
```

### 2. Clone The Repository

```bash
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
```

### 3. Check CUDA

```bash
!python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### 4. Install Dependencies

```bash
!python -m pip install -U pip
!python -m pip install -e '.[dev]'
```

### 5. Copy Or Mount Data

Option A: mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then copy data into the repo layout:

```bash
!mkdir -p data/raw/unlabeled
!cp -r /content/drive/MyDrive/stt_data/raw/unlabeled/* data/raw/unlabeled/
```

Expected structure:

```text
data/raw/unlabeled/manifest.jsonl
data/raw/unlabeled/audio/*.wav
```

### 6. Run CPU Baseline

```bash
!python scripts/run_vad_unlabeled.py \
  --device cpu \
  --max-threads 1 \
  --output-segments data/interim/vad/unlabeled_segments_cpu.jsonl \
  --output-metadata data/interim/vad/vad_cpu_metadata.json
```

### 7. Run CUDA Check

```bash
!python scripts/run_vad_unlabeled.py \
  --device cuda \
  --max-threads 1 \
  --output-segments data/interim/vad/unlabeled_segments_cuda.jsonl \
  --output-metadata data/interim/vad/vad_cuda_metadata.json
```

### 8. Compare CPU And GPU

```bash
!cat data/interim/vad/vad_cpu_metadata.json
!cat data/interim/vad/vad_cuda_metadata.json
```

Compare:

- `num_errors`
- `num_segments`
- `processing_seconds`
- `real_time_factor`
- `peak_rss_mb`

If CUDA is slower than CPU, that is normal for Silero VAD on small/medium files. Keep VAD on CPU and reserve GPU for ASR pseudo-labeling and training.

## Next Stage: Materialize VAD Audio Clips

For fine-tuning, it is usually better to materialize VAD segments into real audio files before pseudo-labeling/training.

Current VAD output:

```text
source_audio_path + vad_start + vad_end
```

Expected next output:

```text
data/processed/unlabeled_vad/audio/*.flac
data/processed/unlabeled_vad/manifest.jsonl
```

Expected manifest row:

```json
{
  "audio_id": "unlabeled_xxx_vad_00001_00",
  "audio_path": "audio/unlabeled_xxx_vad_00001_00.flac",
  "source_audio_id": "unlabeled_xxx",
  "source_audio_path": "data/raw/unlabeled/audio/unlabeled_xxx.wav",
  "source_start": 1.5,
  "source_end": 6.756,
  "duration": 5.256,
  "sample_rate": 16000,
  "channels": 1,
  "language": "uz",
  "dataset": "yakhyo/mozilla-common-voice-uzbek"
}
```

This materialized manifest is the right input for:

1. ASR pseudo-labeling.
2. Audio quality filtering.
3. Final fine-tuning manifest construction.

Planned command:

```bash
.venv/bin/python scripts/materialize_vad_segments.py \
  --segments data/interim/vad/unlabeled_segments.jsonl \
  --output-dir data/processed/unlabeled_vad \
  --format flac
```

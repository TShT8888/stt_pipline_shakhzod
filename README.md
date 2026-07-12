# STT Data Preparation Pipeline

Репозиторий для подготовки данных под fine-tuning STT-моделей на узбекском и русском.

Главная идея: код хранится в Git, а большие аудио-данные и промежуточные артефакты
лежат в `data/` и не коммитятся. Пайплайн должен быть воспроизводимым: каждый этап
читает явный входной manifest/JSONL и пишет явный выходной manifest/JSONL.

## Что уже есть

- потоковое чтение и запись JSONL/JSON;
- общий нормализатор текста для Uzbek Latin, Uzbek Cyrillic и Russian Cyrillic;
- VAD для неразмеченных аудио на базе Silero VAD;
- материализация VAD-сегментов в отдельные `.flac` или `.wav` файлы;
- overlap features для labeled и materialized unlabeled аудио;
- audio quality features для labeled и materialized unlabeled аудио;
- тесты на ключевые части пайплайна.

## Структура репозитория

```text
scripts/
  run_vad_unlabeled.py        # CLI для запуска VAD по manifest неразмеченных аудио
  materialize_vad_segments.py # CLI для нарезки VAD-сегментов в реальные аудио-файлы
  run_overlap.py              # CLI для подсчета overlapped speech features
  run_audio_quality.py        # CLI для подсчета audio quality features

src/
  data/
    jsonl.py                  # общие функции чтения/записи JSONL и JSON
    manifest.py               # общие helpers для audio manifests
    runtime.py                # общие torch/runtime helpers для stage-скриптов
    text_normalization.py     # общий нормализатор текста
    vad.py                    # основная логика VAD
    materialize.py            # основная логика нарезки аудио по VAD-сегментам
    overlap.py                # основная логика подсчета overlap features
    audio_quality.py          # основная логика подсчета audio quality features

tests/
  test_audio_quality.py       # тесты audio quality stage
  test_text_normalization.py  # тесты нормализации текста
  test_vad.py                 # тесты VAD-логики без тяжелого реального прогона модели
  test_materialize.py         # тесты нарезки аудио и manifest-выхода
  test_overlap.py             # тесты overlap-логики без загрузки тяжелой модели

pyproject.toml                # зависимости, dev-зависимости, настройки pytest/ruff
README.md                     # описание пайплайна
```

Локальная структура данных ожидается такая:

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
  processed/
    unlabeled_vad/
      manifest.jsonl
      audio/*.flac
```

Папка `data/` игнорируется Git. Архивы `*.zip` тоже игнорируются, чтобы большие данные
случайно не попали в репозиторий.

## Контракт данных

Все manifest-файлы должны быть в формате JSONL: одна строка - один JSON-объект.
Пустые строки пропускаются. Относительный `audio_path` всегда считается относительно
папки, где лежит сам manifest.

### Raw unlabeled manifest

Файл:

```text
data/raw/unlabeled/manifest.jsonl
```

Используется этапом VAD.

Обязательные поля:

```text
audio_id
audio_path
```

Опциональные поля:

```text
duration
sample_rate
channels
dataset
language
split
source_clips
```

Минимальный валидный пример:

```json
{"audio_id":"unlabeled_001","audio_path":"audio/unlabeled_001.wav"}
```

Рекомендуемый пример:

```json
{"audio_id":"unlabeled_001","audio_path":"audio/unlabeled_001.wav","duration":87.656,"sample_rate":16000,"channels":1,"dataset":"example/dataset","language":"uz"}
```

Если `language`, `dataset`, `split` или похожих полей нет, пайплайн не упадет:
в downstream JSONL эти значения будут `null`. Если нет `audio_id` или `audio_path`,
VAD не сможет обработать строку.

### Raw labeled manifest

Файл:

```text
data/raw/labeled/metadata.jsonl
```

Используется напрямую этапами `overlap` и `audio_quality`.

Обязательные поля:

```text
audio_id
audio_path
duration
```

Опциональные поля:

```text
text
sample_rate
channels
dataset
language
split
```

Минимальный валидный пример:

```json
{"audio_id":"labeled_001","audio_path":"audio/labeled_001.wav","duration":4.788}
```

Рекомендуемый пример:

```json
{"audio_id":"labeled_001","audio_path":"audio/labeled_001.wav","duration":4.788,"sample_rate":16000,"channels":1,"dataset":"example/dataset","language":"uz","split":"train","text":"Salom dunyo"}
```

`text` нужен будущему train-manifest, но текущие `overlap` и `audio_quality` его не
требуют. Если вместо `language` будет `lang`, код не упадет, но поле `language` в
feature JSONL будет `null`. Если вместо `audio_path` будет `path`, текущий код
упадет, потому что `audio_path` - обязательное поле контракта.

### VAD segments manifest

Файл:

```text
data/interim/vad/unlabeled_segments.jsonl
```

Создается `scripts/run_vad_unlabeled.py` и используется `scripts/materialize_vad_segments.py`.

Обязательные поля для materialize:

```text
segment_id
source_audio_path
vad_start
vad_end
```

Эти поля генерируются самим VAD stage. Руками такой manifest обычно писать не нужно.

Пример:

```json
{"segment_id":"unlabeled_001_vad_00000_00","source_audio_id":"unlabeled_001","source_audio_path":"data/raw/unlabeled/audio/unlabeled_001.wav","vad_start":0.0,"vad_end":6.956,"vad_duration":6.956,"language":"uz","dataset":"example/dataset"}
```

### Materialized VAD clips manifest

Файл:

```text
data/processed/unlabeled_vad/manifest.jsonl
```

Создается `scripts/materialize_vad_segments.py`. Дальше используется так же, как
labeled manifest: его можно отдавать в `overlap`, `audio_quality`, pseudo-labeling
и будущий selector.

Обязательные поля:

```text
audio_id
audio_path
duration
```

Опциональные, но полезные поля:

```text
sample_rate
channels
dataset
language
source_audio_id
source_audio_path
source_start
source_end
vad_run_id
```

Пример:

```json
{"audio_id":"unlabeled_001_vad_00000_00","audio_path":"audio/unlabeled_001_vad_00000_00.flac","duration":6.956,"sample_rate":16000,"channels":1,"language":"uz","dataset":"example/dataset","source_audio_id":"unlabeled_001","source_start":0.0,"source_end":6.956}
```

### Feature manifests

Feature-файлы не заменяют исходные manifest-ы. Каждый stage пишет отдельный JSONL:

```text
data/features/overlap/*.jsonl
data/features/audio_quality/*.jsonl
```

Общее обязательное поле для join:

```text
audio_id
```

Идея такая: исходный manifest остается неизменным, а будущий selector объединит
`manifest + overlap + audio_quality + pseudo_labels` по `audio_id` и решит, какие
строки идут в fine-tuning.

## Установка

Нужен Python 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
```

Проверка кода:

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check src tests scripts
```

## Быстрый порядок запуска

Для первого smoke-test лучше прогнать 10 исходных unlabeled аудио:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/debug_10/segments.jsonl \
  --output-summary data/interim/vad/debug_10/summary.jsonl \
  --output-metadata data/interim/vad/debug_10/metadata.json \
  --device cpu \
  --max-threads 1 \
  --limit 10

.venv/bin/python scripts/materialize_vad_segments.py \
  --segments data/interim/vad/debug_10/segments.jsonl \
  --output-dir data/processed/unlabeled_vad_debug_10 \
  --output-metadata data/processed/unlabeled_vad_debug_10/materialize_metadata.json \
  --format flac \
  --overwrite

.venv/bin/python scripts/run_overlap.py \
  --input-manifest data/processed/unlabeled_vad_debug_10/manifest.jsonl \
  --output-features data/features/overlap/unlabeled_vad_debug_10_overlap_10.jsonl \
  --output-metadata data/features/overlap/unlabeled_vad_debug_10_overlap_10_metadata.json \
  --device cpu \
  --limit 10

.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/processed/unlabeled_vad_debug_10/manifest.jsonl \
  --output-features data/features/audio_quality/unlabeled_vad_debug_10_quality_10.jsonl \
  --output-metadata data/features/audio_quality/unlabeled_vad_debug_10_quality_10_metadata.json \
  --limit 10
```

Для labeled данных VAD и materialize не нужны, потому что аудио уже размечены:

```bash
.venv/bin/python scripts/run_overlap.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/overlap/labeled_overlap_10.jsonl \
  --output-metadata data/features/overlap/labeled_overlap_10_metadata.json \
  --device cpu \
  --limit 10

.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/audio_quality/labeled_quality_10.jsonl \
  --output-metadata data/features/audio_quality/labeled_quality_10_metadata.json \
  --limit 10
```

Полный unlabeled запуск отличается только отсутствием `--limit` и production-путями:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --output-summary data/interim/vad/unlabeled_summary.jsonl \
  --output-metadata data/interim/vad/unlabeled_metadata.json \
  --device cpu \
  --max-threads 1

.venv/bin/python scripts/materialize_vad_segments.py \
  --segments data/interim/vad/unlabeled_segments.jsonl \
  --output-dir data/processed/unlabeled_vad \
  --output-metadata data/processed/unlabeled_vad/materialize_metadata.json \
  --format flac \
  --overwrite

.venv/bin/python scripts/run_overlap.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/overlap/unlabeled_vad_overlap.jsonl \
  --output-metadata data/features/overlap/unlabeled_vad_overlap_metadata.json \
  --device auto

.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/audio_quality/unlabeled_vad_quality.jsonl \
  --output-metadata data/features/audio_quality/unlabeled_vad_quality_metadata.json
```

`run_overlap.py` требует доступ к HuggingFace gated-моделям pyannote. Локально можно
положить токен в `.env`:

```text
HF_TOKEN=hf_...
```

## Этап 1. Нормализация текста

Файл: `src/data/text_normalization.py`

Нормализатор приводит текст к стабильному виду для STT-разметки:

- приводит Unicode к каноническому виду;
- унифицирует апострофы и похожие символы;
- приводит текст к lowercase;
- сохраняет русскую кириллицу, узбекскую кириллицу и узбекскую латиницу;
- убирает лишнюю пунктуацию и повторяющиеся пробелы.

Это общий нормализатор. Он не требует заранее детектить язык строки, поэтому подходит
для смешанных Uzbek/Russian данных.

Проверить тестами:

```bash
.venv/bin/python -m pytest tests/test_text_normalization.py
```

## Этап 2. VAD для неразмеченных аудио

Файлы:

- `scripts/run_vad_unlabeled.py` - CLI-обертка;
- `src/data/vad.py` - основная логика.

VAD находит участки речи внутри длинных или неразмеченных аудио. На этом этапе аудио
еще не режется физически. Выход содержит только координаты сегментов:

```text
source_audio_path + vad_start + vad_end
```

Так первый этап остается быстрым и легким по памяти.

Вход:

```text
data/raw/unlabeled/manifest.jsonl
data/raw/unlabeled/audio/*.wav
```

Запуск по умолчанию:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py
```

Явный запуск:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --device cpu \
  --max-threads 1
```

Debug-запуск только на первых 10 строках manifest:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/debug_10/segments.jsonl \
  --output-summary data/interim/vad/debug_10/summary.jsonl \
  --output-metadata data/interim/vad/debug_10/metadata.json \
  --device cpu \
  --max-threads 1 \
  --limit 10
```

Дополнительно можно сохранить audit-файлы:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --output-summary data/interim/vad/unlabeled_audio_summary.jsonl \
  --output-metadata data/interim/vad/unlabeled_run_metadata.json \
  --device cpu \
  --max-threads 1
```

`output-summary` пишет одну строку на исходный аудио-файл.

`output-metadata` пишет статистику запуска:

- количество входных строк;
- количество обработанных аудио;
- количество найденных сегментов;
- количество ошибок;
- длительность аудио;
- время обработки;
- real-time factor;
- peak RSS memory;
- параметры VAD.

Пример строки `unlabeled_segments.jsonl`:

```json
{
  "segment_id": "unlabeled_xxx_vad_00001_00",
  "source_audio_id": "unlabeled_xxx",
  "source_manifest": "data/raw/unlabeled/manifest.jsonl",
  "source_line_number": 1,
  "source_audio_path": "data/raw/unlabeled/audio/unlabeled_xxx.wav",
  "source_audio_relpath": "audio/unlabeled_xxx.wav",
  "dataset": "example-dataset",
  "language": "uz",
  "source_duration": 87.656,
  "source_sample_rate": 16000,
  "source_channels": 1,
  "vad_start": 8.372,
  "vad_end": 16.78,
  "vad_duration": 8.408,
  "vad_model": "silero-vad",
  "vad_threshold": 0.5
}
```

### Sharding

Для больших датасетов можно запускать несколько независимых shard-процессов:

```bash
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --num-shards 4 \
  --shard-index 0 \
  --output-segments data/interim/vad/unlabeled_segments_0.jsonl
```

Потом повторить с `--shard-index 1`, `2`, `3`. Для каждого shard нужен свой
`--output-segments`.

## Этап 3. Нарезка аудио по VAD-сегментам

Файлы:

- `scripts/materialize_vad_segments.py` - CLI-обертка;
- `src/data/materialize.py` - основная логика.

Этот этап читает `unlabeled_segments.jsonl`, группирует сегменты по исходному аудио,
загружает каждый исходный файл один раз и сохраняет найденные VAD-участки как отдельные
аудио-файлы. Это уже удобный формат для pseudo-labeling, фильтрации качества и
fine-tuning.

Запуск:

```bash
.venv/bin/python scripts/materialize_vad_segments.py \
  --segments data/interim/vad/unlabeled_segments.jsonl \
  --output-dir data/processed/unlabeled_vad \
  --output-metadata data/processed/unlabeled_vad/materialize_metadata.json \
  --format flac \
  --overwrite
```

Выход:

```text
data/processed/unlabeled_vad/
  manifest.jsonl
  materialize_metadata.json
  audio/
    *.flac
```

Пример строки `data/processed/unlabeled_vad/manifest.jsonl`:

```json
{
  "audio_id": "unlabeled_xxx_vad_00001_00",
  "audio_path": "audio/unlabeled_xxx_vad_00001_00.flac",
  "source_audio_id": "unlabeled_xxx",
  "source_audio_path": "data/raw/unlabeled/audio/unlabeled_xxx.wav",
  "source_start": 8.372,
  "source_end": 16.78,
  "duration": 8.408,
  "sample_rate": 16000,
  "channels": 1,
  "language": "uz",
  "dataset": "example-dataset"
}
```

Проверить результат:

```bash
wc -l data/processed/unlabeled_vad/manifest.jsonl
find data/processed/unlabeled_vad/audio -type f | head
head -n 3 data/processed/unlabeled_vad/manifest.jsonl
```

## Этап 4. Overlap features

Файлы:

- `scripts/run_overlap.py` - CLI-обертка;
- `src/data/overlap.py` - основная логика.

Этот этап считает, сколько overlapped speech есть в каждом аудио. Входной manifest
должен иметь стандартные поля пайплайна:

```text
audio_id
audio_path
duration
```

Такой формат подходит для:

- labeled данных: `data/raw/labeled/metadata.jsonl`;
- нарезанных unlabeled VAD clips: `data/processed/unlabeled_vad/manifest.jsonl`.

Каждая feature хранится отдельным JSONL-файлом. Quality/audio metrics в будущем тоже
будут писаться отдельными feature-файлами, а финальный selector будет join-ить их по
`audio_id`.

Запуск для labeled:

```bash
HF_TOKEN=... .venv/bin/python scripts/run_overlap.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/overlap/labeled_overlap.jsonl \
  --output-metadata data/features/overlap/labeled_overlap_metadata.json \
  --device auto \
  --limit 10
```

Запуск для materialized unlabeled VAD clips:

```bash
HF_TOKEN=... .venv/bin/python scripts/run_overlap.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/overlap/unlabeled_vad_overlap.jsonl \
  --output-metadata data/features/overlap/unlabeled_vad_overlap_metadata.json \
  --device auto \
  --limit 10
```

Пример строки feature JSONL:

```json
{
  "audio_id": "unlabeled_xxx_vad_00001_00",
  "feature_type": "overlap",
  "feature_version": "1.0",
  "status": "ok",
  "duration": 8.408,
  "overlap_duration": 0.75,
  "overlap_ratio": 0.089201,
  "num_overlap_segments": 2,
  "overlap_segments": [
    {"start": 1.0, "end": 1.35, "duration": 0.35},
    {"start": 4.2, "end": 4.6, "duration": 0.4}
  ]
}

```

## Этап 5. Audio quality features

Файлы:

- `scripts/run_audio_quality.py` - CLI-обертка;
- `src/data/audio_quality.py` - основная логика.

Этот этап считает быстрые технические признаки качества аудио без ASR/ML-модели:

- `rms_dbfs`;
- `peak_dbfs`;
- `noise_floor_dbfs`;
- `snr_estimate_db`;
- `silence_ratio`;
- `active_ratio`;
- `clipping_ratio`;
- `dc_offset_mean_abs`;
- `zero_crossing_rate`;
- `quality_score`.

Запуск для labeled:

```bash
.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/audio_quality/labeled_quality_10.jsonl \
  --output-metadata data/features/audio_quality/labeled_quality_10_metadata.json \
  --limit 10
```

Запуск для materialized unlabeled VAD clips:

```bash
.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/audio_quality/unlabeled_vad_quality_10.jsonl \
  --output-metadata data/features/audio_quality/unlabeled_vad_quality_10_metadata.json \
  --limit 10
```

Пример строки feature JSONL:

```json
{
  "audio_id": "unlabeled_xxx_vad_00001_00",
  "feature_type": "audio_quality",
  "feature_version": "1.0",
  "status": "ok",
  "duration": 8.408,
  "rms_dbfs": -22.5,
  "peak_dbfs": -3.1,
  "snr_estimate_db": 18.2,
  "silence_ratio": 0.12,
  "clipping_ratio": 0.0,
  "quality_score": 0.93
}
```

## Рекомендуемый порядок работы

1. Подготовить `data/raw/unlabeled/manifest.jsonl`.
2. Запустить VAD и получить `data/interim/vad/unlabeled_segments.jsonl`.
3. Проверить metadata: `num_errors`, `num_segments`, `real_time_factor`.
4. Нарезать сегменты через `materialize_vad_segments.py`.
5. Проверить `data/processed/unlabeled_vad/manifest.jsonl` и несколько аудио-файлов.
6. Посчитать overlap features для labeled и materialized unlabeled аудио.
7. Посчитать audio quality features для labeled и materialized unlabeled аудио.
8. Следующим этапом запускать pseudo-labeling ASR-моделью.
9. После pseudo-labeling считать качество текста и собирать финальный training manifest.

## Что запускать перед commit

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check src tests scripts
git status --short
```

## Следующие этапы пайплайна

Планируемые модули после VAD и нарезки:

- pseudo-labeling для неразмеченных VAD-клипов;
- фильтрация плохих сегментов;
- сбор финальных train/valid/test manifests для fine-tuning.

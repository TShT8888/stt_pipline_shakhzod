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
- music detection (детекция музыки) для labeled и materialized unlabeled аудио;
- отбор данных: selection для SSL (unlabeled) и split train/val/test (labeled);
- тесты на ключевые части пайплайна.

## Структура репозитория

```text
scripts/
  run_vad_unlabeled.py        # CLI для запуска VAD по manifest неразмеченных аудио
  materialize_vad_segments.py # CLI для нарезки VAD-сегментов в реальные аудио-файлы
  run_overlap.py              # CLI для подсчета overlapped speech features
  run_audio_quality.py        # CLI для подсчета audio quality features
  run_music_detection.py      # CLI для детекции музыки
  run_select_ssl.py           # CLI для отбора unlabeled клипов под SSL
  run_split_labeled.py        # CLI для фильтрации labeled и сплита train/val/test

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
    music_detection.py        # основная логика детекции музыки (AST/AudioSet)
    selection.py              # общие аудио-гейты, join фич, score, хэш-сплит
    ssl_selection.py          # отбор unlabeled клипов под SSL
    labeled_split.py          # фильтрация labeled + сплит train/val/test

tests/
  test_audio_quality.py       # тесты audio quality stage
  test_text_normalization.py  # тесты нормализации текста
  test_vad.py                 # тесты VAD-логики без тяжелого реального прогона модели
  test_materialize.py         # тесты нарезки аудио и manifest-выхода
  test_overlap.py             # тесты overlap-логики без загрузки тяжелой модели
  test_music_detection.py     # тесты music-detection без загрузки тяжелой модели
  test_selection.py           # тесты гейтов, score и сплита (без аудио и моделей)

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
data/features/music/*.jsonl
```

Общее обязательное поле для join:

```text
audio_id
```

Идея такая: исходный manifest остается неизменным, а selector объединяет
`manifest + overlap + audio_quality + music` по `audio_id` и решает, какие строки
идут в обучение (см. «Этап 7. Отбор данных для обучения»).

## Установка

Нужен Python 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
```

Этап `music_detection` дополнительно использует `transformers` и `torchaudio`
(`torchaudio` также приходит вместе с `pyannote.audio`). Первый запуск детектора
скачивает чекпойнт AST (~350 МБ).

За корпоративным прокси с TLS-инспекцией скачивание с HuggingFace (overlap, music)
может падать с `CERTIFICATE_VERIFY_FAILED`. Тогда укажите доверенный CA-бандл, например
в `.env`:

```text
REQUESTS_CA_BUNDLE=/absolute/path/to/corp-ca.pem
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

.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad_debug_10/manifest.jsonl \
  --output-features data/features/music/unlabeled_vad_debug_10_music_10.jsonl \
  --output-metadata data/features/music/unlabeled_vad_debug_10_music_10_metadata.json \
  --device cpu \
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

.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/music/labeled_music_10.jsonl \
  --output-metadata data/features/music/labeled_music_10_metadata.json \
  --device cpu \
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

.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/music/unlabeled_vad_music.jsonl \
  --output-metadata data/features/music/unlabeled_vad_music_metadata.json \
  --device auto \
  --batch-size 16
```

`run_overlap.py` требует доступ к HuggingFace gated-моделям pyannote. Локально можно
положить токен в `.env`:

```text
HF_TOKEN=hf_...
```

### Полный прогон: собрать данные для SSL и train/val/test

Реальные датасеты собираются без `--limit`, на полных манифестах. Две независимые
ветки. Предпосылки (разово): `pip install -e . --no-deps`; установлены
`silero-vad`, `pyannote.audio`, `transformers`, `torchaudio`; приняты условия
`pyannote/segmentation` на HuggingFace; за корпоративным прокси - CA-бандл в `.env`
(`REQUESTS_CA_BUNDLE=/путь/до/corp-ca.pem`).

**Ветка A. Unlabeled -> SSL:**

```bash
# 1. VAD -> координаты речи
.venv/bin/python scripts/run_vad_unlabeled.py \
  --input-manifest data/raw/unlabeled/manifest.jsonl \
  --output-segments data/interim/vad/unlabeled_segments.jsonl \
  --output-metadata data/interim/vad/unlabeled_metadata.json \
  --device cpu --max-threads 1

# 2. Materialize -> реальные клипы + manifest
.venv/bin/python scripts/materialize_vad_segments.py \
  --segments data/interim/vad/unlabeled_segments.jsonl \
  --output-dir data/processed/unlabeled_vad \
  --output-metadata data/processed/unlabeled_vad/materialize_metadata.json \
  --format flac --overwrite

# 3. Фичи на ВСЕХ клипах (без --limit)
.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/audio_quality/unlabeled_vad_quality.jsonl \
  --output-metadata data/features/audio_quality/unlabeled_vad_quality_metadata.json

.venv/bin/python scripts/run_overlap.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/overlap/unlabeled_vad_overlap.jsonl \
  --output-metadata data/features/overlap/unlabeled_vad_overlap_metadata.json \
  --device auto

.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/music/unlabeled_vad_music.jsonl \
  --output-metadata data/features/music/unlabeled_vad_music_metadata.json \
  --device auto --batch-size 16

# 4. Отбор -> плоский manifest для SSL
.venv/bin/python scripts/run_select_ssl.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --quality-features data/features/audio_quality/unlabeled_vad_quality.jsonl \
  --overlap-features data/features/overlap/unlabeled_vad_overlap.jsonl \
  --music-features  data/features/music/unlabeled_vad_music.jsonl \
  --output-selected data/selection/ssl_selected.jsonl \
  --output-rejected data/selection/ssl_rejected.jsonl \
  --output-metadata data/selection/ssl_metadata.json \
  --max-clips-per-source 100
```

**Ветка B. Labeled -> train/val/test** (VAD и materialize не нужны):

```bash
# 1. Фичи на ВСЕХ размеченных клипах (без --limit)
.venv/bin/python scripts/run_audio_quality.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/audio_quality/labeled_quality.jsonl \
  --output-metadata data/features/audio_quality/labeled_quality_metadata.json

.venv/bin/python scripts/run_overlap.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/overlap/labeled_overlap.jsonl \
  --output-metadata data/features/overlap/labeled_overlap_metadata.json \
  --device auto

.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/music/labeled_music.jsonl \
  --output-metadata data/features/music/labeled_music_metadata.json \
  --device auto --batch-size 16

# 2. Фильтр + сплит 80/10/10
.venv/bin/python scripts/run_split_labeled.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --quality-features data/features/audio_quality/labeled_quality.jsonl \
  --overlap-features data/features/overlap/labeled_overlap.jsonl \
  --music-features  data/features/music/labeled_music.jsonl \
  --output-dir data/selection/labeled_split \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Проверить, что и сколько собралось:

```bash
.venv/bin/python -c "import json; m=json.load(open('data/selection/ssl_metadata.json')); print('SSL:', m['num_selected'], 'клипов /', m['selected_hours'], 'ч; отказы:', m['reject_histogram'])"
.venv/bin/python -c "import json; m=json.load(open('data/selection/labeled_split/split_metadata.json')); print('split:', m['split_counts'], '; отказы:', m['reject_histogram'])"
```

Итог: `data/selection/ssl_selected.jsonl` -> SSL continued pretraining;
`data/selection/labeled_split/{train,val,test}.jsonl` -> supervised fine-tuning.
На CPU самый долгий шаг - music detection; на сервере ставь `--device auto` и при
больших данных шардируй (`--num-shards/--shard-index`).

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
8. Посчитать music detection для materialized unlabeled аудио и пометить `is_music`.
9. Отобрать unlabeled клипы под SSL (`run_select_ssl.py`) и разбить labeled на
   train/val/test (`run_split_labeled.py`) — см. «Этап 7. Отбор данных для обучения».
10. Запустить SSL continued pretraining на отобранном manifest; параллельно
    pseudo-labeling unlabeled клипов под последующий supervised fine-tuning.

## Что запускать перед commit

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check src tests scripts
git status --short
```

## Следующие этапы пайплайна

Планируемые модули (фильтрация и сборка train/val/test уже сделаны на «Этапе 7»):

- pseudo-labeling неразмеченных VAD-клипов (после SSL, под supervised fine-tuning);
- сбор финального train-manifest с псевдо-транскриптами и их фильтрация по качеству текста.


## Этап 6. Music detection

Файлы:

- `scripts/run_music_detection.py` - CLI-обертка;
- `src/data/music_detection.py` - основная логика.

Этот этап помечает аудио, в которых есть музыка. Silero VAD принимает пение и
вокал за речь, поэтому песни и музыкальные вставки проходят VAD и портят корпус
для SSL. Детектор считает вероятность верхнеуровневой AudioSet-метки `Music`
классификатором AST (`MIT/ast-finetuned-audioset-10-10-0.4593`).

Длинные клипы режутся на окна по 10 секунд, все окна одного клипа прогоняются
одним батчем, а per-clip сигнал берется как максимум по окнам. Модель ждет 16 кГц,
поэтому аудио автоматически ресемплится через soxr.

Входной manifest - стандартный для пайплайна:

```text
audio_id
audio_path
duration
```

Подходит для:

- labeled данных: `data/raw/labeled/metadata.jsonl`;
- нарезанных unlabeled VAD clips: `data/processed/unlabeled_vad/manifest.jsonl`.

Как и overlap/audio_quality, этот этап пишет отдельный feature JSONL, а финальный
selector join-ит его по `audio_id` и режет клипы с `is_music == true`.

Зависимости: этот этап требует `transformers` и `torchaudio` (см. «Установка»).
Первый запуск скачивает чекпойнт AST (~350 МБ).

Debug-запуск на первых 10 клипах:

```bash
.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/music/unlabeled_vad_music_10.jsonl \
  --output-metadata data/features/music/unlabeled_vad_music_10_metadata.json \
  --device cpu \
  --limit 10
```

Полный unlabeled запуск (на сервере с GPU лучше `--device auto` и больший batch):

```bash
.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --output-features data/features/music/unlabeled_vad_music.jsonl \
  --output-metadata data/features/music/unlabeled_vad_music_metadata.json \
  --device auto \
  --batch-size 16
```

Запуск для labeled данных:

```bash
.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --output-features data/features/music/labeled_music.jsonl \
  --output-metadata data/features/music/labeled_music_metadata.json \
  --device auto
```

Параметры:

- `--music-threshold` (0.5 по умолчанию) - порог `music_probability` для `is_music`.
  Прогоните debug-запуск, посмотрите `top_labels` у помеченных клипов и подстройте
  под свои данные (для эфиров/медиа часто лучше 0.4-0.6);
- `--window-seconds` (10.0) - размер окна, совпадает с «родным» окном AST;
- `--batch-size` (8) - число окон в одном forward;
- `--top-k-labels` (5) - сколько меток класть в `top_labels` для аудита.

### Sharding

Как и остальные этапы, поддерживает независимые shard-процессы:

```bash
.venv/bin/python scripts/run_music_detection.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --num-shards 4 \
  --shard-index 0 \
  --output-features data/features/music/unlabeled_vad_music_0.jsonl
```

Потом повторить с `--shard-index 1`, `2`, `3`, каждый в свой `--output-features`.

Пример строки feature JSONL:

```json
{
  "audio_id": "unlabeled_xxx_vad_00001_00",
  "feature_type": "music_detection",
  "feature_version": "1.0",
  "status": "ok",
  "duration": 8.408,
  "music_probability": 0.93,
  "music_mean_probability": 0.88,
  "music_ratio": 1.0,
  "is_music": true,
  "num_windows": 1,
  "top_labels": [
    {"label": "Music", "probability": 0.93},
    {"label": "Singing", "probability": 0.81}
  ]
}
```

## Этап 7. Отбор данных для обучения (SSL + train/val/test)

Файлы:

- `scripts/run_select_ssl.py` + `src/data/ssl_selection.py` - отбор unlabeled под SSL;
- `scripts/run_split_labeled.py` + `src/data/labeled_split.py` - фильтр labeled + сплит;
- `src/data/selection.py` - общие аудио-гейты, join фич, score, детерминированный хэш-сплит.

Финальный этап подготовки данных. Оба селектора **читают только JSONL**
(`manifest` + feature-файлы), не открывают аудио и не грузят модели, поэтому быстрые и
CPU-only. Джойн идёт по `audio_id`. Если feature-файл не передан, соответствующий гейт
пропускается (в CLI это включается автоматически по наличию флага).

### Как отбираем: аудио-гейты

Клип должен пройти **все** гейты, иначе уходит в `rejected.jsonl` со списком причин.
Пороги сбалансированные; каждый переопределяется флагом CLI.

| Признак | SSL (unlabeled) | Labeled (train/val/test) | Зачем |
|---|---|---|---|
| статус всех фич | ok | ok | битые/непосчитанные прочь |
| `duration` | 3.0–30.0 с | 0.8–30.0 с | SSL нужен контекст; labeled-клипы короткие |
| `sample_rate` | =16000 | =16000 | энкодер 16 кГц |
| `clipping_ratio` | ≤0.001 | ≤0.001 | искажения |
| `silence_ratio` | ≤0.5 | ≤0.6 | не пустышка |
| `rms_dbfs` | −45…−3 | −45…−3 | не шёпот/перегруз |
| `snr_estimate_db` | ≥8 | ≥5 | labeled щадим сильнее |
| `overlap_ratio` | ≤0.15 | ≤0.2 | одноголосье, без crosstalk |
| `is_music` | должно быть false | должно быть false | музыку/пение вон |
| `dc_offset_mean_abs` | ≤0.01 | ≤0.01 | битая конвертация |
| текст (только labeled) | — | непустой после нормализации, 3–25 симв/с | пустые/битые транскрипты |

Эффективные пороги при запуске через CLI берутся из аргументов скрипта (значения выше -
их дефолты). Меняешь порог - флагом или в `parse_args` соответствующего скрипта.

### SSL: отбор unlabeled клипов

Текст не нужен: SSL/continued pretraining учится из сырого аудио. Берём всё, что прошло
гейты (бюджет и diversity-кап по умолчанию выключены). Выход - плоский manifest, готовый
для SSL-загрузчика, плюс поля `selection_score` и `selection_run_id`.

Debug (на фичах, посчитанных с `--limit 10`):

```bash
.venv/bin/python scripts/run_select_ssl.py \
  --input-manifest data/processed/unlabeled_vad_debug_10/manifest.jsonl \
  --quality-features data/features/audio_quality/unlabeled_vad_debug_10_quality_10.jsonl \
  --overlap-features data/features/overlap/unlabeled_vad_debug_10_overlap_10.jsonl \
  --music-features data/features/music/unlabeled_vad_debug_10_music_10.jsonl \
  --output-selected data/selection/ssl_debug_10_selected.jsonl \
  --output-rejected data/selection/ssl_debug_10_rejected.jsonl \
  --output-metadata data/selection/ssl_debug_10_metadata.json
```

Полный прогон (шардированные фичи подставляются через glob; включаем diversity-кап):

```bash
.venv/bin/python scripts/run_select_ssl.py \
  --input-manifest data/processed/unlabeled_vad/manifest.jsonl \
  --quality-features data/features/audio_quality/unlabeled_vad_quality*.jsonl \
  --overlap-features data/features/overlap/unlabeled_vad_overlap*.jsonl \
  --music-features data/features/music/unlabeled_vad_music*.jsonl \
  --output-selected data/selection/ssl_selected.jsonl \
  --output-rejected data/selection/ssl_rejected.jsonl \
  --output-metadata data/selection/ssl_metadata.json \
  --max-clips-per-source 100
```

Полезные флаги: `--max-clips-per-source N` (не давать одной исходной записи забить
корпус), `--target-hours N` (взять топ-N часов по `quality_score`), `--languages uz ru`.

### Labeled: сплит train/val/test

Дополнительно к аудио-гейтам проверяется текст. Прошедшие клипы **детерминированно**
раскидываются по train/val/test:

- разбиение по hash от `--group-key` (дефолт `audio_id` - поклиповое);
- **защита от утечки диктора**: если в manifest есть поле спикера, задай
  `--group-key speaker_id` - все клипы одного диктора уйдут в один сплит;
- **стратификация** по `--stratify-keys` (дефолт `dataset language`) - пропорции
  языков/датасетов одинаковы во всех сплитах;
- воспроизводимо: тот же вход и `--salt` дают тот же сплит.

```bash
.venv/bin/python scripts/run_split_labeled.py \
  --input-manifest data/raw/labeled/metadata.jsonl \
  --quality-features data/features/audio_quality/labeled_quality_10.jsonl \
  --overlap-features data/features/overlap/labeled_overlap_10.jsonl \
  --music-features data/features/music/labeled_music_10.jsonl \
  --output-dir data/selection/labeled_split \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Выход:

```text
data/selection/labeled_split/
  train.jsonl
  val.jsonl
  test.jsonl
  rejected.jsonl        # что и почему отсеяли
  split_metadata.json   # счётчики, часы, гистограмма причин, разбивка по языкам
```

Строки train/val/test - это исходный labeled-формат (`audio_id`, `audio_path`,
`duration`, `text`, ...) плюс поля `split` и `split_run_id`. Готово для supervised
fine-tuning.

### Что читать в результатах

- `*_metadata.json` / `split_metadata.json`: `num_selected`, `reject_histogram`,
  `selected_hours` / `split_counts`. Много `missing_*` означает, что фичи посчитаны не
  на всех клипах (например, дебажный `--limit 10`) - на полном прогоне этого не будет.
- `rejected.jsonl`: по каждому отсеянному клипу - список `reject_reasons`.
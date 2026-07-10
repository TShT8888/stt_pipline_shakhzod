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
- тесты на ключевые части пайплайна.

## Структура репозитория

```text
scripts/
  run_vad_unlabeled.py        # CLI для запуска VAD по manifest неразмеченных аудио
  materialize_vad_segments.py # CLI для нарезки VAD-сегментов в реальные аудио-файлы

src/
  data/
    jsonl.py                  # общие функции чтения/записи JSONL и JSON
    text_normalization.py     # общий нормализатор текста
    vad.py                    # основная логика VAD
    materialize.py            # основная логика нарезки аудио по VAD-сегментам

tests/
  test_text_normalization.py  # тесты нормализации текста
  test_vad.py                 # тесты VAD-логики без тяжелого реального прогона модели
  test_materialize.py         # тесты нарезки аудио и manifest-выхода

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
  "abs_audio_path": "data/processed/unlabeled_vad/audio/unlabeled_xxx_vad_00001_00.flac",
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

## Рекомендуемый порядок работы

1. Подготовить `data/raw/unlabeled/manifest.jsonl`.
2. Запустить VAD и получить `data/interim/vad/unlabeled_segments.jsonl`.
3. Проверить metadata: `num_errors`, `num_segments`, `real_time_factor`.
4. Нарезать сегменты через `materialize_vad_segments.py`.
5. Проверить `data/processed/unlabeled_vad/manifest.jsonl` и несколько аудио-файлов.
6. Следующим этапом запускать pseudo-labeling ASR-моделью.
7. После pseudo-labeling считать качество аудио/текста и собирать финальный training manifest.

## Что запускать перед commit

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check src tests scripts
git status --short
```

## Следующие этапы пайплайна

Планируемые модули после VAD и нарезки:

- audio quality metrics для labeled и unlabeled аудио;
- overlap/speech metrics по каждому аудио;
- pseudo-labeling для неразмеченных VAD-клипов;
- фильтрация плохих сегментов;
- сбор финальных train/valid/test manifests для fine-tuning.

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TextNormalizationConfig:
    """Настройки безопасной текстовой нормализации перед ASR."""

    lowercase: bool = True
    normalize_unicode: bool = True
    remove_bracketed_text: bool = True
    remove_numbers: bool = False
    keep_word_apostrophe: bool = True


CHAR_MAP: Final[dict[str, str]] = {
    "\u00a0": " ",

    # Апострофы из узбекской латиницы и шумных расшифровок приводим к одному виду.
    "ʻ": "'",
    "ʼ": "'",
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",

    # Кавычки разных видов дальше будут удалены как пунктуация.
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',

    # Тире разных видов нормализуем перед удалением пунктуации.
    "–": "-",
    "—": "-",
    "−": "-",
}

BRACKETED_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}"
)

DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"\d+")
WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Пунктуацию удаляем, но апостроф обрабатываем отдельно.
PUNCT_RE: Final[re.Pattern[str]] = re.compile(
    r"""[!"#$%&()*+,./:;<=>?@\[\]^_`{|}~\\«»“”„…-]"""
)

# Апостроф оставляем только внутри слов:
#   o'zbek, g'arb, san'at
# Вокруг цитат апостроф должен удаляться:
#   'salom' -> salom
BAD_APOSTROPHE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ])'|'(?![0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ])"
)


def normalize_text(
    text: str,
    config: TextNormalizationConfig | None = None,
) -> str:
    """
    Нормализует текст без определения языка.

    Функция рассчитана на смешанные данные: узбекская латиница, узбекская
    кириллица и русский. Здесь нет транслитерации и исправления текста моделью:
    только детерминированная очистка, чтобы не менять смысл транскрипта.
    """
    if config is None:
        config = TextNormalizationConfig()

    text = text.strip()

    if config.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    for source, target in CHAR_MAP.items():
        text = text.replace(source, target)

    if config.lowercase:
        text = text.lower()

    if config.remove_bracketed_text:
        text = BRACKETED_TEXT_RE.sub(" ", text)

    if config.remove_numbers:
        text = DIGITS_RE.sub(" ", text)

    text = PUNCT_RE.sub(" ", text)

    if config.keep_word_apostrophe:
        text = BAD_APOSTROPHE_RE.sub(" ", text)
    else:
        text = text.replace("'", " ")

    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

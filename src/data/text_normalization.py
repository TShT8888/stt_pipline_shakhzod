from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TextNormalizationConfig:
    lowercase: bool = True
    normalize_unicode: bool = True
    remove_bracketed_text: bool = True
    remove_numbers: bool = False
    keep_word_apostrophe: bool = True


CHAR_MAP: Final[dict[str, str]] = {
    "\u00a0": " ",

    # Apostrophes used in Uzbek Latin and noisy transcripts.
    "ʻ": "'",
    "ʼ": "'",
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",

    # Quotes.
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',

    # Dashes.
    "–": "-",
    "—": "-",
    "−": "-",
}

BRACKETED_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}"
)

DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"\d+")
WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

# Remove punctuation, but apostrophe is handled separately.
PUNCT_RE: Final[re.Pattern[str]] = re.compile(
    r"""[!"#$%&()*+,./:;<=>?@\[\]^_`{|}~\\«»“”„…-]"""
)

# Apostrophe is valid only inside words:
#   o'zbek, g'arb, san'at
# It should be removed around quotes:
#   'salom' -> salom
BAD_APOSTROPHE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ])'|'(?![0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ])"
)


def normalize_text(
    text: str,
    config: TextNormalizationConfig | None = None,
) -> str:
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
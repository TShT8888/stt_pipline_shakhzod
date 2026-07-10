from src.data.text_normalization import TextNormalizationConfig, normalize_text


def test_normalizes_uzbek_apostrophes() -> None:
    assert normalize_text("Oʻzbek tili") == "o'zbek tili"
    assert normalize_text("Gʼarb san’ati") == "g'arb san'ati"


def test_removes_punctuation_but_keeps_apostrophe() -> None:
    assert normalize_text("'Salom', dedi u.") == "salom dedi u"
    assert normalize_text("Udan ko'ra balandroq joy bor.") == "udan ko'ra balandroq joy bor"


def test_removes_bracketed_noise() -> None:
    assert normalize_text("[noise] Assalomu alaykum!") == "assalomu alaykum"
    assert normalize_text("(шум) Это хороший звук.") == "это хороший звук"


def test_normalizes_spaces() -> None:
    assert normalize_text("  Bu   juda\n yaxshi.  ") == "bu juda yaxshi"


def test_preserves_numbers_by_default() -> None:
    assert normalize_text("2024-yil yaxshi yil") == "2024 yil yaxshi yil"


def test_normalizes_uzbek_cyrillic_text() -> None:
    assert normalize_text("Ўзбекистон — Қорақалпоғистон.") == "ўзбекистон қорақалпоғистон"
    assert normalize_text("Ғарб, Ҳуқуқ, Қўшиқ!") == "ғарб ҳуқуқ қўшиқ"


def test_uzbek_cyrillic_removes_apostrophe_punctuation() -> None:
    assert normalize_text("Бу 'иқтибос' эмас.") == "бу иқтибос эмас"


def test_normalizes_russian_text() -> None:
    assert normalize_text("Привет, мир!") == "привет мир"
    assert normalize_text("Ёжик шёл домой.") == "ёжик шёл домой"


def test_can_remove_numbers() -> None:
    config = TextNormalizationConfig(remove_numbers=True)
    assert normalize_text("2024-yil yaxshi yil", config) == "yil yaxshi yil"


def test_can_remove_all_apostrophes() -> None:
    config = TextNormalizationConfig(keep_word_apostrophe=False)
    assert normalize_text("o'zbek san'at") == "o'zbek san'at"
    assert normalize_text("o'zbek san'at", config) == "o zbek san at"

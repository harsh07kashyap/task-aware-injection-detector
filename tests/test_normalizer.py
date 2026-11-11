from tapid.normalizer import normalize, normalize_with_span


def test_removes_zero_width_and_homoglyphs():
    assert normalize("ig​nore аll") == "ignore all"   # zero-width space + Cyrillic 'а'


def test_full_width_letters():
    assert normalize("ＩＧＮＯＲＥ") == "IGNORE"


def test_span_moves_with_removed_characters():
    text = "Hi​ there. Ign​ore this. Bye."
    start = text.index("Ign")
    end = text.index("this.") + len("this.")
    clean, (s, e) = normalize_with_span(text, (start, end))
    assert clean[s:e] == "Ignore this."

"""M1 - Normaliser.

Undoes cheap evasion tricks before the text reaches the classifier:
  * Unicode NFKC normalisation (full-width letters, ligatures, etc.)
  * removal of zero-width / invisible characters and Unicode "tag" characters
  * mapping of common homoglyphs (Cyrillic / Greek look-alikes) to ASCII
"""

import re
import unicodedata

# Zero-width and other invisible code points often used to split trigger words.
_INVISIBLE = re.compile(
    "["
    "­"            # soft hyphen
    "᠎"            # mongolian vowel separator
    "​-‏"     # zero-width space / joiners / direction marks
    "‪-‮"     # bidi embedding controls
    "⁠-⁤"     # word joiner, invisible operators
    "﻿"            # zero-width no-break space (BOM)
    "]"
    "|[\U000e0000-\U000e007f]"  # Unicode tag characters ("ASCII smuggling")
)

# Look-alike characters -> ASCII. Kept small and explicit on purpose.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "І": "I",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "ι": "i", "κ": "k",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
}
_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)


def normalize(text: str) -> str:
    """Return a cleaned copy of ``text`` that a detector can read reliably."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_HOMOGLYPH_TABLE)
    return text


def normalize_with_span(text: str, span):
    """Normalise ``text`` and move the character span ``[start, end)`` with it.

    Each side of the span is normalised separately so the injected
    sentence keeps exact boundaries even when invisible characters
    are removed from it.
    """
    if span is None:
        return normalize(text), None
    start, end = span
    before, inside, after = normalize(text[:start]), normalize(text[start:end]), normalize(text[end:])
    return before + inside + after, (len(before), len(before) + len(inside))

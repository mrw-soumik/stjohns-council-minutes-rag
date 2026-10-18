"""Text normalisation and tokenisation for keyword (BM25) search.

The browser demo re-implements these exact rules in ``docs/assets/search.js``, and
``tests/test_text.py`` pins shared fixtures so the two stay in step:

1. lower-case,
2. Unicode NFKD, then drop combining marks U+0300 to U+036F (accents),
3. join hyphenated words ("by-law" -> "bylaw", "re-zoning" -> "rezoning"),
4. split on anything that is not ``a-z`` or ``0-9``,
5. drop stop words and single letters (single digits are kept),
6. apply the S-stemmer (Harman, 1991): "ies" -> "y", "es" -> "e", "s" -> "".
"""

from __future__ import annotations

import re
import unicodedata

# fmt: off
STOPWORDS = frozenset(
    (
        "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
        "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
        "yourselves",
    )
)
# fmt: on

_MARKS_RE = re.compile(r"[\u0300-\u036f]")
_HYPHEN_RE = re.compile(r"([a-z])-(?=[a-z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = _MARKS_RE.sub("", text)
    return _HYPHEN_RE.sub(r"\1", text)


def s_stem(word: str) -> str:
    """Harman's S-stemmer: conservative plural stripping, identical in Python and JS."""
    if len(word) > 3 and word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if len(word) > 2 and word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(normalize(text)):
        if token in STOPWORDS or (len(token) == 1 and not token.isdigit()):
            continue
        tokens.append(s_stem(token) if stem else token)
    return tokens


def contains_normalized(haystack: str, needle: str) -> bool:
    """Case-, accent- and whitespace-insensitive substring test (used by the eval checks)."""

    def squash(s: str) -> str:
        s = normalize(s).replace("\u2019", "'").replace("\u2018", "'")
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-")
        s = s.replace("\u2014", "-")
        return re.sub(r"\s+", " ", s).strip()

    return squash(needle) in squash(haystack)

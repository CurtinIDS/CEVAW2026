import functools
import math
import re
import unicodedata
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pronouncing
import pysbd

# Abbreviations that pysbd's built-in English rules don't cover but that
# occur constantly in Hansard. Extend this list against your own corpus --
# every parliament has its own conventions (state abbreviations, party
# titles, "M.P.", "Rt Hon.", etc.).
DEFAULT_ABBREVIATIONS = [
    "Mr", "Mrs", "Ms", "Dr", "Hon", "Rt Hon", "M.P", "MP", "Prof",
    "Sen", "Rep", "Qld", "N.S.W", "Vic", "St", "vs", "etc", "e.g", "i.e",
]

_PLACEHOLDER = "\u0000"  # a byte that will never appear in real text

def _protect_abbreviations(text: str, abbrevs: list[str]) -> str:
    for a in sorted(abbrevs, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(a) + r"\.", a + _PLACEHOLDER, text)
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


SEG = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    protected = _protect_abbreviations(text, DEFAULT_ABBREVIATIONS)
    return [
        _restore_abbreviations(s).strip()
        for s in SEG.segment(protected)
        if s.strip()
    ]


@functools.lru_cache
def count_sentences(text: str):
    return len(split_sentences(text))


_APOSTROPHES = {"'", "\u2019", "\u02bc"}  # ' ' modifier-letter apostrophe
_NUMERIC_JOINERS = {".", ","}  # kept only between two digits
_HYPHEN = "-"  # ASCII hyphen-minus ONLY (en and em dash count as separators)


def _is_word_char(ch: str) -> bool:
    """Not empty, not whitespace, not any flavour of Unicode punctuation
    (categories Pc/Pd/Pe/Pf/Pi/Po/Ps). Works for any script -- accented
    Latin, Cyrillic, Greek, CJK, etc. are all category L* (letter), so
    they're word characters automatically; no ASCII-only allowlist."""
    if not ch or ch.isspace():
        return False
    return not unicodedata.category(ch).startswith("P")


def split_words(text: str) -> list[str]:
    """Split on whitespace and punctuation of any kind, with three
    exceptions where the punctuation is kept as part of the word:

      1. An apostrophe enclosed by word characters on both sides
         ("isn't", "O'Brien" -- not "'quoted'" or "rock 'n' roll").
      2. An ASCII hyphen enclosed by word characters on both sides
         ("well-being", "3-year-old" -- not a leading/trailing "-").
         En dashes and em dashes are NOT included here, so they
         always act as separators.
      3. A period or comma enclosed by DIGITS on both sides, so numbers
         survive intact ("3.5", "1,234,567" -- but "Dr. Smith" still
         splits, since letters don't qualify for this exception).

    Non-ASCII letters (accents, other scripts) are word characters like
    any other letter and need no special handling.
    """
    n = len(text)
    words: list[str] = []
    current: list[str] = []

    for i, ch in enumerate(text):
        prev_ch = text[i - 1] if i > 0 else ""
        next_ch = text[i + 1] if i + 1 < n else ""

        if ch in _APOSTROPHES:
            if _is_word_char(prev_ch) and _is_word_char(next_ch):
                current.append(ch)
                continue
        elif ch == _HYPHEN:
            if _is_word_char(prev_ch) and _is_word_char(next_ch):
                current.append(ch)
                continue
        elif ch in _NUMERIC_JOINERS:
            if prev_ch.isdigit() and next_ch.isdigit():
                current.append(ch)
                continue

        if _is_word_char(ch):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []

    if current:
        words.append("".join(current))
    return words


@functools.lru_cache
def count_words(text):
    return len(split_words(text))


_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _heuristic_syllables(word: str) -> int:
    w = word.lower()
    if len(w) <= 3:
        return 1
    w = re.sub(r"e$", "", w)  # drop a silent trailing e
    groups = _VOWEL_GROUP_RE.findall(w)
    return max(len(groups), 1)


@functools.lru_cache
def count_syllables_in_word(word: str):
    phones = pronouncing.phones_for_word(word)
    if phones:
        return pronouncing.syllable_count(phones[0])
    else:
        return _heuristic_syllables(word)


@functools.lru_cache
def count_syllables_in_text(text: str):
    return sum(count_syllables_in_word(word) for word in split_words(text))


@functools.lru_cache
def count_polysyllabic_words(text, threshold=3):
    return sum(count_syllables_in_word(word) >= threshold for word in split_words(text))


def flesch_reading_ease(text):
    nsentences = count_sentences(text)
    nwords = count_words(text)
    nsyllables = count_syllables_in_text(text)
    try:
        return 206.835 - 1.015 * (nwords / nsentences) - 84.6 * (nsyllables / nwords)
    except ZeroDivisionError:
        return np.nan


def flesch_kincaid(text):
    nsentences = count_sentences(text)
    nwords = count_words(text)
    nsyllables = count_syllables_in_text(text)
    try:
        return 0.39 * (nwords / nsentences) + 11.8 * (nsyllables / nwords) - 15.59
    except ZeroDivisionError:
        return np.nan


def gunning_fog(text):
    nsentences = count_sentences(text)
    nwords = count_words(text)
    ncomplex = count_polysyllabic_words(text, 3)
    try:
        return 0.4 * (nwords / nsentences + 100 * (ncomplex / nwords))
    except ZeroDivisionError:
        return np.nan


def smog(text):
    nsentences = count_sentences(text)
    ncomplex = count_polysyllabic_words(text, 3)
    try:
        return 1.043 * math.sqrt(ncomplex * 30 / nsentences) + 3.1291
    except ZeroDivisionError:
        return np.nan


with open("dale-chall-word-list.txt") as f:
    DALE_CHALL_WORD_LIST = [word.strip() for word in f.readlines()]


def dale_chall(text):
    ncomplex = sum(word.lower() not in DALE_CHALL_WORD_LIST for word in split_words(text))
    nwords = count_words(text)
    nsentences = count_sentences(text)
    try:
        score =  0.1579 * 100 * (ncomplex / nwords) + 0.0496 * (nwords / nsentences)
        return score + 3.6365 if  100 * ncomplex / nwords > 5 else score
    except ZeroDivisionError:
        return np.nan


TESTS = {
    "readability_flesch_reading_ease": flesch_reading_ease,
    "readability_flesch_kincaid": flesch_kincaid,
    "readability_gunning_fog": gunning_fog,
    "readability_smog": smog,
}


def score_row(body):
    return {name: fn(body) for name, fn in TESTS.items()}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(score_row, df["body"]))

    for name in TESTS:
        df[name] = [r[name] for r in results]

    print(df)

    df.to_parquet(args.parquet)
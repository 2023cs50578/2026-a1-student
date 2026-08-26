"""
submission/porter.py — the Porter (1980) suffix-stripping stemmer, implemented
from the published algorithm description.

Why this file exists at all: the assignment permits NLTK's stemmer, but a
stemmer is a pure string function with no data files, and shipping our own
keeps `requirements.txt` (and therefore the grading image) minimal and keeps
the whole preprocessing pipeline inside `submission/`, where we can reason
about and tune it. See the report's code-provenance statement.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program 14(3),
1980, pp. 130-137. This follows the *original* 1980 algorithm (the same thing
NLTK exposes as `PorterStemmer.ORIGINAL_ALGORITHM`), not the later Porter2 /
Snowball revision, and not NLTK's own extra tweaks.

Terminology from the paper, used throughout below:

    consonant  a letter other than a/e/i/o/u, and other than `y` preceded by
               a vowel (so `y` in "toy" is a consonant, `y` in "happy" is a
               vowel).
    m          the "measure" of a stem: the number of vowel-consonant
               sequences in it, i.e. write the stem as [C](VC){m}[V] and read
               off m. Roughly "how many syllables of real word is left" — the
               conditions (m>0), (m>1) exist so that we only strip a suffix
               when enough word survives to still be a meaningful stem.
    *v*        the stem contains a vowel
    *d         the stem ends in a double consonant ("fall", "hopp")
    *o         the stem ends consonant-vowel-consonant, where the final
               consonant is not w, x or y ("hop", "fil")

The stemmer is the single most important preprocessing knob in this
assignment after stopwording: it conflates "vaccinate/vaccination/vaccines"
into one index term, which is exactly the recall the BM25 scorer needs on
short natural-language queries.
"""

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    """Is word[i] a consonant, in Porter's sense?"""
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        # `y` is a consonant at the start of a word or after a vowel,
        # and a vowel after a consonant.
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """m: the number of VC sequences in `stem` ([C](VC){m}[V])."""
    n = 0
    i = 0
    length = len(stem)
    # Leading consonant run (the optional [C]) does not count.
    while i < length and _is_consonant(stem, i):
        i += 1
    while i < length:
        while i < length and not _is_consonant(stem, i):  # skip the V run
            i += 1
        if i >= length:  # trailing [V] with no consonant after it
            break
        n += 1
        while i < length and _is_consonant(stem, i):  # skip the C run
            i += 1
    return n


def _contains_vowel(stem: str) -> bool:
    """*v* — does the stem contain a vowel?"""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    """*d — does the stem end in two identical consonants?"""
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem: str) -> bool:
    """*o — consonant-vowel-consonant, final consonant not w/x/y.

    This is the "short word" test: words like "hop" get an `e` restored
    ("hopping" -> "hop" -> "hope" would be wrong, but "hoping" -> "hop" ->
    "hope" is right), which is what step 1b's fixups use it for.
    """
    if len(stem) < 3:
        return False
    if not _is_consonant(stem, len(stem) - 1):
        return False
    if _is_consonant(stem, len(stem) - 2):
        return False
    if not _is_consonant(stem, len(stem) - 3):
        return False
    return stem[-1] not in "wxy"


def _replace_if(word: str, suffix: str, replacement: str, min_measure: int):
    """If `word` ends in `suffix` and the surviving stem has m > min_measure,
    return the rewritten word; otherwise return None (rule did not fire)."""
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_measure:
        return stem + replacement
    return None


# Step 2: condense double suffixes ("-ational" -> "-ate"). Order matters —
# longer, more specific suffixes must be tested before the shorter ones they
# contain (IZATION before ATION, ENTLI before ELI, ...).
_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)

# Step 3: strip the remaining derivational suffix down to its root form.
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)

# Step 4: remove the final suffix entirely, but only from a stem with m > 1
# (i.e. one long enough that something recognisable is left).
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _step1a(word: str) -> str:
    """Plurals."""
    if word.endswith("sses"):
        return word[:-2]          # caresses -> caress
    if word.endswith("ies"):
        return word[:-2]          # ponies   -> poni
    if word.endswith("ss"):
        return word               # caress   -> caress  (unchanged)
    if word.endswith("s"):
        return word[:-1]          # cats     -> cat
    return word


def _step1b(word: str) -> str:
    """Past participles and gerunds (-eed / -ed / -ing)."""
    if word.endswith("eed"):
        # (m>0) EED -> EE : "agreed" -> "agree", but "feed" stays "feed".
        if _measure(word[:-3]) > 0:
            return word[:-1]
        return word

    stripped = None
    if word.endswith("ed") and _contains_vowel(word[:-2]):
        stripped = word[:-2]
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        stripped = word[:-3]
    if stripped is None:
        return word

    # Post-strip cleanup: restore the surface form the suffix had absorbed.
    if stripped.endswith(("at", "bl", "iz")):
        return stripped + "e"                        # conflat(ed) -> conflate
    if _ends_double_consonant(stripped) and stripped[-1] not in "lsz":
        return stripped[:-1]                         # hopp(ing)   -> hop
    if _measure(stripped) == 1 and _ends_cvc(stripped):
        return stripped + "e"                        # fil(ing)    -> file
    return stripped


def _step1c(word: str) -> str:
    """(*v*) Y -> I, so "happy"/"happier" both reach "happi"."""
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step2(word: str) -> str:
    for suffix, replacement in _STEP2:
        out = _replace_if(word, suffix, replacement, 0)
        if out is not None:
            return out
    return word


def _step3(word: str) -> str:
    for suffix, replacement in _STEP3:
        out = _replace_if(word, suffix, replacement, 0)
        if out is not None:
            return out
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if not word.endswith(suffix):
            continue
        stem = word[: len(word) - len(suffix)]
        if suffix == "ion" and not (stem and stem[-1] in "st"):
            # -ION only comes off after S or T ("adoption" -> "adopt"),
            # otherwise "lion" would become "l".
            continue
        if _measure(stem) > 1:
            return stem
        # A suffix that matched but failed the measure test ends step 4;
        # falling through to a shorter suffix would over-strip.
        return word
    return word


def _step5(word: str) -> str:
    """Tidy up a trailing `e` and a doubled final `l`."""
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    if (
        len(word) > 1
        and word.endswith("l")
        and _ends_double_consonant(word)
        and _measure(word) > 1
    ):
        word = word[:-1]
    return word


def stem(word: str) -> str:
    """Return the Porter stem of an already-lowercased alphabetic token.

    Words of two letters or fewer are returned unchanged: the algorithm's
    measure conditions can never fire on them and the paper's own
    implementation short-circuits here too.
    """
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5(word)
    return word

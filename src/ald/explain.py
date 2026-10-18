"""Which word n-grams drive each prediction? Reads the logistic-regression weights.

Slurs are masked (first letter + asterisks) using the hate lexicon published with
Davidson et al. (2017), so the output can be shared.
"""
import re

import numpy as np
import pandas as pd

from .config import CLASSES, HATE_LEXICON

_lexicon = None


def _slurs():
    global _lexicon
    if _lexicon is None:
        raw = HATE_LEXICON.read_text(encoding="utf8", errors="ignore").replace("\r", "\n")
        terms = [line.split(",")[0].strip().lower() for line in raw.splitlines()[1:] if line.strip()]
        # Single-word entries only: splitting phrases would mask ordinary words inside them.
        _lexicon = {t for t in terms if " " not in t} - {"raped", "savage", "inbred"}
    return _lexicon


def mask(ngram):
    def m(word):
        bare = word.removeprefix("neg_")
        singular = bare[:-1] if len(bare) > 4 and bare.endswith("s") else bare
        if bare in _slurs() or singular in _slurs():
            return word[: len(word) - len(bare) + 1] + "*" * (len(bare) - 1)
        return word
    return " ".join(m(w) for w in ngram.split())


def top_word_ngrams(pipeline, n=15, bigrams_only=False):
    """Top positive-weight word n-grams for each class."""
    words = pipeline.named_steps["features"].transformer_list[0][1]
    vocab = np.array(words.get_feature_names_out())
    n_words = len(vocab)
    coef = pipeline.named_steps["clf"].coef_[:, :n_words]
    keep = np.array([" " in v for v in vocab]) if bigrams_only else np.ones(n_words, bool)
    rows = []
    for k, cls in enumerate(CLASSES):
        idx = np.where(keep)[0]
        order = idx[np.argsort(-coef[k, idx])][:n]
        rows += [{"class": cls, "rank": r + 1, "ngram": mask(vocab[i]), "weight": float(coef[k, i])} for r, i in enumerate(order)]
    return pd.DataFrame(rows)


def ngram_weights(pipeline, ngrams):
    """Weight of specific n-grams in each class (NaN if not in the vocabulary)."""
    words = pipeline.named_steps["features"].transformer_list[0][1]
    index = {v: i for i, v in enumerate(words.get_feature_names_out())}
    coef = pipeline.named_steps["clf"].coef_
    return {g: ({c: float(coef[k, index[g]]) for k, c in enumerate(CLASSES)} if g in index else None) for g in ngrams}

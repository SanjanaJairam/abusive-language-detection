"""Counterfactual data augmentation for identity terms.

For each training tweet that mentions an identity group, add a copy where that group is
swapped for another group of the same kind (Muslim -> Christian, gay -> straight, women -> men),
with the same label. The model then can't learn "this group's name signals abuse", only how
people talk *about* groups. This follows gender-swap augmentation (Park, Shin & Fung, 2018)
and counterfactual token fairness (Garg et al., 2019).

Swaps stay within a category so the sentence stays plausible. Very common ambiguous words
("old", "young") are left out to avoid adding noise.
"""
import random
import re

import pandas as pd

from .config import SEED

CATEGORIES = {
    "sexuality and gender identity": ["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt",
                                      "lgbtq", "homosexual", "straight", "heterosexual", "nonbinary"],
    "gender": ["male", "female"],
    "gender (plural nouns)": ["women", "men", "girls", "boys"],
    "race and ethnicity": ["african american", "african", "black", "white", "european", "hispanic", "latino",
                           "latina", "latinx", "asian", "middle eastern"],
    "nationality": ["mexican", "canadian", "american", "indian", "chinese", "japanese"],
    "religion": ["christian", "muslim", "jewish", "buddhist", "catholic", "protestant", "sikh", "hindu"],
    "religion (plural nouns)": ["christians", "muslims", "jews", "buddhists", "catholics", "sikhs", "hindus"],
    "disability": ["blind", "deaf", "disabled", "paralyzed"],
}
_TERM_TO_CATEGORY = {t: c for c, terms in CATEGORIES.items() for t in terms}
# Longest terms first so "african american" wins over "african" and "american".
_PATTERN = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _TERM_TO_CATEGORY), key=len, reverse=True)) + r")\b")


def swap_identities(text, rng):
    """Replace every identity term with a different term from the same category (None if there are none)."""
    if not _PATTERN.search(text):
        return None
    def repl(m):
        term = m.group(1)
        options = [t for t in CATEGORIES[_TERM_TO_CATEGORY[term]] if t != term]
        return rng.choice(options)
    return _PATTERN.sub(repl, text)


def augment(texts, labels, copies=1, only_labels=None):
    """Return original + counterfactual copies. `texts` must already be cleaned (lowercase).

    `only_labels` restricts swapping to tweets with those labels. Swapping only harmless tweets
    adds benign contexts for every identity term, in the spirit of Dixon et al. (2018).
    """
    rng = random.Random(SEED)
    new_texts, new_labels = [], []
    for text, label in zip(texts, labels):
        if only_labels is not None and label not in only_labels:
            continue
        for _ in range(copies):
            swapped = swap_identities(text, rng)
            if swapped is not None:
                new_texts.append(swapped)
                new_labels.append(label)
    return list(texts) + new_texts, pd.concat([pd.Series(list(labels)), pd.Series(new_labels)], ignore_index=True)

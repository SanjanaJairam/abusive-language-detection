"""The improved model as one object, shared by run_all.py and the notebooks."""
import numpy as np
import pandas as pd

from . import models
from .augment import augment
from .config import NEITHER
from .text import clean_text


def clean(texts):
    return [clean_text(t) for t in texts]


def dialect_groups(props, threshold=0.6):
    """'aae' / 'white' / 'other' from TwitterAAE proportions (see dialect.py)."""
    return np.where(props[:, 0] > threshold, "aae", np.where(props[:, 3] > threshold, "white", "other"))


class TextModel:
    """TF-IDF + balanced logistic regression with calibrated probabilities, plus the options that built it.

    olid      also train on OLID's everyday tweets
    negation  mark words after a negation ("don't hate" -> "don't neg_hate")
    swaps     None, "all" (swap identity terms in every tweet) or "benign" (only in harmless tweets)
    """

    def __init__(self, name, olid=False, negation=False, swaps=None):
        if swaps not in (None, "all", "benign"):
            raise ValueError(f"swaps must be None, 'all' or 'benign', not {swaps!r}")
        self.name, self.olid, self.negation, self.swaps = name, olid, negation, swaps

    def training_data(self, train, olid_train):
        df = pd.concat([train, olid_train]) if self.olid else train
        texts, labels = clean(df.tweet), df.label.reset_index(drop=True)
        if self.swaps == "all":
            return augment(texts, labels)
        if self.swaps == "benign":
            return augment(texts, labels, only_labels={NEITHER})
        return texts, labels

    def fit(self, train, val, olid_train, olid_val):
        X, y = self.training_data(train, olid_train)
        self.pipe = models.improved_model(negation=self.negation).fit(X, y)
        cal = pd.concat([val, olid_val]) if self.olid else val
        self.cal = models.calibrate(self.pipe, clean(cal.tweet), cal.label)
        return self

    def labels(self, texts):
        return self.pipe.predict(clean(texts))

    def proba(self, texts):
        return self.cal.predict_proba(clean(texts))

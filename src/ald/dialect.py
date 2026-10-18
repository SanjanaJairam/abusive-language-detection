"""Dialect estimates with the TwitterAAE model (Blodgett, Green & O'Connor, 2016).

Adapted (ported to Python 3) from slanglab/twitteraae `code/predict.py`,
MIT License, Copyright (c) 2019 Su Lin Blodgett and Brendan O'Connor.

For each tweet the model estimates the *proportion of words* drawn from African American-,
Hispanic-, Asian- and White-aligned language. It does not identify the author's race; the paper's authors consider only the
African American and White dimensions reliable.

Davidson, Bhattacharya & Weber (2019) call a tweet AAE-aligned when the African American
proportion exceeds 0.8. In this smaller dataset that leaves only 18 White-aligned tweets, so
the main analysis uses 0.6 (4,439 AAE- and 1,087 White-aligned tweets), reports 0.5 as a
robustness check, and also tracks error rates across the continuous AAE proportion.
"""
import re

import numpy as np

from .config import TWITTERAAE_DIR

_model = {}
_TOKEN = re.compile(r"[a-z0-9_']+|[^\sa-z0-9_']", re.I)
_CLEAN = re.compile(r"https?://\S+|@\w+|&#?\w+;")


def _load():
    if not _model:
        counts = np.loadtxt(TWITTERAAE_DIR / "model_count_table.txt")
        vocab = [line.split("\t")[-1].strip() for line in open(TWITTERAAE_DIR / "model_vocab.txt", encoding="utf8")]
        _model["probs"] = (counts + 1) / counts.sum(0)
        _model["index"] = {w: i for i, w in enumerate(vocab)}
    return _model


def _infer(token_ids, probs, alpha=1.0, passes=5):
    """Collapsed variational Bayes (CVB0) inference, as in the original script."""
    q = probs[token_ids].copy()
    q /= q.sum(1, keepdims=True)
    lik = q.copy()
    q_k = q.sum(0)
    for _ in range(1, passes):
        for i in range(len(token_ids)):
            q_k -= q[i]
            q[i] = lik[i] * (q_k + alpha)
            q[i] /= q[i].sum()
            q_k += q[i]
    return q_k / q_k.sum()


def proportions(tweet, min_tokens=1, min_share=0.2):
    """[AA, Hispanic, Asian, White] proportions, or None when too few words are in the model's vocabulary."""
    m = _load()
    tokens = [t.lower() for t in _TOKEN.findall(_CLEAN.sub(" ", tweet))]
    ids = [m["index"][t] for t in tokens if t in m["index"]]
    if len(ids) < min_tokens or (tokens and len(ids) / len(tokens) < min_share):
        return None
    return _infer(ids, m["probs"])


def group(tweet, threshold=0.6):
    """'aae', 'white' or 'other'."""
    p = proportions(tweet)
    if p is None:
        return "other"
    if p[0] > threshold:
        return "aae"
    if p[3] > threshold:
        return "white"
    return "other"

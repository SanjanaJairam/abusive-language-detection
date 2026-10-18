"""GloVe Twitter embeddings (Pennington et al., 2014), 25 dimensions, averaged per tweet as in the original project."""
import gzip

import numpy as np

from .config import GLOVE_GZ

_vectors = None


def load():
    """Read the gensim-data word2vec text file once: {word: vector}."""
    global _vectors
    if _vectors is None:
        vectors = {}
        with gzip.open(GLOVE_GZ, "rt", encoding="utf8") as f:
            next(f)  # header: vocabulary size and dimension
            for line in f:
                word, *vals = line.rstrip().split(" ")
                if len(vals) == 25:
                    vectors[word] = np.asarray(vals, dtype=np.float32)
        _vectors = vectors
    return _vectors


def embed(token_lists):
    """Average the vectors of known words; all zeros when no word is known (as in the original)."""
    vectors = load()
    out = np.zeros((len(token_lists), 25), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        known = [vectors[t] for t in tokens if t in vectors]
        if known:
            out[i] = np.mean(known, axis=0)
    return out

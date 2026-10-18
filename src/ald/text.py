"""Text preprocessing.

`original_tokens` reproduces the 2023 Kaggle notebook as closely as possible.
`clean_text` is the improved version used by the new model.
"""
import html
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# The original notebook used a Kaggle English stopword list that, like scikit-learn's,
# includes negations ("not", "no", "never"). scikit-learn's list stands in for it here.
STOPWORDS = set(ENGLISH_STOP_WORDS)
_PUNCT = str.maketrans("", "", string.punctuation)
_ORIGINAL_STRIP = re.compile(r'http\S+|www\S+|https\S+|@\S+|#\S+|"|\n|\t')


def original_tokens(tweet):
    """Original pipeline: drop URLs/mentions/hashtags, punctuation, case and stopwords."""
    tweet = _ORIGINAL_STRIP.sub("", tweet).translate(_PUNCT).lower()
    return [t for t in tweet.split() if t not in STOPWORDS]


_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^\s*rt\b[:\s]*", re.I)
_REPEAT = re.compile(r"(.)\1{3,}")


def clean_text(tweet):
    """Improved pipeline: keep negations and hashtag words, normalise URLs and mentions.

    Decodes HTML entities (the raw tweets contain &amp; and &#8220;), removes the
    retweet marker, and shortens long character runs ("soooooo" -> "sooo").
    """
    t = html.unescape(tweet).replace("\u2019", "'").replace("\u2018", "'")
    t = _RT.sub("", t)
    t = _URL.sub(" URL ", t)
    t = _MENTION.sub(" @USER ", t)
    t = t.replace("#", " ")
    t = _REPEAT.sub(r"\1\1\1", t)
    return re.sub(r"\s+", " ", t).strip().lower()


# ---------- Negation marking ----------
# Words after a negation, up to the next punctuation mark, get a "neg_" prefix, so the model
# sees "don't hate women" as different from "hate women" (Das & Chen, 2001; Pang, Lee &
# Vaithyanathan, 2002). Plain bigrams aren't enough: in the first model the unigram "hate"
# outweighed the bigram "don't hate".
NEGATIONS = {"not", "no", "never", "nobody", "nothing", "nowhere", "neither", "nor", "cannot", "without"}
# Tokens of 2+ characters, matching scikit-learn's default token pattern, so that adding negation
# marking changes nothing else about the features. Punctuation is kept only to end a negated clause.
_WORD = re.compile(r"[a-z0-9@_'][a-z0-9@_']+|[.,!?;:]")
_CLAUSE_END = {".", ",", "!", "?", ";", ":"}


def is_negation(token):
    return token in NEGATIONS or token.endswith("n't") or token in {"dont", "doesnt", "didnt", "isnt", "arent",
                                                                    "wasnt", "werent", "cant", "wont", "aint"}


def negation_tokens(text):
    """Tokens of already-cleaned text, with words in a negated clause prefixed by 'neg_'."""
    out, negated = [], False
    for tok in _WORD.findall(text):
        if tok in _CLAUSE_END:
            negated = False
            continue
        if is_negation(tok):
            out.append(tok)
            negated = True
        else:
            out.append("neg_" + tok if negated else tok)
    return out


def word_ngrams(text, n_max=2):
    """Unigrams and bigrams over negation-marked tokens (used as a TF-IDF analyzer)."""
    toks = negation_tokens(text)
    grams = list(toks)
    for n in range(2, n_max + 1):
        grams += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return grams

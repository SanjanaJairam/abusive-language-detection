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
    t = html.unescape(tweet)
    t = _RT.sub("", t)
    t = _URL.sub(" URL ", t)
    t = _MENTION.sub(" @USER ", t)
    t = t.replace("#", " ")
    t = _REPEAT.sub(r"\1\1\1", t)
    return re.sub(r"\s+", " ", t).strip().lower()

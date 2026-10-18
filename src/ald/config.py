"""Paths and constants shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42

# Davidson et al. (2017) label scheme
CLASSES = ["hate", "offensive", "neither"]
HATE, OFFENSIVE, NEITHER = 0, 1, 2

DAVIDSON_CSV = DATA / "hate-speech-and-offensive-language" / "data" / "labeled_data.csv"
HATECHECK_CSV = DATA / "hatecheck-data" / "test_suite_cases.csv"
MADLIBS_CSV = DATA / "unintended-ml-bias-analysis" / "bias_madlibs_77k.csv"
IDENTITY_TERMS = DATA / "unintended-ml-bias-analysis" / "identity_terms.txt"
TWITTERAAE_DIR = DATA / "twitteraae" / "model"
GLOVE_GZ = DATA / "glove-twitter-25.gz"
OLID_DIR = DATA / "tweeteval" / "offensive"
HATE_LEXICON = DATA / "hate-speech-and-offensive-language" / "lexicons" / "refined_ngram_dict.csv"

for _d in (RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

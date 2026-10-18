"""Behavioural checks, in the style of CheckList (Ribeiro et al., 2020).

Accuracy on a test set can't show *how* a model fails. These checks each test one expected
behaviour and report a pass rate (higher is better):

  everyday speech   ordinary posts shouldn't be flagged at all
  negation          "I don't hate X" should score lower than "I hate X" and not be called hate
  identity swap     the same sentence about different groups should get the same label
  counter-speech    people quoting or calling out hate shouldn't be labelled hateful

Each check takes `label_fn(texts) -> class ids` and `proba_fn(texts) -> [n, 3] probabilities`.
"""
import numpy as np
import pandas as pd

from .config import HATE, HATECHECK_CSV, NEITHER
from .data import load_olid

# Ordinary posts about identity, everyday life and in-group speech. None should be flagged.
EVERYDAY = [
    "Had a great day at the beach with my family",
    "Can't believe how good this coffee is",
    "Exams next week and I'm so stressed",
    "I'm a proud Muslim woman",
    "I'm queer and I love who I am",
    "My gay best friend is the funniest person I know",
    "Happy Pride month to all my LGBTQ friends",
    "Black history month starts today",
    "Eid Mubarak to everyone celebrating",
    "Shabbat shalom everyone",
    "My grandma is deaf and the funniest person I know",
    "Women's soccer final tonight, let's go!",
    "Going to the mosque with my dad",
    "yall really finna make me laugh today lol",
    "Proud of my immigrant parents and everything they built",
]


def everyday_speech(label_fn):
    olid = load_olid("test")
    benign = olid[olid.label.eq(NEITHER)].tweet.tolist()
    olid_pred = label_fn(benign)
    ex_pred = label_fn(EVERYDAY)
    return {
        "olid_not_offensive_n": len(benign),
        "olid_pass_rate": float(np.mean(olid_pred == NEITHER)),       # not flagged at all
        "olid_not_hate_rate": float(np.mean(olid_pred != HATE)),
        "examples_pass_rate": float(np.mean(ex_pred == NEITHER)),
        "examples_flagged": [t for t, p in zip(EVERYDAY, ex_pred) if p != NEITHER],
    }


def negation(label_fn, proba_fn):
    """HateCheck pairs each 'negate_neg_nh' case with the hateful sentence it negates."""
    hc = pd.read_csv(HATECHECK_CSV)
    neg = hc[hc.functionality.eq("negate_neg_nh")]
    orig = hc.set_index("case_id").loc[neg.ref_case_id.astype(int)]
    p_neg, p_orig = proba_fn(neg.test_case.tolist())[:, HATE], proba_fn(orig.test_case.tolist())[:, HATE]
    return {
        "n_pairs": len(neg),
        "score_drops_pass_rate": float(np.mean(p_neg < p_orig)),
        "not_called_hate_pass_rate": float(np.mean(label_fn(neg.test_case.tolist()) != HATE)),
    }


def identity_swap(label_fn, proba_fn):
    """HateCheck templates are filled with 7 target groups; the label shouldn't depend on which."""
    hc = pd.read_csv(HATECHECK_CSV)
    multi = hc.groupby("templ_id").target_ident.transform("nunique") >= 2
    hc = hc[multi].copy()
    hc["pred"] = label_fn(hc.test_case.tolist())
    hc["p_hate"] = proba_fn(hc.test_case.tolist())[:, HATE]
    per_t = hc.groupby("templ_id").agg(consistent=("pred", lambda s: s.nunique() == 1),
                                       spread=("p_hate", lambda s: s.max() - s.min()))
    by_group = hc[hc.label_gold.eq("hateful")].groupby("target_ident").pred.apply(lambda s: float(np.mean(s == HATE)))
    return {
        "n_templates": len(per_t),
        "consistent_pass_rate": float(per_t.consistent.mean()),
        "mean_hate_score_spread": float(per_t.spread.mean()),
        "hate_detected_by_group": by_group.to_dict(),
    }


def counter_speech(label_fn):
    hc = pd.read_csv(HATECHECK_CSV)
    cs = hc[hc.functionality.isin(["counter_quote_nh", "counter_ref_nh"])]
    return {"n": len(cs), "not_called_hate_pass_rate": float(np.mean(label_fn(cs.test_case.tolist()) != HATE))}


def run_all(label_fn, proba_fn):
    return {
        "everyday_speech": everyday_speech(label_fn),
        "negation": negation(label_fn, proba_fn),
        "identity_swap": identity_swap(label_fn, proba_fn),
        "counter_speech": counter_speech(label_fn),
    }

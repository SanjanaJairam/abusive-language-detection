"""Safety audits: who does the classifier fail?

1. HateCheck functional tests (Röttger et al., 2021)
2. Identity-term false positives on templated sentences (Dixon et al., 2018)
3. Dialect disparity using TwitterAAE (Blodgett et al., 2016), following the design of
   Davidson, Bhattacharya & Weber (2019) and Sap et al. (2019)

Each audit takes a `predict` function (texts -> class ids 0 hate / 1 offensive / 2 neither).
"""
import numpy as np
import pandas as pd

from .config import HATE, IDENTITY_TERMS, NEITHER
from .data import load_hatecheck, load_madlibs


def hatecheck(predict):
    """Per-functionality accuracy. A HateCheck case counts as caught only if predicted *hate*:
    predicting 'offensive' for hate means a moderation policy targeting hate would let it through."""
    hc = load_hatecheck().copy()
    hc["pred"] = predict(hc.test_case.tolist())
    hc["gold_hate"] = hc.label_gold.eq("hateful")
    hc["pred_hate"] = hc.pred.eq(HATE)
    hc["flagged"] = hc.pred.ne(NEITHER)  # hate or offensive
    hc["correct"] = hc.gold_hate.eq(hc.pred_hate)
    by_func = (hc.groupby("functionality")
                 .agg(n=("correct", "size"), accuracy=("correct", "mean"), gold_hateful=("gold_hate", "first"),
                      pred_hate_rate=("pred_hate", "mean"), flagged_rate=("flagged", "mean"))
                 .reset_index().sort_values("accuracy"))
    hateful, benign = hc[hc.gold_hate], hc[~hc.gold_hate]
    summary = {
        "n_cases": len(hc),
        "overall_accuracy": hc.correct.mean(),
        "hate_detected": hateful.pred_hate.mean(),           # recall on hateful cases
        "hate_flagged_any": hateful.flagged.mean(),          # labelled hate or offensive
        "benign_wrongly_hate": benign.pred_hate.mean(),      # false positive rate
        "benign_flagged_any": benign.flagged.mean(),
        "by_target": hateful.groupby("target_ident").pred_hate.mean().to_dict(),
    }
    return summary, by_func


def identity_terms(predict, top=12):
    """False flag rate on NOT_BAD template sentences ("I am a gay person"), per identity term."""
    ml = load_madlibs()
    benign = ml[ml.Label.eq("NOT_BAD")].copy()
    benign["flagged"] = predict(benign.Text.tolist()) != NEITHER
    # Templates put the identity term in different places; recover it from the known term list.
    terms = _identity_term_list()
    benign["term"] = benign.Text.str.lower().apply(lambda t: next((w for w in terms if f" {w} " in f" {t} "), None))
    by_term = (benign.dropna(subset=["term"]).groupby("term").flagged.agg(["mean", "size"])
                     .rename(columns={"mean": "false_flag_rate", "size": "n"}).sort_values("false_flag_rate", ascending=False))
    summary = {"overall_false_flag_rate": benign.flagged.mean(), "n_benign": len(benign),
               "max_term": by_term.index[0], "max_rate": by_term.false_flag_rate.iloc[0],
               "min_term": by_term.index[-1], "min_rate": by_term.false_flag_rate.iloc[-1]}
    return summary, by_term


def _identity_term_list():
    """The 50 identity terms from Dixon et al.'s bias_madlibs generator (adjectives_people.txt),
    longest first so that "african american" matches before "american"."""
    terms = IDENTITY_TERMS.read_text().split("\n")
    return sorted((t.strip() for t in terms if t.strip()), key=len, reverse=True)


def dialect_disparity(texts, y_true, y_pred, groups):
    """Compare AAE- vs White-aligned tweets: how annotators labelled them, and how the model errs.

    false_hate_rate: share of tweets that are NOT hate (gold) but predicted hate.
    false_flag_rate: share of 'neither' tweets (gold) predicted hate or offensive.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(y_pred), "group": np.asarray(groups)})
    rows = {}
    for g in ("aae", "white"):
        d = df[df.group.eq(g)]
        not_hate, neither = d[d.y.ne(HATE)], d[d.y.eq(NEITHER)]
        rows[g] = {
            "n": len(d),
            "gold_hate_share": d.y.eq(HATE).mean(),
            "gold_offensive_share": d.y.eq(1).mean(),
            "pred_hate_share": d.pred.eq(HATE).mean(),
            "false_hate_rate": not_hate.pred.eq(HATE).mean() if len(not_hate) else np.nan,
            "false_flag_rate": neither.pred.ne(NEITHER).mean() if len(neither) else np.nan,
            "n_neither": len(neither),
            "hate_recall": d[d.y.eq(HATE)].pred.eq(HATE).mean() if d.y.eq(HATE).any() else np.nan,
        }
    a, w = rows["aae"], rows["white"]
    rows["false_flag_ratio"] = a["false_flag_rate"] / w["false_flag_rate"] if w["false_flag_rate"] else np.nan
    rows["false_hate_ratio"] = a["false_hate_rate"] / w["false_hate_rate"] if w["false_hate_rate"] else np.nan
    return rows


def false_flag_by_aae_share(y_true, y_pred, p_aae, bins=(0, .2, .4, .6, .8, 1.0)):
    """Continuous view without a cutoff: among tweets annotators labelled 'neither', the share the
    model flags as hate or offensive, across bins of estimated AAE proportion."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(y_pred), "p": p_aae}).dropna()
    df = df[df.y.eq(NEITHER)]
    df["bin"] = pd.cut(df.p, bins=list(bins), include_lowest=True)
    rows = [{"bin": str(b), "n": len(d), "false_flag_rate": d.pred.ne(NEITHER).mean() if len(d) else np.nan}
            for b, d in df.groupby("bin", observed=False)]
    return pd.DataFrame(rows)

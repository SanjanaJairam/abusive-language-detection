"""Moderation policies built on the same classifier, compared on what they do to people.

The Facebook Files (Wall Street Journal, 2021) reported that Facebook relied increasingly on
automated removal while cutting human review time, and that its systems removed posts
accounting for only 3-5% of hate-speech views. The question here is not "which model is
best" but "given an imperfect model, which *policy* catches the most hate while wrongly
silencing the fewest people, and does the burden fall equally?"

Assumption (stated in the README): human reviewers reach the gold label. This is
optimistic; annotator bias is part of the dialect problem (Sap et al., 2019).
"""
import numpy as np

from .config import HATE, NEITHER, OFFENSIVE

ALLOW, NUDGE, REVIEW, REMOVE = "allow", "nudge", "review", "remove"


def _lowest_threshold(scores, is_positive, target_precision, min_support=20):
    """Lowest score at which flagging everything above it is at least `target_precision` precise."""
    order = np.argsort(-scores)
    precision = is_positive[order].cumsum() / np.arange(1, len(order) + 1)
    ok = np.where((precision >= target_precision) & (np.arange(1, len(order) + 1) >= min_support))[0]
    best = float(precision[min_support - 1:].max())
    return (float(scores[order][ok.max()]) if len(ok) else None), best


def choose_thresholds(p_val, y_val, target_precision=0.80, review_budget=0.05, nudge_precision=0.95):
    """Pick thresholds on validation data only.

    t_auto:   lowest hate score at which automatic removal is at least `target_precision` precise
              (None if the model never gets there: then nothing is removed without a person).
    t_review: lower bound of the human-review band, sized so about `review_budget` of posts are reviewed.
    t_nudge:  offensive score above which at least `nudge_precision` of nudged posts really are
              offensive or hateful, so ordinary posts are rarely nudged.
    """
    p_hate, p_off = p_val[:, HATE], p_val[:, OFFENSIVE]
    t_auto, best_precision = _lowest_threshold(p_hate, y_val == HATE, target_precision)

    below = p_hate[p_hate < t_auto] if t_auto is not None else p_hate
    n_review = int(review_budget * len(p_hate))
    t_review = float(np.sort(below)[::-1][min(n_review, len(below) - 1)])

    t_nudge, _ = _lowest_threshold(p_off, y_val != NEITHER, nudge_precision)
    return {"t_auto": t_auto, "t_review": t_review, "t_nudge": t_nudge, "target_precision": target_precision,
            "best_hate_precision": best_precision, "review_budget": review_budget, "nudge_precision": nudge_precision}


def remove_predicted_hate(pred):
    """Facebook-style: remove whatever the model labels hate, with no human in the loop."""
    pred = np.asarray(pred)
    return np.where(pred == HATE, REMOVE, ALLOW).astype(object)


def decide(probs, policy, th=None):
    """One action per post. `probs` columns: hate, offensive, neither (calibrated)."""
    pred = probs.argmax(1)
    p_hate = probs[:, HATE]
    actions = np.full(len(probs), ALLOW, dtype=object)
    if policy == "remove_all_flagged":          # over-cautious: remove hate and offensive alike
        actions[pred != NEITHER] = REMOVE
    elif policy == "triage":
        # Remove automatically only where the model is reliably right; send the next-riskiest
        # posts to a person; ask authors of offensive-but-not-hateful posts to reconsider.
        auto = (p_hate >= th["t_auto"]) if th["t_auto"] is not None else np.zeros(len(probs), bool)
        review = (p_hate >= th["t_review"]) & ~auto
        confident_offensive = probs[:, OFFENSIVE] >= th["t_nudge"] if th["t_nudge"] is not None else np.zeros(len(probs), bool)
        actions[confident_offensive & ~auto & ~review] = NUDGE
        actions[review] = REVIEW
        actions[auto] = REMOVE
    else:
        raise ValueError(policy)
    return actions


def review_budget_sweep(p_val, y_val, p_test, y_test, groups, budgets=(.01, .02, .05, .10, .15, .20)):
    """How much hate a triage policy catches as the human-review budget grows."""
    rows = []
    for b in budgets:
        th = choose_thresholds(p_val, y_val, review_budget=b)
        o = outcomes(decide(p_test, "triage", th), y_test, groups)
        rows.append({"review_budget": b, **o})
    return rows


def outcomes(actions, y_true, groups):
    """What happened to people. Reviewed posts are removed only if they are actually hate."""
    y_true, groups = np.asarray(y_true), np.asarray(groups)
    is_hate = y_true == HATE
    removed = (actions == REMOVE) | ((actions == REVIEW) & is_hate)
    wrongly_removed = removed & ~is_hate

    def wrongful_rate(mask):
        base = mask & ~is_hate
        return float(wrongly_removed[base].mean()) if base.any() else float("nan")

    return {
        "hate_caught": float(removed[is_hate].mean()),
        "wrongful_removals": int(wrongly_removed.sum()),
        "wrongful_removal_rate": wrongful_rate(np.ones_like(is_hate)),
        "wrongful_rate_aae": wrongful_rate(groups == "aae"),
        "wrongful_rate_white": wrongful_rate(groups == "white"),
        "review_load": float((actions == REVIEW).mean()),
        "nudged": float((actions == NUDGE).mean()),
        "auto_removed": float((actions == REMOVE).mean()),
    }

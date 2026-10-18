"""Figures for the README. Muted palette, one message per chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, FIGURES

INK, MUTED, GRID = "#2A2150", "#6B6492", "#E4DEF6"
BLUE, ORANGE, GREEN, RED = "#4C6EF5", "#E8833A", "#2F9E77", "#D6455D"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


# Readable names for HateCheck functionalities (Röttger et al., 2021, Table 1)
HATECHECK_NAMES = {
    "derog_neg_emote_h": "Strong negative emotion", "derog_neg_attrib_h": "Negative attribute",
    "derog_dehum_h": "Dehumanisation", "derog_impl_h": "Implicit derogation",
    "threat_dir_h": "Direct threat", "threat_norm_h": "Threat as normative statement",
    "slur_h": "Slur", "slur_homonym_nh": "Non-hateful homonym of a slur", "slur_reclaimed_nh": "Reclaimed slur",
    "profanity_h": "Hate with profanity", "profanity_nh": "Profanity, not hateful",
    "ref_subs_clause_h": "Pronoun reference (clause)", "ref_subs_sent_h": "Pronoun reference (sentence)",
    "negate_pos_h": "Negated positive statement", "negate_neg_nh": "Negated hateful statement",
    "phrase_question_h": "Hate phrased as a question", "phrase_opinion_h": "Hate phrased as an opinion",
    "ident_neutral_nh": "Neutral mention of identity", "ident_pos_nh": "Positive statement about identity",
    "counter_quote_nh": "Counter-speech quoting hate", "counter_ref_nh": "Counter-speech referencing hate",
    "target_obj_nh": "Abuse at an object", "target_indiv_nh": "Abuse at an individual",
    "target_group_nh": "Abuse at a non-protected group",
    "spell_char_swap_h": "Spelling: swapped characters", "spell_char_del_h": "Spelling: missing characters",
    "spell_space_del_h": "Spelling: missing spaces", "spell_space_add_h": "Spelling: added spaces",
    "spell_leet_h": "Spelling: leet speak",
}


def _save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)


def confusion(cm_a, cm_b, title_a, title_b):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, cm, title in zip(axes, (cm_a, cm_b), (title_a, title_b)):
        cm = np.asarray(cm, float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", color="white" if norm[i, j] > .55 else INK)
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Where each true class ends up (row-normalised, test set)", y=1.02)
    _save(fig, "confusion_matrices.png")


def per_class_recall(rows):
    """rows: [(model name, {class: recall})]"""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(3); w = .8 / len(rows)
    for k, (name, rec) in enumerate(rows):
        vals = [rec[c] for c in CLASSES]
        bars = ax.bar(x + (k - (len(rows) - 1) / 2) * w, vals, w, label=name, color=[MUTED, BLUE, GREEN][k % 3])
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=9, padding=2)
    ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1.12); ax.set_ylabel("recall")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Share of each class the model gets right")
    _save(fig, "per_class_recall.png")


def hatecheck_functionalities(by_func_a, by_func_b, name_a, name_b, n=12):
    """Worst functionalities for the improved model, with the original alongside."""
    m = by_func_b.merge(by_func_a[["functionality", "accuracy"]], on="functionality", suffixes=("", "_orig"))
    m = m.sort_values("accuracy").head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, .42 * n + 1))
    y = np.arange(len(m))
    ax.barh(y + .2, m.accuracy_orig, .4, color=MUTED, label=name_a)
    ax.barh(y - .2, m.accuracy, .4, color=BLUE, label=name_b)
    labels = [f"{HATECHECK_NAMES.get(f, f)} ({'hateful' if g else 'not hateful'})" for f, g in zip(m.functionality, m.gold_hateful)]
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("accuracy on HateCheck functional tests")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"The {n} HateCheck tests the improved model fails most")
    _save(fig, "hatecheck_worst.png")


def identity_false_flags(by_term_a, by_term_b, name_a, name_b, n=15):
    m = by_term_b.join(by_term_a[["false_flag_rate"]], rsuffix="_orig").head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, .36 * n + 1))
    y = np.arange(len(m))
    ax.barh(y + .2, m.false_flag_rate_orig, .4, color=MUTED, label=name_a)
    ax.barh(y - .2, m.false_flag_rate, .4, color=BLUE, label=name_b)
    ax.set_yticks(y, m.index, fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("benign sentences wrongly flagged (e.g. \"I am a ___ person\")")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Identity terms that most often get harmless sentences flagged")
    _save(fig, "identity_false_flags.png")


def dialect(rows_by_model):
    """rows_by_model: [(name, disparity dict)] -> harmless tweets flagged, AAE vs White-aligned."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = [n for n, _ in rows_by_model]
    x = np.arange(len(names))
    aae = [r["aae"]["false_flag_rate"] for _, r in rows_by_model]
    white = [r["white"]["false_flag_rate"] for _, r in rows_by_model]
    b1 = ax.bar(x - .2, aae, .4, color=ORANGE, label="AAE-aligned tweets")
    b2 = ax.bar(x + .2, white, .4, color=BLUE, label="White-aligned tweets")
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in bars.datavalues], fontsize=9, padding=2)
    ax.set_xticks(x, names, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("harmless tweets flagged\nas hate or offensive")
    ax.set_ylim(0, max(aae + white) * 1.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Who gets wrongly flagged (5-fold cross-validated)")
    _save(fig, "dialect_false_flags.png")


def review_sweep(rows, auto):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    b = [r["review_budget"] for r in rows]
    caught = [r["hate_caught"] for r in rows]
    ax.plot(b, caught, marker="o", color=GREEN, label="Triage: hate caught (no wrongful removals)")
    ax.axhline(auto["hate_caught"], color=RED, ls="--",
               label=f"Auto-remove: {auto['hate_caught']:.0%} caught, {auto['wrongful_removals']} harmless posts removed")
    for x, y in zip(b, caught):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_xticks(b, [f"{v:.0%}" for v in b])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of posts a human reviews")
    ax.set_ylabel("hate caught")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Hate caught as the human-review budget grows (test set)")
    _save(fig, "review_budget_sweep.png")


def policies(results):
    """results: [(label, outcomes dict)]"""
    labels = [l for l, _ in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    specs = [("hate_caught", "Hate caught", GREEN, "higher is better"),
             ("wrongful_removal_rate", "Harmless posts wrongly removed", RED, "lower is better"),
             ("review_load", "Posts sent to a human", BLUE, "cost")]
    y = np.arange(len(labels))[::-1]
    for ax, (key, title, color, note) in zip(axes, specs):
        vals = [r[key] for _, r in results]
        bars = ax.barh(y, vals, color=color)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], fontsize=10, padding=3)
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlim(0, max(vals) * 1.35 if max(vals) else 1)
    axes[0].set_yticks(y, labels, fontsize=10.5)
    fig.suptitle("Same data, different moderation policies (test set)", y=1.03, fontsize=13)
    _save(fig, "policy_comparison.png")


def wrongful_by_dialect(results):
    labels = [l.replace(", ", ",\n").replace(": ", ":\n") for l, _ in results]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(labels))
    a = [r["wrongful_rate_aae"] for _, r in results]
    w = [r["wrongful_rate_white"] for _, r in results]
    b1 = ax.bar(x - .2, a, .4, color=ORANGE, label="AAE-aligned")
    b2 = ax.bar(x + .2, w, .4, color=BLUE, label="White-aligned")
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in bars.datavalues], fontsize=8, padding=2)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("harmless posts\nwrongly removed")
    ax.set_ylim(0, max(a + w) * 1.3 if max(a + w) else 1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Who bears the cost of each policy")
    _save(fig, "policy_by_dialect.png")

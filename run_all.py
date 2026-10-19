"""Reproduce every number and figure in the README.

    bash get_data.sh      # once: downloads the public datasets into data/
    python run_all.py     # about 10-15 minutes on a laptop

Results go to results/ (JSON and CSV) and figures/ (PNG).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ald import audits, checks, data, dialect, explain, glove, models, plots, policy  # noqa: E402
from ald.pipeline import TextModel, clean, dialect_groups  # noqa: E402
from ald.config import CLASSES, NEITHER, RESULTS, SEED  # noqa: E402
from ald.metrics import evaluate  # noqa: E402
from ald.text import original_tokens  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main():
    out = {}
    df = data.load_davidson()
    train, val, test = data.split(df)
    test_index = test["index"].to_numpy()
    olid = {s: data.load_olid(s) for s in ("train", "val", "test")}
    counts = lambda d: d.label.value_counts().sort_index().rename(dict(enumerate(CLASSES))).to_dict()  # noqa: E731
    out["data"] = {"davidson": {"n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
                                "class_counts": counts(df)},
                   "olid": {s: counts(d) for s, d in olid.items()}}
    log(f"Davidson et al. 2017: {len(df)} tweets (train {len(train)}, val {len(val)}, test {len(test)}); "
        f"OLID: {sum(len(d) for d in olid.values())} tweets")

    # ---------- 1. Baselines: the original model and the survey's bigram SVM ----------
    log("Original model: GloVe Twitter-25 averages + classifiers")
    def glove_X(texts):
        return glove.embed([original_tokens(t) for t in texts])
    Xtr, Xte = glove_X(train.tweet), glove_X(test.tweet)
    out["original"] = {}
    orig_name = "SVM (rbf, C=10, tuned)"
    for name, m in models.original_models().items():
        m.fit(Xtr, train.label)
        out["original"][name] = evaluate(test.label, m.predict(Xte))
        if name == orig_name:
            orig = m
    orig_labels = lambda texts: orig.predict(glove_X(texts))  # noqa: E731
    orig_scores = lambda texts: orig.decision_function(glove_X(texts))  # noqa: E731
    out["balanced_glove_svm"] = evaluate(test.label, models.balanced_glove_svm().fit(Xtr, train.label).predict(Xte))

    log("Survey baseline: bigram TF-IDF + linear SVM")
    bigram = models.bigram_svm().fit(clean(train.tweet), train.label)
    bigram_labels = lambda texts: bigram.predict(clean(texts))  # noqa: E731
    bigram_scores = lambda texts: bigram.decision_function(clean(texts))  # noqa: E731
    out["bigram_svm"] = {"davidson_test": evaluate(test.label, bigram_labels(test.tweet)),
                         "olid_test": evaluate(olid["test"].label, bigram_labels(olid["test"].tweet)),
                         "checks": checks.run_all(bigram_labels, bigram_scores)}
    out["original_checks"] = checks.run_all(orig_labels, orig_scores)
    out["original_olid_test"] = evaluate(olid["test"].label, orig_labels(olid["test"].tweet))

    # ---------- 2. Improved models, one change at a time ----------
    variants = [
        TextModel("TF-IDF + balanced LR"),
        TextModel("+ everyday tweets (OLID)", olid=True),
        TextModel("+ negation marking", olid=True, negation=True),
        TextModel("+ identity swaps (final)", olid=True, negation=True, swaps="all"),
    ]
    out["variants"] = {}
    for v in variants:
        log(f"Training {v.name}")
        v.fit(train, val, olid["train"], olid["val"])
        out["variants"][v.name] = {
            "davidson_test": evaluate(test.label, v.labels(test.tweet)),
            "olid_test": evaluate(olid["test"].label, v.labels(olid["test"].tweet)),
            "checks": checks.run_all(v.labels, v.proba),
        }
    first, final = variants[0], variants[-1]

    plots.confusion(out["original"][orig_name]["confusion_matrix"], out["variants"][final.name]["davidson_test"]["confusion_matrix"],
                    "Original: GloVe + SVM", "Final: TF-IDF + balanced LR")
    plots.per_class_recall([
        ("Original SVM", {c: out["original"][orig_name]["per_class"][c]["recall"] for c in CLASSES}),
        ("Bigram TF-IDF + SVM", {c: out["bigram_svm"]["davidson_test"]["per_class"][c]["recall"] for c in CLASSES}),
        ("Final model", {c: out["variants"][final.name]["davidson_test"]["per_class"][c]["recall"] for c in CLASSES}),
    ])
    plots.checks([("Original SVM", out["original_checks"]), ("Bigram TF-IDF + SVM", out["bigram_svm"]["checks"])]
                 + [(v.name, out["variants"][v.name]["checks"]) for v in variants])

    # ---------- 3. What the model relies on: bigrams and negation ----------
    log("Top word n-grams")
    explain.top_word_ngrams(final.pipe, n=15, bigrams_only=True).to_csv(RESULTS / "top_bigrams_final.csv", index=False)
    out["ngram_weights"] = {
        first.name: explain.ngram_weights(first.pipe, ["hate", "don't hate", "dont hate"]),
        final.name: explain.ngram_weights(final.pipe, ["hate", "neg_hate", "don't neg_hate", "dont neg_hate"]),
    }

    # ---------- 4. Safety audits on the original and final models ----------
    log("Audit: HateCheck and identity terms")
    hc_o, hc_o_func = audits.hatecheck(orig_labels)
    hc_f, hc_f_func = audits.hatecheck(final.labels)
    out["hatecheck"] = {"original": hc_o, "bigram_svm": audits.hatecheck(bigram_labels)[0],
                        "first": audits.hatecheck(first.labels)[0], "final": hc_f}
    hc_f_func.merge(hc_o_func[["functionality", "accuracy"]], on="functionality", suffixes=("_final", "_original")) \
        .to_csv(RESULTS / "hatecheck_by_functionality.csv", index=False)
    plots.hatecheck_functionalities(hc_o_func, hc_f_func, "Original SVM", "Final model")

    id_o, id_o_terms = audits.identity_terms(orig_labels)
    id_f, id_f_terms = audits.identity_terms(final.labels)
    out["identity_terms"] = {"original": id_o, "bigram_svm": audits.identity_terms(bigram_labels)[0],
                             "first": audits.identity_terms(first.labels)[0], "final": id_f}
    id_f_terms.join(id_o_terms[["false_flag_rate"]], rsuffix="_original").to_csv(RESULTS / "identity_false_flags.csv")
    plots.identity_false_flags(id_o_terms, id_f_terms, "Original SVM", "Final model")

    log("Audit: dialect disparity (TwitterAAE), 5-fold cross-validated predictions on all Davidson tweets")
    # Each tweet is predicted by a model that never saw it. For models trained with OLID, the
    # OLID training tweets are added to every fold, as in the main experiment.
    props = np.array([p if (p := dialect.proportions(t)) is not None else [np.nan] * 4 for t in df.tweet])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_preds = {"Original SVM": cross_val_predict(models.original_models()[orig_name], glove_X(df.tweet), df.label, cv=folds)}
    for label, v in (("TF-IDF + balanced LR", first), ("Final model", final)):
        pred = np.empty(len(df), dtype=int)
        for tr, te in folds.split(df, df.label):
            X, y = v.training_data(df.iloc[tr], olid["train"])
            pred[te] = models.improved_model(negation=v.negation).fit(X, y).predict(clean(df.tweet.iloc[te]))
        cv_preds[label] = pred
    out["dialect"] = {}
    for th in (0.6, 0.5):
        g = dialect_groups(props, th)
        out["dialect"][f"threshold_{th}"] = {n: audits.dialect_disparity(df.tweet, df.label, p, g) for n, p in cv_preds.items()}
    out["dialect"]["by_aae_share"] = {n: audits.false_flag_by_aae_share(df.label, p, props[:, 0]).to_dict("records")
                                      for n, p in cv_preds.items()}
    plots.dialect([(n, out["dialect"]["threshold_0.6"][n]) for n in cv_preds])

    # ---------- 5. Moderation policies (Davidson test set, final model) ----------
    log("Comparing moderation policies")
    groups = dialect_groups(props, 0.6)[test_index]
    p_val, p_test = final.proba(val.tweet), final.proba(test.tweet)
    y_val, y_test = val.label.to_numpy(), test.label.to_numpy()
    # Ordinary posts (OLID validation tweets marked not offensive) set a ceiling on how often people get nudged.
    p_everyday = final.proba(olid["val"][olid["val"].label.eq(NEITHER)].tweet)
    th5 = policy.choose_thresholds(p_val, y_val, review_budget=0.05, p_everyday=p_everyday)
    th10 = policy.choose_thresholds(p_val, y_val, review_budget=0.10, p_everyday=p_everyday)
    out["policy_thresholds"] = {"review_5": th5, "review_10": th10}
    pol = {
        "Original model, auto-remove": policy.outcomes(policy.remove_predicted_hate(orig.predict(Xte)), y_test, groups),
        "Final model, auto-remove": policy.outcomes(policy.remove_predicted_hate(final.labels(test.tweet)), y_test, groups),
        "Remove anything flagged": policy.outcomes(policy.decide(p_test, "remove_all_flagged"), y_test, groups),
        "Triage: 5% to human review": policy.outcomes(policy.decide(p_test, "triage", th5), y_test, groups),
        "Triage: 10% to human review": policy.outcomes(policy.decide(p_test, "triage", th10), y_test, groups),
    }
    out["policies"] = pol
    # What triage does to an everyday feed: OLID tweets annotators marked not offensive.
    benign = olid["test"][olid["test"].label.eq(NEITHER)].tweet
    acts = policy.decide(final.proba(benign), "triage", th10)
    out["policy_everyday_feed"] = {a: float(np.mean(acts == a)) for a in ("allow", "nudge", "review", "remove")}
    out["review_budget_sweep"] = policy.review_budget_sweep(p_val, y_val, p_test, y_test, groups, p_everyday=p_everyday)
    plots.policies(list(pol.items()))
    plots.wrongful_by_dialect(list(pol.items()))
    plots.review_sweep(out["review_budget_sweep"], pol["Final model, auto-remove"])

    (RESULTS / "results.json").write_text(json.dumps(to_json(out), indent=2))
    log("Done. Results in results/results.json, figures in figures/")


if __name__ == "__main__":
    main()

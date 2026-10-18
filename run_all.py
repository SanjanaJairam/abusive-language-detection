"""Reproduce every number and figure in the README.

    bash get_data.sh      # once: downloads the public datasets into data/
    python run_all.py     # about 10 minutes on a laptop

Results go to results/ (JSON and CSV) and figures/ (PNG).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ald import audits, data, dialect, glove, models, plots, policy  # noqa: E402
from ald.config import CLASSES, RESULTS, SEED  # noqa: E402
from ald.metrics import evaluate  # noqa: E402
from ald.text import clean_text, original_tokens  # noqa: E402


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


def dialect_groups(props, threshold):
    """'aae' / 'white' / 'other' from TwitterAAE proportions (see src/ald/dialect.py)."""
    return np.where(props[:, 0] > threshold, "aae", np.where(props[:, 3] > threshold, "white", "other"))


def main():
    out = {}
    df = data.load_davidson()
    train, val, test = data.split(df)
    test_index = test["index"].to_numpy()
    out["data"] = {"n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "class_counts": df.label.value_counts().sort_index().rename(dict(enumerate(CLASSES))).to_dict()}
    log(f"Davidson et al. 2017: {len(df)} tweets; train {len(train)}, val {len(val)}, test {len(test)}")

    # ---------- 1. Reproduce the original model ----------
    log("Embedding tweets with GloVe Twitter-25 (original preprocessing)")
    def glove_X(texts):
        return glove.embed([original_tokens(t) for t in texts])
    Xtr, Xte = glove_X(train.tweet), glove_X(test.tweet)

    out["original"] = {}
    fitted = {}
    for name, m in models.original_models().items():
        log(f"  training {name}")
        m.fit(Xtr, train.label)
        fitted[name] = m
        out["original"][name] = evaluate(test.label, m.predict(Xte))
    orig_name = "SVM (rbf, C=10, tuned)"
    orig = fitted[orig_name]
    orig_pred = lambda texts: orig.predict(glove_X(texts))  # noqa: E731

    log("  training balanced GloVe SVM")
    bal = models.balanced_glove_svm().fit(Xtr, train.label)
    out["balanced_glove_svm"] = evaluate(test.label, bal.predict(Xte))

    # ---------- 2. Improved, calibrated model ----------
    log("Training improved model (TF-IDF word + char n-grams, balanced logistic regression)")
    clean = lambda s: [clean_text(t) for t in s]  # noqa: E731
    imp = models.improved_model().fit(clean(train.tweet), train.label)
    cal = models.calibrate(imp, clean(val.tweet), val.label)
    imp_pred = lambda texts: imp.predict(clean(texts))  # noqa: E731
    out["improved"] = evaluate(test.label, imp_pred(test.tweet))

    plots.confusion(out["original"][orig_name]["confusion_matrix"], out["improved"]["confusion_matrix"],
                    "Original: GloVe + SVM", "Improved: TF-IDF + balanced LR")
    plots.per_class_recall([
        ("Original SVM", {c: out["original"][orig_name]["per_class"][c]["recall"] for c in CLASSES}),
        ("Original SVM + class weights", {c: out["balanced_glove_svm"]["per_class"][c]["recall"] for c in CLASSES}),
        ("Improved model", {c: out["improved"]["per_class"][c]["recall"] for c in CLASSES}),
    ])

    # ---------- 3. Safety audits ----------
    log("Audit: HateCheck functional tests")
    hc_o, hc_o_func = audits.hatecheck(orig_pred)
    hc_i, hc_i_func = audits.hatecheck(imp_pred)
    out["hatecheck"] = {"original": hc_o, "improved": hc_i}
    hc_i_func.merge(hc_o_func[["functionality", "accuracy"]], on="functionality", suffixes=("_improved", "_original")) \
        .to_csv(RESULTS / "hatecheck_by_functionality.csv", index=False)
    plots.hatecheck_functionalities(hc_o_func, hc_i_func, "Original SVM", "Improved model")

    log("Audit: identity-term false flags")
    id_o, id_o_terms = audits.identity_terms(orig_pred)
    id_i, id_i_terms = audits.identity_terms(imp_pred)
    out["identity_terms"] = {"original": id_o, "improved": id_i}
    id_i_terms.join(id_o_terms[["false_flag_rate"]], rsuffix="_original").to_csv(RESULTS / "identity_false_flags.csv")
    plots.identity_false_flags(id_o_terms, id_i_terms, "Original SVM", "Improved model")

    log("Audit: dialect disparity (TwitterAAE), 5-fold cross-validated predictions on all tweets")
    # The test split alone has only 13 AAE-aligned 'neither' tweets, so this audit predicts every
    # tweet with a model that never saw it in training (5-fold cross-validation).
    props = np.array([p if (p := dialect.proportions(t)) is not None else [np.nan] * 4 for t in df.tweet])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_all = glove_X(df.tweet)
    cv_preds = {
        "Original SVM": cross_val_predict(models.original_models()[orig_name], X_all, df.label, cv=folds),
        "Improved model": cross_val_predict(models.improved_model(), clean(df.tweet), df.label, cv=folds),
    }
    out["dialect"] = {}
    for th in (0.6, 0.5):
        g = dialect_groups(props, th)
        out["dialect"][f"threshold_{th}"] = {name: audits.dialect_disparity(df.tweet, df.label, p, g)
                                            for name, p in cv_preds.items()}
    out["dialect"]["by_aae_share"] = {name: audits.false_flag_by_aae_share(df.label, p, props[:, 0]).to_dict("records")
                                      for name, p in cv_preds.items()}
    plots.dialect([(n, out["dialect"]["threshold_0.6"][n]) for n in cv_preds])

    # ---------- 4. Moderation policies (test set) ----------
    log("Comparing moderation policies")
    groups = dialect_groups(props, 0.6)[test_index]
    p_val = cal.predict_proba(clean(val.tweet))
    p_test = cal.predict_proba(clean(test.tweet))
    y_test = test.label.to_numpy()
    th = policy.choose_thresholds(p_val[:, 0], val.label.to_numpy())
    out["policy_thresholds"] = th
    pol = {
        "Original model, auto-remove": policy.outcomes(policy.remove_predicted_hate(orig.predict(Xte)), y_test, groups),
        "Improved model, auto-remove": policy.outcomes(policy.remove_predicted_hate(imp_pred(test.tweet)), y_test, groups),
        "Remove anything flagged": policy.outcomes(policy.decide(p_test, "remove_all_flagged"), y_test, groups),
        "Triage: 5% to human review + nudges": policy.outcomes(policy.decide(p_test, "triage", th), y_test, groups),
    }
    out["policies"] = pol
    out["review_budget_sweep"] = policy.review_budget_sweep(p_val, val.label.to_numpy(), p_test, y_test, groups)
    plots.policies(list(pol.items()))
    plots.wrongful_by_dialect(list(pol.items()))
    plots.review_sweep(out["review_budget_sweep"], pol["Improved model, auto-remove"])

    (RESULTS / "results.json").write_text(json.dumps(to_json(out), indent=2))
    log("Done. Results in results/results.json, figures in figures/")


if __name__ == "__main__":
    main()

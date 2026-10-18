"""Try the models on your own sentences.

    python demo.py "I'm a proud Muslim woman" "this game is so damn hard lol"
    python demo.py            # runs a built-in set of examples

Shows what the original and improved models predict, how confident the improved model is,
and what the triage policy would do with each post. Models are trained once (about a minute)
and cached in data/models.pkl.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ald import data, glove, models, policy  # noqa: E402
from ald.config import CLASSES, DATA  # noqa: E402
from ald.text import clean_text, original_tokens  # noqa: E402

CACHE = DATA / "models.pkl"

# Harmless examples, a counter-speech example, and two HateCheck test cases (Röttger et al., 2021).
EXAMPLES = [
    "Had a great day at the beach with my family",
    "I'm a proud Muslim woman",
    "I'm queer and I love who I am",
    "My gay best friend is the funniest person I know",
    "yall really finna make me laugh today lol",
    "this game is so damn hard lol",
    "You are an idiot",
    "Saying immigrants are animals is disgusting and wrong",
    "I hate women.",
    "I don't hate women.",
]


def load_models():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    print("Training models (first run only)...")
    train, val, _ = data.split(data.load_davidson())
    clean = lambda s: [clean_text(t) for t in s]  # noqa: E731
    original = models.original_models()["SVM (rbf, C=10, tuned)"].fit(
        glove.embed([original_tokens(t) for t in train.tweet]), train.label)
    improved = models.improved_model().fit(clean(train.tweet), train.label)
    calibrated = models.calibrate(improved, clean(val.tweet), val.label)
    thresholds = policy.choose_thresholds(calibrated.predict_proba(clean(val.tweet))[:, 0], val.label.to_numpy())
    bundle = {"original": original, "improved": improved, "calibrated": calibrated, "thresholds": thresholds}
    CACHE.write_bytes(pickle.dumps(bundle))
    return bundle


def main(texts):
    m = load_models()
    orig = m["original"].predict(glove.embed([original_tokens(t) for t in texts]))
    cleaned = [clean_text(t) for t in texts]
    imp = m["improved"].predict(cleaned)
    probs = m["calibrated"].predict_proba(cleaned)
    actions = policy.decide(probs, "triage", m["thresholds"])

    action_text = {"allow": "allow", "nudge": "ask author to reconsider", "review": "send to a human", "remove": "remove"}
    for t, o, i, p, a in zip(texts, orig, imp, probs, actions):
        print(f"\n{t}")
        print(f"  original model: {CLASSES[o]:9s}   improved model: {CLASSES[i]:9s}")
        print("  improved model's confidence: " + ", ".join(f"{c} {v:.0%}" for c, v in zip(CLASSES, p)))
        print(f"  triage policy: {action_text[a]}")


if __name__ == "__main__":
    main(sys.argv[1:] or EXAMPLES)

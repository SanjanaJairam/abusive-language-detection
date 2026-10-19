"""Compute the explainer's numbers from the real final model.

    python explainer/make_data.py explainer/data.json "sentence one" "sentence two" ...

The final model is linear: each class score = intercept + sum(feature value x weight). Each
feature comes from specific words of the cleaned text (a word, a word pair, or a character chunk
inside a word), so every feature's push is split across the words it came from, exactly.
Pushes are shown centred across the three classes (softmax ignores a shared shift).
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from ald import data, glove, models, policy  # noqa: E402
from ald.config import CLASSES  # noqa: E402
from ald.explain import mask  # noqa: E402
from ald.pipeline import TextModel  # noqa: E402
from ald.text import _WORD, clean_text, negation_tokens, original_tokens  # noqa: E402

CACHE = REPO / "data" / "explainer_models.pkl"
if CACHE.exists():
    b = pickle.loads(CACHE.read_bytes())
else:
    df = data.load_davidson()
    train, val, test = data.split(df)
    olid = {s: data.load_olid(s) for s in ("train", "val")}
    final = TextModel("final", olid=True, negation=True, swaps="all").fit(train, val, olid["train"], olid["val"])
    original = models.original_models()["SVM (rbf, C=10, tuned)"].fit(
        glove.embed([original_tokens(t) for t in train.tweet]), train.label)
    b = {"final": final, "original": original}
    CACHE.write_bytes(pickle.dumps(b))
final, original = b["final"], b["original"]
_, _val, _ = data.split(data.load_davidson())
_olid_val = data.load_olid("val")
th = policy.choose_thresholds(final.proba(_val.tweet), _val.label.to_numpy(), review_budget=0.10,
                              p_everyday=final.proba(_olid_val[_olid_val.label.eq(2)].tweet))

pipe = final.pipe
words_vec, chars_vec = [t for _, t in pipe.named_steps["features"].transformer_list]
clf = pipe.named_steps["clf"]
n_words = len(words_vec.vocabulary_)
names = np.concatenate([words_vec.get_feature_names_out(), chars_vec.get_feature_names_out()])
coef, intercept = clf.coef_, clf.intercept_


def centred(v):
    v = np.asarray(v, float)
    return v - v.mean(axis=-1, keepdims=True)


def explain(raw):
    text = clean_text(raw)
    display = text.split()                     # what the reader sees, one chip per word
    # Map each negation-marked token to the display word it came from.
    toks = negation_tokens(text)
    # negation_tokens drops clause-ending punctuation; rebuild the index list without those tokens.
    tok_word = [wi for wi, w in enumerate(display) for t in _WORD.findall(w) if t not in {".", ",", "!", "?", ";", ":"}]
    assert len(tok_word) == len(toks), (toks, display)

    x = pipe.named_steps["features"].transform([text]).tocsr()
    push = np.zeros((len(display), 3))
    pieces = [[] for _ in display]
    for j, val in zip(x.indices, x.data):
        contrib = val * coef[:, j]
        name = names[j]
        if j < n_words:                         # a word or a word pair
            parts = name.split(" ")
            hits = [i for i in range(len(toks) - len(parts) + 1) if toks[i:i + len(parts)] == parts]
            share = contrib / (len(hits) * len(parts))
            for i in hits:
                for k in range(len(parts)):
                    push[tok_word[i + k]] += share
            kind = "pair" if len(parts) == 2 else "word"
            for i in hits:
                pieces[tok_word[i]].append((kind, name, contrib))
        else:                                   # a character chunk inside a word (char_wb)
            counts = Counter({wi: f" {w} ".count(name) for wi, w in enumerate(display)})
            total = sum(counts.values())
            for wi, c in counts.items():
                if c:
                    push[wi] += contrib * c / total
                    pieces[wi].append(("chunk", name, contrib * c / total))
    logits = intercept + push.sum(0)
    raw_p = np.exp(logits - logits.max()); raw_p /= raw_p.sum()
    p = final.cal.predict_proba([text])[0]
    label = int(pipe.predict([text])[0])
    acts = policy.decide(p[None, :], "triage", th)[0]
    orig_label = int(original.predict(glove.embed([original_tokens(raw)]))[0])

    def top_pieces(lst):
        # The pieces that push this word hardest toward any class, for the "pieces" step.
        lst = sorted(lst, key=lambda t: -np.abs(centred(t[2])).max())[:6]
        return [{"kind": k, "text": mask(n.replace(" ", "·")) if k == "chunk" else mask(n),
                 "push": [round(float(v), 3) for v in centred(c)]} for k, n, c in lst]

    n_feats = {"word": int(sum(1 for j in x.indices if j < n_words and " " not in names[j])),
               "pair": int(sum(1 for j in x.indices if j < n_words and " " in names[j])),
               "chunk": int(sum(1 for j in x.indices if j >= n_words))}
    return {
        "raw": raw, "clean": text, "tokens": [mask(t) for t in toks],
        "words": [{"text": mask(w), "push": [round(float(v), 3) for v in centred(push[i])],
                   "pieces": top_pieces(pieces[i])} for i, w in enumerate(display)],
        "start": [round(float(v), 3) for v in centred(intercept)],
        "features": n_feats,
        "raw_probs": [round(float(v), 3) for v in raw_p],
        "probs": [round(float(v), 3) for v in p],
        "label": CLASSES[label], "original_label": CLASSES[orig_label],
        "triage": str(acts), "auto_remove": "remove" if label == 0 else "allow",
        "check": round(float(np.abs(raw_p - pipe.predict_proba([text])[0]).max()), 6),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    sentences = sys.argv[2:]
    res = [explain(s) for s in sentences]
    out.write_text(json.dumps({"examples": res, "thresholds": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in th.items()}}, indent=1))
    for r in res:
        print(f"{r['label']:9s} orig={r['original_label']:9s} triage={r['triage']:6s} "
              f"p={r['probs']} check={r['check']}  {r['raw']}")

"""Assemble the explainer page: model numbers + story text + study figures.

    python explainer/build.py explainer/data.json results/results.json explainer/index.html

The story text in META is written for the sentences in make_data.py's order (see README.md here).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
model = json.loads(Path(sys.argv[1]).read_text())        # explainer.json from explainer_data.py
results = json.loads(Path(sys.argv[2]).read_text())      # results/results.json
out = Path(sys.argv[3])

META = [
    ("Works as intended", "An ordinary post",
     "Most words push gently toward neither, and the post is allowed under every policy. Even so, letter chunks pull in odd directions: "
     "“family” starts with the same two letters as a slur that is common in the training tweets, so it leans toward hate. The other words outweigh it."),
    ("Works as intended", "Swearing, not hate",
     "“damn” is the strongest push, toward offensive. That matches how the training tweets were labelled: profanity counts as offensive, not hate. "
     "Hate stays at 1%, so a hate policy leaves the post up, and triage asks the author to reconsider instead of removing it. The original model called this post neither."),
    ("Negation", "A strong opinion",
     "“hate” pushes toward hate (+1.16), but the food words pull the other way and the post is labelled neither, just: 51% against 47% offensive. "
     "Hate stays at 2%. Now compare the next post, which only adds “don't”."),
    ("Negation", "The same opinion, softened",
     "Adding “don't” makes things worse. Negation marking turns “hate” into <span class=\"mono\">neg_hate</span>, as designed, but 21 of the 23 training tweets containing “don't hate” were labelled offensive, "
     "so <span class=\"mono\">neg_hate</span> learned to push toward offensive (+0.78 here). The kinder sentence goes from neither to 69% offensive, and triage asks the author to reconsider."),
    ("Where it fails", "Disliking a music genre",
     "Not liking rock music isn't hate speech, but the model gives it a 15% chance of hate, just above the 13% review bar, so triage would send it to a person. "
     "“hate” pushes hard toward hate (+1.57), and “music” adds to it: 14% of the 74 training tweets containing “music” were labelled hate. Over-flagging like this is what fills reviewers' queues."),
    ("Where it fails", "An identity word",
     "A harmless sentence from an identity-term test set. “queer” alone pushes +3.03 toward hate, the strongest push on this page, because 36% of training tweets containing it were labelled hate. "
     "The model labels the sentence hate. Auto-removal would delete it; triage sends it to a person instead."),
    ("Where it fails", "Objecting to an insult",
     "Someone pushing back against rudeness. Repeating the insult brings in “idiots”, which pushes +2.14 toward offensive, and “rude” and “wrong” barely help. "
     "The model calls it 91% offensive, so triage nudges the person who was objecting. Counter-speech against real hate trips the model the same way: in the study's tests, 12% of it was called hate outright."),
    ("Where it fails", "African American English",
     "Ordinary speech with habitual “be” and “tho”; the dialect model rates its language 62% African American English. No single word is strong, but most push slightly toward offensive, reaching 59%. "
     "It falls just under the nudge bar, so triage allows it, but the model still labels it offensive and “remove anything flagged” would delete it. Across the dataset, that pattern is what makes harmless AAE-aligned posts far more likely to be flagged."),
    ("Better than before", "A proud identity statement",
     "“Muslim” still pushes toward hate (+1.09), but “proud” and the starting point outweigh it, and the post is labelled neither at 63%. "
     "The original model called this post offensive. Training on everyday tweets and swapping identity words is what changed it."),
]
assert len(META) == len(model["examples"])
for ex, (group, title, takeaway) in zip(model["examples"], META):
    ex.update(group=group, title=title, takeaway=takeaway)

d = results["dialect"]["threshold_0.6"]["Final model"]
pol = results["policies"]
auto, tri = pol["Final model, auto-remove"], pol["Triage: 10% to human review"]
orig = results["original"]["SVM (rbf, C=10, tuned)"]
facts = [
    {"big": f"{orig['per_class']['hate']['recall']:.0%} of hate caught",
     "text": f"The original model scored {orig['accuracy']:.0%} accuracy, but it labelled almost every hateful tweet merely offensive. Accuracy hid the failure."},
    {"big": f"{d['aae']['false_flag_rate']:.0%} vs {d['white']['false_flag_rate']:.0%}",
     "text": "Harmless AAE-aligned tweets flagged by the final model, against harmless White-aligned tweets. The bias starts in the human labels the model learned from."},
    {"big": f"{tri['hate_caught']:.0%} vs {auto['hate_caught']:.0%}",
     "text": f"Hate caught by sending the riskiest 10% of posts to a person, against automatic removal, which also wrongly removed {auto['wrongful_removals']} harmless posts. Assumes reviewers are always right."},
]
model["facts"] = facts
html = (HERE / "template.html").read_text().replace("/*DATA*/", json.dumps(model, ensure_ascii=False))
if out.name == "index.html":  # a standalone page needs its own document skeleton
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    base = ("<style>\n[hidden] { display: none !important }\n"
            "html { padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px) }\n")
    html = (head + html.replace("<style>", base, 1).replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1)
            + "\n</body>\n</html>\n")
out.write_text(html)
print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
for f in facts:
    print(" ", f["big"], "|", f["text"][:80])

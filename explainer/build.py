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
     "Words mostly push toward neither. Allowed under every policy."),
    ("Works as intended", "Swearing, not hate",
     "“damn” pushes toward offensive, but hate stays at 1%. Triage nudges instead of removing."),
    ("Negation", "A strong opinion",
     "“hate” pushes toward hate, but the post is still labelled neither, just. Compare the next one."),
    ("Negation", "The same opinion, softened",
     "Adding “don't” makes it worse: 69% offensive. In training, 21 of 23 tweets with “don't hate” were offensive."),
    ("Where it fails", "Disliking a music genre",
     "Not hate speech, but 15% hate is over the review bar, so a person would have to check it."),
    ("Where it fails", "An identity word",
     "A harmless sentence called 62% offensive. Harmless sentences with “latinx” get flagged 80% of the time, and 92% by a modern model."),
    ("Where it fails", "Objecting to an insult",
     "Repeating the insult makes it 91% offensive, so the person pushing back gets nudged."),
    ("Where it fails", "African American English",
     "Everyday speech, labelled offensive. Across the dataset, harmless AAE-aligned posts were flagged 37% of the time vs 6% for White-aligned."),
    ("Better than before", "A proud identity statement",
     "Now labelled neither. The original model called it offensive."),
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
     "text": f"by the original model, despite {orig['accuracy']:.0%} accuracy."},
    {"big": f"{d['aae']['false_flag_rate']:.0%} vs {d['white']['false_flag_rate']:.0%}",
     "text": "harmless AAE-aligned vs White-aligned tweets flagged."},
    {"big": f"{tri['hate_caught']:.0%} vs {auto['hate_caught']:.0%}",
     "text": f"hate caught with human review vs auto-removal (which also removed {auto['wrongful_removals']} harmless posts)."},
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

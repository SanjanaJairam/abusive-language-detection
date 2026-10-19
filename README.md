# Who Does a Hate-Speech Filter Fail?

This project started as a course assignment where I trained models to classify tweets as hateful, offensive, or neither. I wanted to revisit it and see what accuracy scores weren't telling me, especially whose language was being flagged unfairly.

**[Try the interactive explainer →](https://abusive-language-detection-explain.vercel.app/)**

> **Content note:** The datasets contain hateful language. No examples are shown here.

## What I Did

I compared my original classifier with an improved model using around 25,000 labelled tweets and additional everyday tweets. I tested both models for mistakes involving identity terms, African American English (AAE), and counter-speech.

I also explored whether adding human review could make moderation decisions safer.

## What I Found

- **Accuracy was misleading.** My original model had 86% accuracy but only caught 2% of actual hate speech.
- **Bias was still a problem.** The improved model flagged 37% of harmless AAE-aligned tweets, compared to 6% of White-aligned tweets. These are language patterns, not the authors' races.
- **Human review helped.** In my simulation, reviewing the riskiest 10% of posts caught 57% of hate speech, compared to 41% through automatic removal, assuming reviewers made correct decisions.

![Harmless tweets wrongly flagged by dialect](figures/dialect_false_flags.png)

## What I Learned

A model being more accurate doesn't necessarily mean it's fairer. The data it's trained on and how its predictions are used matter just as much.

The experiments use older Twitter data and simulated moderation decisions, so the results aren't meant to represent real-world platform performance.

## Running the Project

```bash
pip install -r requirements.txt
bash get_data.sh
python run_all.py
```

The project also includes three notebooks:

- [Original model](notebooks/01_original_model.ipynb)
- [Safety and bias audit](notebooks/02_safety_audit.ipynb)
- [Modern transformer model](notebooks/03_modern_model.ipynb) ([Open in Colab](https://colab.research.google.com/github/SanjanaJairam/abusive-language-detection/blob/main/notebooks/03_modern_model.ipynb))

## License

Code is under the [MIT License](LICENSE). Datasets and pretrained models retain their original licenses.

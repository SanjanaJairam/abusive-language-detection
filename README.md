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

## Data and Credits

- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *ICWSM*.
- Röttger, P., et al. (2021). HateCheck: Functional tests for hate speech detection models. *ACL*.
- Dixon, L., et al. (2018). Measuring and mitigating unintended bias in text classification. *AIES*.
- Zampieri, M., et al. (2019). Predicting the type and target of offensive posts in social media (OLID). *NAACL*. Used via TweetEval (Barbieri et al., 2020).
- Blodgett, S. L., Green, L., & O'Connor, B. (2016). Demographic dialectal variation in social media: A case study of African-American English. *EMNLP*. `src/ald/dialect.py` is adapted from their code.
- Davidson, T., Bhattacharya, D., & Weber, I. (2019) and Sap, M., et al. (2019), whose dialect-bias audits this one follows.
- Antypas, D., & Camacho-Collados, J. (2023). Robust hate speech detection in social media: A cross-dataset empirical evaluation. *WOAH*. Used in notebook 3.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
  
## License

Code is under the [MIT License](LICENSE). Datasets and pretrained models retain their original licenses.

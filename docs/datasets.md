# Extending this work: newer datasets, and how to credit them

The audit in this repository uses 2016–2018 resources. Since then, research has moved from
"does a classifier detect hate?" toward "how do safety classifiers and AI systems behave with
real people?" These newer, publicly documented resources fit the next steps of this project.

## Newer datasets

| Dataset | What it is | How it could extend this project |
| --- | --- | --- |
| **ToxiGen** (Hartvigsen et al., ACL 2022) | Large machine-generated set of *implicit* hate and benign statements about 13 minority groups | Implicit derogation was among the hardest HateCheck tests; ToxiGen tests that at scale |
| **ToxicChat** (Lin et al., 2023) | Toxicity labels on real user prompts sent to an AI chatbot | Moves the audit from social media posts to human-AI conversation, which is closer to Emi |
| **XSTest** (Röttger et al., NAACL 2024) | Safe prompts that *look* unsafe ("how do I kill a Python process?") | The AI version of Finding 2: measures over-refusal, the chatbot equivalent of wrongful removal |
| **WildGuardMix** (Han et al., 2024) | Prompts and responses labelled for harm, refusal and jailbreaks | Benchmark for comparing an LLM moderator with the classifiers here |

Two older resources are also worth knowing:
- **HateCheck** has multilingual versions, which would address the English-only limitation.
- **The Measuring Hate Speech corpus** (Kennedy et al., 2020) records annotator demographics, so it can test *annotator* bias directly. That's the root cause found in Finding 3.

**Before using any of these**, read its dataset card for the license and terms. Several restrict commercial use or require agreeing to terms before download.

## The natural next experiment

Run an LLM moderator, and an LLM safety classifier such as WildGuard, through the *same* audits:
- **HateCheck and identity terms:** does it over-flag identity words?
- **The dialect audit:** is the AAE false-flag gap smaller or larger?
- **XSTest:** does it over-refuse harmless requests?

Then compare the auto-remove and triage policies again. This answers whether newer AI moderation fixes the failures shown here or repeats them.

## How to use other people's work properly

Using published datasets, code and methods is normal and expected in research, as long as you credit them clearly. Every time:

1. **Cite the paper** in your README and any write-up. Put the full reference in a References section, as this repository does.
2. **Link the original source.** Don't re-upload the data; download it in a script like `get_data.sh`. Many licenses forbid redistribution, and the original page tells readers about updates and terms.
3. **State the license** of each dataset, model and piece of code.
4. **Keep notices on adapted code.** If you adapt code (as `src/ald/dialect.py` adapts TwitterAAE), keep the original copyright and license notice in the file, and say what you changed.
5. **Separate their method from your contribution.** For example: "Following Davidson et al. (2019), I estimate dialect with TwitterAAE; unlike them, I use a 0.6 threshold because…". Reviewers want to see exactly what is new.
6. **For blog posts or tutorials** (Medium and similar): take the *idea*, write your own code, and credit the post in a comment or footnote if it shaped your approach. Copying code without a license is risky, and it's better practice to reimplement from the original paper.

## References for this page

- Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., & Kamar, E. (2022). ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. *ACL*. https://arxiv.org/abs/2203.09509
- Lin, Z., Wang, Z., Tong, Y., Wang, Y., Guo, Y., Wang, Y., & Shang, J. (2023). ToxicChat: Unveiling hidden challenges of toxicity detection in real-world user-AI conversation. *Findings of EMNLP*. https://arxiv.org/abs/2310.17389
- Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). XSTest: A test suite for identifying exaggerated safety behaviours in large language models. *NAACL*. https://arxiv.org/abs/2308.01263
- Han, S., Rao, K., Ettinger, A., Jiang, L., Lin, B. Y., Lambert, N., Choi, Y., & Dziri, N. (2024). WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. https://arxiv.org/abs/2406.18495
- Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. https://arxiv.org/abs/2009.10277

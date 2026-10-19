# How a Filter Reads a Post

A guided explainer: nine example posts, each walked through the five steps the final model takes (clean, split into pieces, weigh each word, add it up, decide). Live at https://abusive-language-detection-explaine.vercel.app/ (Vercel, root directory `explainer`). Locally, open `index.html` in a browser; it needs no server.

Every number comes from the real final model. The model is linear, so each word's push toward *hate*, *offensive* or *neither* adds up exactly to the model's own score (`make_data.py` checks this).

To rebuild after retraining:

```bash
python explainer/make_data.py explainer/data.json \
  "Had a great day at the beach with my family" "this damn printer jammed again" \
  "I hate pineapple on pizza" "I don't hate pineapple on pizza" "I hate rock music" \
  "I am a white person" "Saying rock fans are idiots is rude and wrong" \
  "my brother be cooking for real tho" "I'm a proud Canadian woman"
python explainer/build.py explainer/data.json results/results.json explainer/index.html
```

The explanation for each post lives in `build.py` (`META`). If the numbers change, check those sentences still hold.

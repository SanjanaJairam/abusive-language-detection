# How a Filter Reads a Post

A guided explainer: nine example posts, each walked through the five steps the final model takes (clean, split into pieces, weigh each word, add it up, decide). Open `index.html` in a browser; it needs no server.

Every number comes from the real final model. The model is linear, so each word's push toward *hate*, *offensive* or *neither* adds up exactly to the model's own score (`make_data.py` checks this).

To rebuild after retraining:

```bash
python explainer/make_data.py explainer/data.json \
  "Had a great day at the beach with my family" "this damn printer jammed again" \
  "I hate immigrants" "I don't hate immigrants" "Immigrants are a plague on this country" \
  "I am a queer person" "Saying immigrants are animals is disgusting and wrong" \
  "my brother be cooking for real tho" "I'm a proud Muslim woman"
python explainer/build.py explainer/data.json results/results.json explainer/index.html
```

The explanation for each post lives in `build.py` (`META`). If the numbers change, check those sentences still hold.

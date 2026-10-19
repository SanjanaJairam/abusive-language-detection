#!/usr/bin/env bash
# Download the public datasets and models used in this project into data/.
# None of them are redistributed in this repository; please cite them (see README).
set -euo pipefail
mkdir -p data && cd data

clone() { [ -d "$2" ] || git clone --depth 1 "https://github.com/$1.git" "$2"; }

# Davidson et al. (2017): hate / offensive / neither tweets
clone t-davidson/hate-speech-and-offensive-language hate-speech-and-offensive-language
# Röttger et al. (2021): HateCheck functional tests
clone paul-rottger/hatecheck-data hatecheck-data
# Blodgett, Green & O'Connor (2016): TwitterAAE dialect model
clone slanglab/twitteraae twitteraae
# Dixon et al. (2018): identity-term templates (large repo; only two files are kept)
if [ ! -f unintended-ml-bias-analysis/bias_madlibs_77k.csv ]; then
  clone conversationai/unintended-ml-bias-analysis _bias_tmp
  mkdir -p unintended-ml-bias-analysis
  cp _bias_tmp/archive/unintended_ml_bias/eval_datasets/bias_madlibs_77k.csv unintended-ml-bias-analysis/
  cp _bias_tmp/archive/unintended_ml_bias/bias_madlibs_data/adjectives_people.txt unintended-ml-bias-analysis/identity_terms.txt
  rm -rf _bias_tmp
fi
# Zampieri et al. (2019): OLID offensive / not offensive tweets, as packaged by TweetEval (Barbieri et al., 2020)
if [ ! -f tweeteval/offensive/train_text.txt ]; then
  clone cardiffnlp/tweeteval _tweeteval_tmp
  mkdir -p tweeteval
  cp -r _tweeteval_tmp/datasets/offensive tweeteval/
  rm -rf _tweeteval_tmp
fi
# Pennington, Socher & Manning (2014): GloVe Twitter 25d, as packaged by gensim-data
[ -f glove-twitter-25.gz ] || curl -L -o glove-twitter-25.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/glove-twitter-25.gz

echo "Data ready in data/"

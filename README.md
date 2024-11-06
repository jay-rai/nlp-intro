### NLP-Intro Repository

The goal of this repository is to show an accumulation of both academic and personal work in understanding NLP.

Where `A#` series is primarily work done from my time at UCSC doing NLP. Where the project comes from a combination of the work and personal research done at university.

A1
---
For A1, where are given a dataset of movie reviews prelabeled positive or negative. Expanded on some given code, we create a Naive Bayes and Logistic Regression classifier in order to classify unseen movie reviews.

Run the following to see to see the performance of the models
```
python main.py --model NaiveBayes
python main.py --model LogisticRegression
```

A2
---
For A2 where are given a subset of data from the One Billion Word Language Modeling Benchmark. In the first part of the assignment we are tasked to create an **ngram** language model utilizing **MLE without smoothing**, in order to run this problem utilize the following

```
python main.py --model Ngrams
```

*Note the above prints the results for HDTV . hitting the test numbers assigned*

Part two asks us to utilize *add one smoothing* to run and see the performance of this you can run
```
python main.py --model Ngrams --feature wSmoothing
```

Part 3 asks for smoothing with interpolation, in order to run this model you can utilize the following command
```
python main.py --model Interpolation --feature wSmoothing
```

Running it without the `--feature` or `-f` will run it without smoothing

Note this will run the first deliverable of the assignment where lambdas = 0.1, 0.3, 0.6
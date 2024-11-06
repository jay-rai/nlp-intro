import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))

def getaccuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy_value = correct / len(pred)
    print("Accuracy: %i / %i = %.4f " % (correct, len(pred), accuracy_value))
    return accuracy_value


def read_data(path):
    train_frame = pd.read_csv(path + 'test.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'dev.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, test_frame = read_data(args.path)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']


    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']
    

    if args.model == "AlwaysPredictZero":
        model = AlwaysPredictZero()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
        sklearn_model = MultinomialNB(alpha=1.0)
        sklearn_model.fit(X_train, Y_train)
        sklearn_train_pred = sklearn_model.predict(X_train)
        sklearn_test_pred = sklearn_model.predict(X_test)
        print("===== Sklearn Train Accuracy =====")
        accuracy(sklearn_train_pred, Y_train)
        sklearn_train_accuracy = getaccuracy(sklearn_train_pred, Y_train)
        print("===== Sklearn Test Accuracy =====")
        accuracy(sklearn_test_pred, Y_test)
        sklearn_test_accuracy = getaccuracy(sklearn_test_pred, Y_test)
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()

        sklearn_log_model = LogisticRegression(penalty='l2', max_iter=50)
        sklearn_log_model.fit(X_train, Y_train)
        sklearn_log_model_train_pred = sklearn_log_model.predict(X_train)
        sklearn_log_model_test_pred = sklearn_log_model.predict(X_test)
        print("===== Sklearn Train Accuracy =====")
        accuracy(sklearn_log_model_train_pred, Y_train)
        sklearn_log_train_accuracy = getaccuracy(sklearn_log_model_train_pred, Y_train)
        print("===== Sklearn Test Accuracy =====")
        accuracy(sklearn_log_model_test_pred, Y_test)
        sklearn_log_test_accuracy = getaccuracy(sklearn_log_model_test_pred, Y_test)


    elif args.model == 'BonusClassifier':
        model = BonusClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")


    start_time = time.time()
    model.fit(X_train,Y_train)
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)
    train_accuracy = getaccuracy(model.predict(X_train), Y_train)
    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)
    test_accuracy = getaccuracy(model.predict(X_test), Y_test)
    print("Time for training and test: %.2f seconds" % (time.time() - start_time))


def graphPlots(train_accuracy, test_accuracy, sklearntrain, sklearntest):
    models = ['Our LR w/ L2', 'Sklearn LR w/ L2']
    train_accuracies = [train_accuracy, sklearntrain]
    test_accuracies = [test_accuracy, sklearntest]

    fig, ax = plt.subplots(figsize=(10,6))
    bar_width = 0.35
    index = range(len(models))
    # Plot the bars
    bar1 = ax.bar(index, train_accuracies, bar_width, label='Training Accuracy', color='blue')
    bar2 = ax.bar([i + bar_width for i in index], test_accuracies, bar_width, label='Test Accuracy', color='green')

    # Add labels, title, and legend
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison of Logistic Regression Model', fontsize=15)
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(models, fontsize=12)
    ax.legend()

    # Display the chart
    plt.show()



if __name__ == '__main__':
    main()

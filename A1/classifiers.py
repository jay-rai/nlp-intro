import numpy as np
from collections import Counter
import re
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# You need to build your own model here instead of using existing Python
# packages such as sklearn!

## But you may want to try these for comparison, that's fine.
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression

"""
Utilizng nltk library we can find stop words and remove those safely

"""

stop_words = set(stopwords.words('english'))

def proccess_text(text):
    # remove alphanumeric characters, keep spaces
    # get stop words from pre defined library and remove
    print(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens



class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
              is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where
              N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
            is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the
            number of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(BinaryClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        
        self.class_log_prior = None
        self.word_likelihood = None
        self.vocab = None

    def fit(self, X, Y):
        # Add your code here!
        
        n_samples, n_features = X.shape
        #calculates log of class prior, added during predict
        self.class_log_prior = np.log(np.bincount(Y) / n_samples)

        # likelihood matrix and do the add one smoothing 
        self.word_likelihood = np.ones((2, n_features))
        class_counts = np.ones(2)

        for i in range(n_samples):
            label = Y[i]
            self.word_likelihood[label] += X[i]
            class_counts[label] += np.sum(X[i])

        # counts to probabilities P(w|+/-) count of word in +/- review / total word
        self.word_likelihood[0] /= class_counts[0]
        self.word_likelihood[1] /= class_counts[1]

        # Take log for numerical stability during prediction
        self.word_likelihood = np.log(self.word_likelihood)

        
    
    def predict(self, X):
        # Add your code here!
        """
        Assumption is that the words contionall dependent given class
        calculates the sum of log-probabilities for each word
        """
        
        log_probs = np.dot(X, self.word_likelihood.T) + self.class_log_prior
        return np.argmax(log_probs, axis=1)
    
    



# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, learning_rate = 0.01, epochs = 50, l2_lambda=0.0):
        # Add your code here!
        """
        Intialize what we need (equation given for LR)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.l2_lambda = l2_lambda
        
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        # Add your code here!
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            """
            gradient descent with l2 reg
            """
            dw = (1/n_samples) * np.dot(X.T,(y_pred - Y)) + (self.l2_lambda / n_samples) * self.weights
            db = (1/n_samples) * np.sum(y_pred - Y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        
    
    def predict(self, X):
        # Add your code here!
        z = np.dot(X, self.weights) + self.bias
        y_pred_prob = self.sigmoid(z)
        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]
        return np.array(y_pred_class)


# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()

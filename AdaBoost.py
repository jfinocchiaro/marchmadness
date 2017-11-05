
import gzip
import math
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_20newsgroups_vectorized


class Numbers:
    """
    Class to store Kaggle Competition Data
    """

    def __init__(self, location):
        # Load the dataset
        # with gzip.open(location, 'rb') as f:
        #     train_set, valid_set, test_set = pickle.load(f)
        newsgroups_train = fetch_20newsgroups_vectorized(subset='train')
        newsgroups_test = fetch_20newsgroups_vectorized(subset='test')
        self.train_x = newsgroups_train.data
        self.train_y = newsgroups_train.target
        self.test_x = newsgroups_test.data
        self.test_y = newsgroups_test.target

class AdaBoost:
    '''
    AdaBoost classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, n_estimators=50, learning_rate=1.0):
        '''
        initialize Adaboost classifier
        '''
        # Store the training and test data
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        # Store the parameters for the model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        # Create the model
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=1)

    def train(self):
        """
        trains the model with the training data passed to it
        """
        self.model.fit(self.train_x, self.train_y)
        pass

    def evaluate(self):
        """
        evaluates the accuracy of the training model
        """
        return self.model.score(self.test_x, self.test_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaBoost Classifier Options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("kaggle_data.pkl.gz")

    boost = AdaBoost(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    boost.train()
    acc = boost.evaluate()
    print(acc)

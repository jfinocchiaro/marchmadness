import gzip
import math
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups_vectorized


class Numbers:
    """
    Class to store Kaggle Competition Data
    """

    def __init__(self, train_x_fname, train_y_fname, test_x_fname=None, test_y_fname=None):
        # Load the dataset
        with open(train_x_fname, 'rb') as f:
            train_x = pickle.load(f)
        with open(train_y_fname, 'rb') as f:
            train_y = pickle.load(f)
        self.train_x = train_x
        self.train_y = train_y

        if not test_x_fname is None or not test_y_fname is None:
            # Load the dataset
            with open(test_x_fname, 'rb') as f:
                test_x = pickle.load(f)
            with open(test_y_fname, 'rb') as f:
                test_y = pickle.load(f)
            self.test_x = test_x
            self.test_y = test_y

class AdaBoost:
    '''
    AdaBoost classifier
    '''
    def __init__(self, train_x, train_y, test_x=None, test_y=None, n_estimators=50, learning_rate=1.0):
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
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=self.n_estimators, learning_rate=self.learning_rate)

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

    # Load the data into memory
    train_x_fname = "pickled_files/train_x.p"
    train_y_fname = "pickled_files/train_y.p"
    test_x_fname = "pickled_files/test_x.p"
    test_y_fname = "pickled_files/test_y.p"

    # train_x_fname = "pickled_files/all_train_x.p"
    # train_y_fname = "pickled_files/all_train_y.p"
    # test_x_fname = None
    # test_y_fname = None

    if test_x_fname is None or test_y_fname is None:
        data = Numbers(train_x_fname, train_y_fname)
        boost = AdaBoost(data.train_x[:args.limit], data.train_y[:args.limit])
        scores = cross_val_score(boost.model, data.train_x, data.train_y, cv=3, verbose=2)
        acc = scores.mean()
    else:
        data = Numbers(train_x_fname, train_y_fname, test_x_fname, test_y_fname)
        boost = AdaBoost(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
        boost.train()
        acc = boost.evaluate()

    print(acc)

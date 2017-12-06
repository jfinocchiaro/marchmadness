
import gzip
import math
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
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
    def __init__(self, train_x, train_y, test_x, test_y, base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, learning_rate=1.0):
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
        self.base_estimator = base_estimator

        # Create the model
        self.model = AdaBoostClassifier(self.base_estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate)

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

def cv_performance(classifier, x, y, num_folds):
    """This function evaluates average accuracy in cross validation for a given classifer."""
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []
    accuracy = 1

    for j, split in enumerate(splits):

        train_x = np.array([x[i] for i in split.train])
        train_y = np.array([y[i] for i in split.train])

        test_x = np.array([x[i] for i in split.test])
        test_y = np.array([y[i] for i in split.test])

        classifier.fit(train_x, train_y)
        accuracy = classifier.score(test_x, test_y)

        accuracy_array.append(accuracy)

        print(j)

    return np.mean(accuracy_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaBoost Classifier Options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("kaggle_data.pkl.gz")

    # boost = AdaBoost(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    #
    # # Define the parameter set for each of the types of models
    # boost_param_grid = {'n_estimators': [50, 100], 'learning_rate': [1, 2], 'algorithm': ['SAMME', 'SAMME.R']}
    #
    # # Use GridSearchCV to exhaustively find the tuned hyperparameters
    # boost = GridSearchCV(boost.model, boost_param_grid, cv=3, verbose=2)
    # boost.fit(data.train_x, data.train_y)
    #
    # # Show optimal parameters
    # print('--------------------------------')
    # print(boost.best_params_)

    # Perform cross validation on each of the optimal models and show the accuracy
    boost_best_params = {'algorithm': 'SAMME.R', 'learning_rate': 1, 'n_estimators': 50}
    boost = AdaBoost(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y, n_estimators=boost_best_params['n_estimators'], learning_rate=boost_best_params['learning_rate'])
    boos_acc = cv_performance(boost.model, data.train_x, data.train_y, 5)

    print(boost_acc)

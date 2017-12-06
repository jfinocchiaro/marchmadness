import gzip
import math
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


class Numbers:
    """
    Class to store Kaggle Competition Data
    """

    def __init__(self, location):
        X = pickle.load(open("pickled_files/all_train_x.p", 'rb'))
        Y = pickle.load(open("pickled_files/all_train_y.p", 'rb'))

        training_data, test_data, training_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.train_x = training_data
        self.train_y = training_labels
        self.test_x = test_data
        self.test_y = test_labels

class AdaBoost:
    '''
    AdaBoost classifier
    '''
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None, base_estimator=LogisticRegressionCV(), n_estimators=50, learning_rate=1.0):
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

        # Store the feature vec, for the indices the training array
        self.feature_vec = pickle.load(open("pickled_files/all_feature_vec.p", 'rb'))

        # Create the model
        self.model = AdaBoostClassifier(self.base_estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate)

    def train_season(self, season):
        # Open the whole dataset generated
        X = pickle.load(open("pickled_files/all_train_x.p", 'rb'))
        Y = pickle.load(open("pickled_files/all_train_y.p", 'rb'))
        self.train_x = []
        self.train_y = []

        # Add only items from the season to the training data
        index = 0
        for year in self.feature_vec:
            if year == season:
                for i in range(len(self.feature_vec[year])):
                    self.train_x.append(X[index+i])
                    self.train_y.append(Y[index+i])
            index += len(self.feature_vec[year])

        # Rebuild the model
        self.model = AdaBoostClassifier(self.base_estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
        self.model.fit(self.train_x, self.train_y)

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

    def load(self, filename=None):
        if filename == None:
            return pickle.load(open("pickled_files/adaboost.p", 'rb'))
        else:
            return pickle.load(open(filename, 'rb'))

    def dump(self, filename=None):
        if filename == None:
            pickle.dump(self.model, open("pickled_files/adaboost.p", 'wb'))
        else:
            pickle.dump(self.model, open(filename, 'wb'))

    def predict(self, x):
        """
        evaluates the prediction for a given X
        """
        return self.model.predict(x)


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
    boost.train()
    boost_acc = boost.evaluate()
    boost.dump()

    print(boost_acc)

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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


class Numbers:
    """
    Class to store Kaggle Competition Data
    """

    def __init__(self, train_x_fname, train_y_fname, test_x_fname=None, test_y_fname=None):
        # Load the dataset

        if test_x_fname is None or test_y_fname is None:
            # Load the dataset
            with open(train_x_fname, 'rb') as f:
                X = pickle.load(f)
            with open(train_y_fname, 'rb') as f:
                Y = pickle.load(f)
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

        else:
            with open(train_x_fname, 'rb') as f:
                train_x = pickle.load(f)
            with open(train_y_fname, 'rb') as f:
                train_y = pickle.load(f)
            self.train_x = train_x
            self.train_y = train_y

            with open(test_x_fname, 'rb') as f:
                test_x = pickle.load(f)
            with open(test_y_fname, 'rb') as f:
                train_y = pickle.load(f)
            self.test_x = test_x
            self.test_y = test_y



class KNN:
    '''
    KNN classifier
    '''
    def __init__(self, train_x, train_y, test_x=None, test_y=None, num_nei=9):
        '''
        initialize Adaboost classifier
        '''
        # Store the training and test data
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


        # Create the model
        self.model = KNeighborsClassifier(n_neighbors=num_nei, algorithm='ball_tree')

    def train(self):
        """
        trains the model with the training data passed to it
        """
        self.model = self.model.fit(self.train_x, self.train_y)


    def evaluate(self):
        """
        evaluates the accuracy of the training model
        """
        return self.model.score(self.test_x, self.test_y)

    def saveModel(self, filename='knn.p'):
        pickle.dump(self.model, open(filename, "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Classifier Options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    # Load the data into memory


    train_x_fname = "../AdaBoost/pickled_files/all_train_x.p"
    train_y_fname = "../AdaBoost/pickled_files/all_train_y.p"




    data = Numbers(train_x_fname, train_y_fname)
    knn = KNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    knn = knn.train()
    acc = knn.evaluate()
    print(acc)

    knn.saveModel()

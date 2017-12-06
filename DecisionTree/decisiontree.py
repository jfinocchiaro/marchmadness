from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import gzip
import argparse


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


class DecisionTree:
    def __init__(self):
        self.dt = DecisionTreeClassifier()



    def train(self, training_data, training_labels, test_data, test_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.dt = self.dt.fit(training_data, training_labels)
        #return self.dt


    def scorefunc(self):
        prob_predictions = self.dt.predict_proba(self.test_data)
        print(prob_predictions)
        pred = self.dt.predict(self.test_data)

        score_count = 0
        for x in range(len(pred)):
            if pred[x] == self.test_labels[x]:
                score_count += 1
        accuracy = float(score_count) / len(pred)
        print(accuracy)
        return accuracy

    def predict(self, test_data):
        print(self.dt.predict(test_data))
        self.dt.predict(test_data)

    def cross_validate(self, cv=5):
        dt = DecisionTreeClassifier()
        scores = cross_val_score(self.training_data, self.training_labels, cv=cv)
        print(scores.mean())
        print(scores.std() ** 2)

    def saveModel(self, filename='dt.p'):
        pickle.dump(self.dt, open(filename, "wb"))

    def load(self, filename='dt.p'):
        return pickle.load(open(filename, "rb"))



def main():
    parser = argparse.ArgumentParser(description='Decision Tree Classifier Options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    # Load the data into memory


    train_x_fname = "../AdaBoost/pickled_files/all_train_x.p"
    train_y_fname = "../AdaBoost/pickled_files/all_train_y.p"




    data = Numbers(train_x_fname, train_y_fname)
    print(len(data.train_x[0]))
    dt = DecisionTree()
    dt.train(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    acc = dt.scorefunc()
    print(acc)
    dt.saveModel()



if __name__ == '__main__':
    main()

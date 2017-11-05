from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

def knn(training_data, training_labels, test_data, test_labels):
    nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    nbrs.fit(training_data, training_labels)

    predictions = nbrs.predict(test_data)

    accuracy = accuracy_score(predictions, test_labels)

    print(accuracy)

def cross_validate(training_data, training_labels, test_data, test_labels):
    hyperparams= {'n_neighbors' : [1, 2, 3, 5, 7, 9, 11, 15, 19, 21, 25]}
    nbrs = KNeighborsClassifier()
    gscv = GridSearchCV(nbrs, hyperparams)
    gscv.fit(training_data, training_labels)
    predictions = gscv.predict(test_data)
    print(gscv.best_score_)
    print(gscv.best_estimator_.n_neighbors)



iris = load_iris()
X = iris.data[:,:2]
y = iris.target

training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
print (len(training_data))
print (len(training_labels))
print (len(test_data))
print (len(test_labels))

cross_validate(training_data, training_labels, test_data, test_labels)

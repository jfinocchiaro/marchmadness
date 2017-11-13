from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

def knn(training_data, training_labels, test_data, test_labels, num_nei):
    nbrs = KNeighborsClassifier(n_neighbors=num_nei, algorithm='ball_tree')
    nbrs.fit(training_data, training_labels)

    predictions = nbrs.predict(test_data)

    accuracy = accuracy_score(predictions, test_labels)
    print(accuracy)

def cross_validate(training_data, training_labels, test_data, test_labels):
    hyperparams= {'n_neighbors' : [1, 5, 7, 9, 11, 15 ]}
    nbrs = KNeighborsClassifier()
    gscv = GridSearchCV(nbrs, hyperparams)
    gscv.fit(training_data, training_labels)
    predictions = gscv.predict(test_data)
    print(gscv.best_score_)
    print(gscv.best_estimator_.n_neighbors)



def main():
    feat_vectors_file = open('decay_True_normalized_feature_vec.p','rb')
    feat_vectors = pickle.load(feat_vectors_file)
    feat_vectors_file.close()
    tuple_file = open('season_tuples.p','rb')
    feat_labels = pickle.load(tuple_file)
    tuple_file.close()

    num_features = len(feat_vectors['2003']['1104'])

    for i in range(1,num_features):
        X = []
        y = []
        j = 0
        for game in feat_labels:
            #X.append((feat_vectors[game[0]][game[1]][:i]) + ((feat_vectors[game[0]][game[2]][:i])))
            X.append(list(feat_vectors[game[0]][game[1]][:i]))
            X[j].extend(list(feat_vectors[game[0]][game[2]][:i]))

            y.append(game[3])
            j += 1

        #print(X[0])
        #feat_vectors[game[0]][game[1]][:i] + feat_vectors[game[0]][game[2]][:i]
        training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

        knn(training_data, training_labels, test_data, test_labels, 9)


if __name__ == '__main__':
    main()

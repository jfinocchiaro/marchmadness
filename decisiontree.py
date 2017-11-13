from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle

def scorefunc(pred, test_labels):
    score_count = 0
    for x in range(len(pred)):
        if pred[x] == test_labels[x]:
            score_count += 1
    return float(score_count) / len(pred)


def decisiontree(training_data, training_labels, test_data, test_labels):
    dt = DecisionTreeClassifier()
    dt.fit(training_data, training_labels)

    prob_predictions = dt.predict_proba(test_data)
    print(prob_predictions)
    predictions = dt.predict(test_data)
    accuracy = scorefunc(predictions, test_labels)
    print(accuracy)


def cross_validate(training_data, training_labels):
    dt = DecisionTreeClassifier()
    scores = cross_val_score(training_data, training_labels, cv=5)
    print(scores.mean())
    print(scores.std() ** 2)



feat_vectors_file = open('decay_True_normalized_feature_vec.p','rb')
feat_vectors = pickle.load(feat_vectors_file)
feat_vectors_file.close()

tuple_file = open('season_tuples.p','rb')
feat_labels = pickle.load(tuple_file)
tuple_file.close()



X = []
y = []
j = 0
for game in feat_labels:

    X.append(list(feat_vectors[game[0]][game[1]]))
    X[j].extend(list(feat_vectors[game[0]][game[2]]))
    y.append(game[3])
    j += 1


print(X[0])



#y = np.array(y)
#y.reshape(-1,1)
#print(X[0])
#print(y[0])

training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

decisiontree(training_data, training_labels, test_data, test_labels)
quit()
cross_validate(X,y)

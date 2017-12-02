import pickle
import numpy as np
import editdistance
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
feat_vectors_file = open('pickled_files/decay_True_normalized_feature_vec.p','rb')
feat_vectors = pickle.load(feat_vectors_file)
feat_vectors_file.close()

tuple_file = open('pickled_files/season_tuples.p','rb')
feat_labels = pickle.load(tuple_file)
tuple_file.close()

train_x = []
train_y = []
for i, game in enumerate(feat_labels):
    train_x.append(list(feat_vectors[game[0]][game[1]]))
    train_x[i].extend(list(feat_vectors[game[0]][game[2]]))
    train_y.append(game[3])

# Split into train and test data
training_data, test_data, training_labels, test_labels = train_test_split(train_x, train_y, test_size=0.2, random_state=None)

# Troubleshoot
print(feat_labels[0])
print(train_x[0])
print(train_y[0])

pickle.dump(train_x, open('pickled_files/all_train_x.p', 'wb'))
pickle.dump(train_y, open('pickled_files/all_train_y.p', 'wb'))
pickle.dump(training_data, open('pickled_files/train_x.p', 'wb'))
pickle.dump(training_labels, open('pickled_files/train_y.p', 'wb'))
pickle.dump(test_data, open('pickled_files/test_x.p', 'wb'))
pickle.dump(test_labels, open('pickled_files/test_y.p', 'wb'))

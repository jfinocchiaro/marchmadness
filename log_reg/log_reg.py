#!/usr/bin/env python3

import numpy as np
import pickle

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

class LogReg():
	def __init__(self):
		self.l1 = LogisticRegressionCV()

	def test_model(self, X, Y):
		training_data, test_data, training_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)

		parameters = {'fit_intercept':[True, False]}
		self.l1 = GridSearchCV(self.l1, parameters, cv=None, verbose=2)

		#'''
		self.l1 = self.l1.fit(training_data, training_labels)
		print("r2/variance: %s" % self.l1.score(test_data, test_labels))
		print("Residual sum of squares: %.2f" % np.mean(self.l1.predict(test_data) - test_labels) ** 2)
		#'''

		predictions = self.l1.predict(test_data)
		accuracy = accuracy_score(predictions, test_labels)

		print(accuracy)

	def train(self, X, Y):
		parameters = {'fit_intercept':[True, False]}
		self.l1 = GridSearchCV(self.l1, parameters, cv=None, verbose=2)
		self.l1.fit(X, Y)

	def load(self):
		return pickle.load(open("log_reg.p", 'rb'))

	def dump(self):
		pickle.dump(self.l1, open("log_reg.p", 'wb'))

	def predict(self, x):
		prediction = self.l1.predict(x)
		print(prediction)

def main():

	# Open feature vector
	#feature_vec_file = "../decay_True_normalized_feature_vec.p"
	#feature_vec_file = "../decay_False_normalized_feature_vec.p"
	feature_vec_file = "../AdaBoost/pickled_files/all_feature_vec.p"
	feature_vec = pickle.load(open(feature_vec_file, 'rb'))

	# Open training tuples
	tuple_file = "../season_tuples.p"
	feat_labels = pickle.load(open(tuple_file,'rb'))

	X = []
	Y = []
	for season, team_1, team_2, winner in feat_labels:
		#print("season: %s\t| team_1: %s\t| team_2: %s\t| winner: %s" % (season, team_1, team_2, winner))

		try:
			x_vec = list(feature_vec[season][team_1]) + list(feature_vec[season][team_2])
		except:
			continue
		X.append(x_vec)
		Y.append(winner)

		#print("x: %s" % x_vec)
		#print("y: %s\n" % winner)

	#pickle.dump(X, open("log_reg_x.p", 'wb'))
	#pickle.dump(Y, open("log_reg_y.p", 'wb'))

	#X = pickle.load(open("train_x.p", 'rb'))
	#Y = pickle.load(open("train_y.p", 'rb'))
	## Kenpaum
	X = pickle.load(open("../AdaBoost/pickled_files/all_train_x.p", 'rb'))
	Y = pickle.load(open("../AdaBoost/pickled_files/all_train_y.p", 'rb'))
	print(len(X[0]))

	#training_data, test_data, training_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
	#print (len(training_data))
	#print (len(training_labels))
	#print (len(test_data))
	#print (len(test_labels))

	lr = LogReg()
	lr.test_model(X, Y)
	lr.dump()

	#cross_validate(training_data, training_labels, test_data, test_labels)

if __name__ == "__main__":
	main()

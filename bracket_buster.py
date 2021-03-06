#!/usr/bin/env python3

import pickle
import random

## Remember to include "__init__.py" in the folders with your models
from log_reg.log_reg import LogReg
from AdaBoost.AdaBoost import AdaBoost

class BracketBuster():
	def __init__(self):
		self.train_x = pickle.load(open("AdaBoost/pickled_files/all_train_x.p", 'rb'))
		self.train_y = pickle.load(open("AdaBoost/pickled_files/all_train_y.p", 'rb'))

		self.bracket_seeds = pickle.load(open("bracket_seeds.p", 'rb'))
		self.bracket_tuples = pickle.load(open("bracket_tuples.p", 'rb'))

		self.feature_vec = pickle.load(open("AdaBoost/pickled_files/all_feature_vec.p", 'rb'))

	def seed_predict(self, season, team1, team2):
		#print("season: %s" % season)
		seed1 = self.bracket_seeds[season][team1]
		seed2 = self.bracket_seeds[season][team2]
		return seed1 > seed2

	def log_reg_predict(self, season, team1, team2):
		# Initialize external model
		lr = LogReg()
		lr = lr.load()

		# Load feature vecs for each team
		print("season: %s" % season)
		try:
			x_vec = list(self.feature_vec[season][team1]) + list(self.feature_vec[season][team2])
			return lr.predict([x_vec])
		except:
			return random.random()

	def adaboost_predict(self, season, team1, team2):
		# Initialize external model
		boost = LogReg()
		boost = boost.load()

		# Load feature vecs for each team
		#print("season: %s" % season)
		try:
			x_vec = list(self.feature_vec[season][team1]) + list(self.feature_vec[season][team2])
			return lr.predict([x_vec])
		except:
			if random.random() >= 0.5:
				return 1
			else:
				return 0


	def test_brackets(self):
		for season in self.bracket_tuples:
			correct_seed_guesses = 0
			correct_model_guesses = 0
			total_guesses = 0
			for game in range(len(self.bracket_tuples[season])):
				total_guesses += 1

				matchup = self.bracket_tuples[season][game]
				team1, team2, winner = matchup

				# Add model predicts here
				prediction = self.seed_predict(season, team1, team2)
				if prediction == winner:
					correct_seed_guesses += 1

				# Test model
				#prediction = self.log_reg_predict(season, team1, team2)
				prediction = self.adaboost_predict(season, team1, team2)
				if prediction == winner:
					correct_model_guesses += 1

			accuracy = float(correct_seed_guesses) / total_guesses
			print("SEED:\tSeason: %s\t| Accuracy: %s" % (season, accuracy))
			accuracy = float(correct_model_guesses) / total_guesses
			# print("LOGREG:\tSeason: %s\t| Accuracy: %s" % (season, accuracy))
			print("ADABOOST:\tSeason: %s\t| Accuracy: %s" % (season, accuracy))

def main():
	bb = BracketBuster()
	bb.test_brackets()

if __name__ == "__main__":
	main()

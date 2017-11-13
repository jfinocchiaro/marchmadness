import numpy as np
import pickle
import random

def training_tuples(filename):

	results = []
	with open(filename, "r") as fi:
		next(fi)

		for lines in fi:
			# Parse the line
			l = lines.rstrip('\n').split(',')
			#print(len(lineinfo))

			season  = l[0]
			day_num = l[1]
			w_team  = l[2]
			w_score = int(l[3])
			l_team  = l[4]
			l_score = int(l[5])

			if random.random() < 0.5:
				results.append((season, w_team, l_team, 1))
			else:
				results.append((season, l_team, w_team, 0))

	return results

season_file = "../data/RegularSeasonDetailedResults.csv"
results = training_tuples(season_file)
#results = pickle.load(open("season_tuples.p"))

pickle.dump(results, open("season_tuples.p", "wb"))
print(results)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

bracket_file = "../data/TourneyDetailedResults.csv"
results = training_tuples(season_file)
#results = pickle.load(open("bracket_tuples.p"))

pickle.dump(results, open("pickled_files/bracket_tuples.p", "wb"))
print(results)

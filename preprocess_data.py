#!/usr/bin/env python

import numpy as np
import math
import re
import pickle
import random

from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn import preprocessing

def get_decay_time(data, season, team, norm):
	# Arbitrary normalization
	return data[season][team]["num_games"] / float(norm)

def init_feature_vector(decay_rate=0, norm=50):
	dir = "data/"
	detailed_results = str(dir)+"RegularSeasonDetailedResults.csv"

	data = {}
	location = {"H": 1, "N": 0, "A": -1}
	with open(detailed_results) as fi:
		header = fi.readline().rstrip('\r\n').split(',')
		
		#c = 0
		for lines in fi:
			'''c += 1
			if (c == 4616): # 4616 for 2003 data
				break'''

			# Parse the line
			l = lines.rstrip('\r\n').split(',')
			#print(len(lineinfo))

			season  = l[0]
			day_num = l[1]
			w_team  = l[2]
			w_score = int(l[3])
			l_team  = l[4]
			l_score = int(l[5])
			w_loc   = l[6]
			#numot  = l[7]

			# Init for new keys
			if season not in data:
				data[season] = {}
			if w_team not in data[season]:
				data[season][w_team] = {"total_score": 0, "loc_sum": 0, "num_games": 0, "game_results": [], "end_streak": 0, "max_streak": 0, "total_def_reb": 0, "total_off_reb": 0, "total_num_plays": 0}
			if l_team not in data[season]:
				data[season][l_team] = {"total_score": 0, "loc_sum": 0, "num_games": 0, "game_results": [], "end_streak": 0, "max_streak": 0, "total_def_reb": 0, "total_off_reb": 0, "total_num_plays": 0}

			# Standard metrics for winning team
			data[season][w_team]["num_games"]   += 1
			decay_w = get_decay_time(data, season, w_team, norm)
			data[season][w_team]["total_score"] += w_score * decay_w if decay_rate else w_score
			data[season][w_team]["loc_sum"]     += location[w_loc] 
			data[season][w_team]["game_results"].append(1)
			data[season][w_team]["end_streak"]  += 1 
			data[season][w_team]["max_streak"]  = max(data[season][w_team]["max_streak"], data[season][w_team]["end_streak"])

			# Standard metrics for losing team
			data[season][l_team]["num_games"]   += 1 
			decay_l = get_decay_time(data, season, l_team, norm)
			data[season][l_team]["total_score"] += l_score * decay_l if decay_rate else l_score
			data[season][l_team]["loc_sum"]     += -location[w_loc]
			data[season][l_team]["game_results"].append(-1)
			data[season][l_team]["end_streak"]  = 0

			num_features = 13
			for i in range(8, 21):
				if header[i] not in data[season][w_team]:
					data[season][w_team][header[i]] = 0
				if header[i] not in data[season][l_team]:
					data[season][l_team][header[i]] = 0

				data[season][w_team][header[i]] += int(l[i]) * decay_w if decay_rate else int(l[i])
				data[season][l_team][header[i]] += int(l[i+num_features]) * decay_l if decay_rate else int(l[i+num_features])



			# Winning team additional metrics
			data[season][w_team]["total_def_reb"] += float(data[season][w_team]["dr"]) / (data[season][w_team]["dr"] + data[season][l_team]["or"]) * decay_w if decay_rate else float(data[season][w_team]["dr"]) / (data[season][w_team]["dr"] + data[season][l_team]["or"])
			data[season][w_team]["total_off_reb"] += float(data[season][w_team]["or"]) / (data[season][w_team]["or"] + data[season][l_team]["dr"]) * decay_l if decay_rate else float(data[season][w_team]["or"]) / (data[season][w_team]["or"] + data[season][l_team]["dr"])

			#data[season][w_team]["total_num_plays"] = data[season][w_team]["blk"] + data[season][w_team]["stl"] + data[season][w_team]["or"] + data[season][w_team]["dr"]

			# Losing team additional metrics
			data[season][l_team]["total_def_reb"] += float(data[season][l_team]["dr"]) / (data[season][l_team]["dr"] + data[season][w_team]["or"]) * decay_w if decay_rate else float(data[season][l_team]["dr"]) / (data[season][l_team]["dr"] + data[season][w_team]["or"]) 
			data[season][l_team]["total_off_reb"] += float(data[season][l_team]["or"]) / (data[season][l_team]["or"] + data[season][w_team]["dr"]) * decay_l if decay_rate else float(data[season][l_team]["or"]) / (data[season][l_team]["or"] + data[season][w_team]["dr"])
			#data[season][w_team]["total_num_plays"] = data[season][w_team]["blk"] + data[season][w_team]["stl"] + data[season][w_team]["or"] + data[season][w_team]["dr"]

			#print("Season = %s\t| W_team = %s\t| L_team = %s" % (lineinfo[0], lineinfo[2],lineinfo[4]))

	return data, header

def add_vector_averages(data, header):
	for season in data:
		for team in data[season]:
			num_games = data[season][team]["num_games"]

			for feat in header[8:21]:
				new_feature = "avg_"+feat
				total_val = data[season][team][feat]

				data[season][team][new_feature] = float(total_val)/num_games

			# Average Scores
			total_score = data[season][team]["total_score"]
			data[season][team]["avg_score"] = total_score / num_games

			# Average rebound percentage
			data[season][team]["avg_off_reb_percentage"] = data[season][team]["total_off_reb"] / float(num_games)
			data[season][team]["avg_def_reb_percentage"] = data[season][team]["total_def_reb"] / float(num_games)

	return data

def add_seeds(data):
	seed_data = "data/TourneySeeds.csv"

	with open(seed_data, "rb") as fi:
		next(fi)
		for line in fi:
			season, seed_string, team = line.rstrip('\r\n').split(",")
			seed = 1 / float(re.findall('\d+', seed_string)[0])

			if int(season) < 2003:
				continue

			#print("Season: %s\t| seed: %s\t| team: %s" % (season, seed, team))

			data[season][team]["seed_string"] = seed_string
			data[season][team]["bracket_seed"] = seed

	return data

def add_momentum(data, decay_rate):
	for season in data:
		for team in data[season]:
			data[season][team]["momentum"] = 0
			num_games = data[season][team]["num_games"]

			for i in range(len(data[season][team]["game_results"])):
				decay_t = num_games - i
				game_result = data[season][team]["game_results"][i]

				data[season][team]["momentum"] += game_result * np.exp(-decay_t * decay_rate)

	return data

def add_percentages(data):
	for season in data:
		for team in data[season]:
			data[season][team]["win_percentage"] = 0
			num_games = data[season][team]["num_games"]

			for i in range(len(data[season][team]["game_results"])):
				game_result = data[season][team]["game_results"][i]

				if game_result == 1:
					data[season][team]["win_percentage"] += 1

			data[season][team]["win_percentage"] /= float(num_games)

			##
			data[season][team]["fg_percentage"] = data[season][team]["fgm"] / float(data[season][team]["fga"])
			data[season][team]["fg3_percentage"] = data[season][team]["fgm3"] / float(data[season][team]["fga3"])
			data[season][team]["ft_percentage"] = data[season][team]["ftm"] / float(data[season][team]["fta"])

	return data

def decay_test(data):
	for decay_rate in decay_rates:

		data = add_momentum(data, decay_rate)

		x = []
		y = []
		for team in data["2003"]:
			x.append(team)
			y.append(data["2003"][team]["momentum"])

		plt.scatter(x, y)
		plt.title("Decay Rate: %s" % (decay_rate))
		plt.savefig("decay_rate_"+str(decay_rate)+".png")
		plt.show()

def print_feature(feat):
	for season in data:
		for team in data[season]:
			print("Season: %s\t| Team: %s\t| %s: %s" % (season, team, feat, data[season][team][feat]))

def load_data():
	data = pickle.load(open("data.p", "rb"))
	return data

def dump_data(data):
	pickle.dump(data, open("data.p", "wb"))

def feature_vectorizor(data, feature_list):
	feature_vec = {}
	normalize_vec = []

	for season in data:
		for team in data[season]:
			for feature in feature_list:
				if season not in feature_vec:
					feature_vec[season] = {}
				if team not in feature_vec[season]:
					feature_vec[season][team] = []

				if feature not in data[season][team]:
					#print("data['%s']['%s']: has no %s" % (season, team, feature))
					feature_vec[season][team].append(-1)
				else:
					feature_vec[season][team].append(data[season][team][feature])

			normalize_vec.append(feature_vec[season][team])

	# Normalize
	features_normalized = preprocessing.normalize(normalize_vec, axis=0)

	i=0
	#print("Total [season][team]: %s\t| feature_vec: %s" % (len(data)*len(data["2003"]), len(features_normalized)))

	for season in data:
		for team in data[season]:
			feature_vec[season][team] = features_normalized[i]
			i += 1

	return feature_vec
	
def training_tuples(filename):

	results = []
	with open(filename, "rb") as fi:
		next(fi)

		for lines in fi:
			# Parse the line
			l = lines.rstrip('\r\n').split(',')
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialize variables
norm = 50
decay_rate = 0
decay_rates = [1, 1.1, 1.2, 1.3, 1.4, 1.5]

feature_list = ['avg_def_reb_percentage', 'avg_score', 'avg_fgm3', 'avg_dr', 'avg_fga3', 'avg_off_reb_percentage', 'end_streak', 'avg_stl', 'avg_ast', 'fg_percentage', 'avg_or', 'momentum', 'avg_fgm', 'fg3_percentage', 'avg_fga', 'win_percentage', 'num_games', 'avg_blk', 'avg_ftm', 'avg_fta', 'max_streak', 'ft_percentage', 'avg_to', 'avg_pf', 'bracket_seed']



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialize data
#data = load_data()
'''
if decay_rate == 0:
	data_file = "data.p"
else:
	data_file = "decay_"+str(decay_rate)+"_data.p"
data = pickle.load(open(data_file))
'''

## Generate data
data, header = init_feature_vector(decay_rate, norm)

## Add additional metrics
#'''
data = add_vector_averages(data, header)
data = add_momentum(data, decay_rates[0])
data = add_percentages(data)
data = add_seeds(data)
#'''

## Dump data
#dump_data(data)
#'''
if decay_rate == 0:
	data_file = "data.p"
else:
	data_file = "decay_"+str(decay_rate)+"_data.p"
pickle.dump(data, open(data_file, "wb"))
#'''


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialize feature_vec
'''if decay_rate == 0:
	feature_file = "normalized_feature_vec.p"
else:
	feature_file = "decay_"+str(decay_rate)+"_normalized_feature_vec.p"
feature_vec = pickle.load(open(feature_file))
'''

## Generate Data
feature_vec= feature_vectorizor(data, feature_list)

## Dump feature_vec
#'''
if decay_rate == 0:
	feature_file = "normalized_feature_vec.p"
else:
	feature_file = "decay_"+str(decay_rate)+"_normalized_feature_vec.p"
pickle.dump(feature_vec, open(feature_file, "wb"))
#'''


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialize Season data queries
season_file = "data/RegularSeasonDetailedResults.csv"
#results = training_tuples(season_file)
#results = pickle.load(open("season_tuples.p"))

#pickle.dump(results, open("season_tuples.p", "wb"))
#print(results)
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

## Initialize Bracket data queries
bracket_file = "data/TourneyDetailedResults.csv"
#results = training_tuples(bracket_file)
#results = pickle.load(open("bracket_tuples.p"))

#pickle.dump(results, open("bracket_tuples.p", "wb"))
#print(results)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Random printing
for i in range(len(feature_list)):
	print("%s:\t%s" % (feature_list[i], feature_vec["2003"]["1104"][i]))


#!/usr/bin/env python

import numpy as np
import math
from collections import defaultdict
from matplotlib import pyplot as plt

def init_feature_vector():
	dir = "data/"
	detailed_results = str(dir)+"RegularSeasonDetailedResults.csv"

	data = {}
	location = {"H": 1, "N": 0, "A": -1}
	with open(detailed_results) as fi:
		header = fi.readline().rstrip('\r\n').split(',')
		
		c = 0
		for lines in fi:
			c += 1
			if (c == 4616):
				break

			# Parse the line
			l = lines.rstrip('\r\n').split(',')
			#print(len(lineinfo))

			season  = l[0]
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
			data[season][w_team]["total_score"] += w_score
			data[season][w_team]["loc_sum"]     += location[w_loc]
			data[season][w_team]["num_games"]   += 1
			data[season][w_team]["game_results"].append(1)
			data[season][w_team]["end_streak"]  += 1
			data[season][w_team]["max_streak"]  = max(data[season][w_team]["max_streak"], data[season][w_team]["end_streak"])

			# Standard metrics for losing team
			data[season][l_team]["total_score"] += l_score
			data[season][l_team]["loc_sum"]     += -location[w_loc]
			data[season][l_team]["num_games"]   += 1
			data[season][l_team]["game_results"].append(-1)
			data[season][l_team]["end_streak"]  = 0

			num_features = 13
			for i in range(8, 21):
				if header[i] not in data[season][w_team]:
					data[season][w_team][header[i]] = 0
				if header[i+num_features] not in data[season][l_team]:
					data[season][l_team][header[i]] = 0

				data[season][w_team][header[i]]              += int(l[i])
				data[season][l_team][header[i]] += int(l[i+num_features])

			# Winning team additional metrics
			data[season][w_team]["total_def_reb"] += float(data[season][w_team]["dr"]) / (data[season][w_team]["dr"] + data[season][l_team]["or"])
			data[season][w_team]["total_off_reb"] += float(data[season][w_team]["or"]) / (data[season][w_team]["or"] + data[season][l_team]["dr"])
			#data[season][w_team]["total_num_plays"] = data[season][w_team]["blk"] + data[season][w_team]["stl"] + data[season][w_team]["or"] + data[season][w_team]["dr"]

			# Losing team additional metrics
			data[season][l_team]["total_def_reb"] += float(data[season][l_team]["dr"]) / (data[season][l_team]["dr"] + data[season][w_team]["or"])
			data[season][l_team]["total_off_reb"] += float(data[season][l_team]["or"]) / (data[season][l_team]["or"] + data[season][w_team]["dr"])
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

decay_rates = [1, 1.1, 1.2, 1.3, 1.4, 1.5]

data, header = init_feature_vector()
data = add_vector_averages(data, header)
data = add_momentum(data, decay_rates[0])
data = add_percentages(data)

#print_feature("end_streak")
#print_feature("win_percentage")
print_feature("avg_off_reb_percentage")
print(len(data["2003"]["1104"]))

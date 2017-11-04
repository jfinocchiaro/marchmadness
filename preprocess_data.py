#!/usr/bin/env python

import numpy as np
from collections import defaultdict

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
			if (c == 36):
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

			if season not in data:
				data[season] = {}
			if w_team not in data[season]:
				data[season][w_team] = {"total_score": 0, "loc_sum": 0, "num_games": 0}
			if l_team not in data[season]:
				data[season][l_team] = {"total_score": 0, "loc_sum": 0, "num_games": 0}

			data[season][w_team]["total_score"] += w_score
			data[season][w_team]["loc_sum"]     += location[w_loc]
			data[season][w_team]["num_games"]   += 1

			data[season][l_team]["total_score"] += l_score
			data[season][l_team]["loc_sum"]     += -location[w_loc]
			data[season][l_team]["num_games"]   += 1

			num_features = 13
			for i in range(8, 21):
				if header[i] not in data[season][w_team]:
					data[season][w_team][header[i]] = 0
				if header[i+num_features] not in data[season][l_team]:
					data[season][l_team][header[i]] = 0

				data[season][w_team][header[i]]              += int(l[i])
				data[season][l_team][header[i]] += int(l[i+num_features])

			#print("Season = %s\t| W_team = %s\t| L_team = %s" % (lineinfo[0], lineinfo[2],lineinfo[4]))

	return data, header

def add_vector_averages(data, header):
	for season in data:
		for team in data[season]:
			num_games   = data[season][team]["num_games"]

			for feat in header[8:21]:
				new_feature = "avg_"+feat
				total_val = data[season][team][feat]

				data[season][team][new_feature] = float(total_val)/num_games

	return data

data, header = init_feature_vector()
data = add_vector_averages(data, header)

print(data["2003"]["1104"])




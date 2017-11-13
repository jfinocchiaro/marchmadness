import pickle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def init_feature_vector():
	detailed_results = "../data/RegularSeasonDetailedResults.csv"

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
				data[season][w_team] = {"end_streak": 0, "max_streak": 0}
			if l_team not in data[season]:
				data[season][l_team] = {"end_streak": 0, "max_streak": 0}

			# Standard metrics for winning team
			data[season][w_team]["end_streak"]  += 1
			data[season][w_team]["max_streak"]  = max(data[season][w_team]["max_streak"], data[season][w_team]["end_streak"])

			# Standard metrics for losing team
			data[season][l_team]["end_streak"]  = 0

	return data, header

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
    min_max = MinMaxScaler()
    features_normalized = min_max.fit_transform(normalize_vec)
    i=0
    #print("Total [season][team]: %s\t| feature_vec: %s" % (len(data)*len(data["2003"]), len(features_normalized)))

    for season in data:
        for team in data[season]:
        	feature_vec[season][team] = features_normalized[i]
        	i += 1

    return feature_vec

feature_list = ['end_streak', 'max_streak']

data, header = init_feature_vector()

print(data['2004'])
max_streak = 0
for team in data['2004']:
    if data['2004'][team]['max_streak'] > max_streak:
        max_streak = data['2004'][team]['max_streak']
print(max_streak)
# #data = load_data()
#
feature_vec= feature_vectorizor(data, feature_list)
max_streak = 0
for team in feature_vec['2004']:
    if feature_vec['2004'][team][1] > max_streak:
        max_streak = feature_vec['2004'][team][1]
print(max_streak)
print(feature_vec['2004'])
#feature_vec = pickle.load(open("normalized_feature_vec.p"))

pickle.dump(feature_vec, open("pickled_files/normalized_feature_vec.p", "wb"))

import pickle
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
import editdistance

# Load the dataset
feature_vec_fname = 'decay_False_normalized_feature_vec.p'
season_tuples_fname = 'season_tuples.p'
kenpom_fname = 'kenpom.csv'
teams_fname = '../data/Teams.csv'

with open(feature_vec_fname, 'rb') as f:
    feature_vec = pickle.load(f)
with open(season_tuples_fname, 'rb') as f:
    season_tuples = pickle.load(f)
with open(kenpom_fname, 'rb') as f:
    kenpom_data = genfromtxt('kenpom.csv', delimiter=',', names=True)

# Remove all the nan's and year from the data from the array generated from the team name
new_kenpom_data = []
for team in kenpom_data:
    new_team = []
    for i,x in enumerate(team):
        if i != 0 and i != 2 and i != 3:
            if str(x) == 'nan':
                new_team.append(0)
            else:
                new_team.append(x)

    new_kenpom_data.append(new_team)

new_kenpom_data = preprocessing.normalize(new_kenpom_data, axis=0)

# Import the team names for each of the rows in the kenpom data
kenpom_teams = []
kenpom_years = []
with open(kenpom_fname) as f:
    header = f.readline().rstrip('\n').split(',')
    for lines in f:
    	# Parse the line
        l = lines.rstrip('\n').split(',')
        kenpom_teams.append(l[2])
        kenpom_years.append(l[0])

# Import the teams file to cross reference the ID with the name of the team
teams = {}
with open(teams_fname) as f:
    header = f.readline().rstrip('\n').split(',')
    for lines in f:
    	# Parse the line
        l = lines.rstrip('\n').split(',')
        teams[l[0]] = l[1]
#print(teams)

# Concatenate the kenpom data and the feature vec data
new_feature_vec = {}
for x in range(len(kenpom_teams)):
    min_dist = 9999
    min_team = 'None'
    for team in teams.values():
        distance = editdistance.eval(kenpom_teams[x], team)
        if distance <= min_dist:
            min_dist = distance
            min_team = team
    if min_team != None and int(kenpom_years[x]) > 2002:
        if new_feature_vec.get(str(kenpom_years[x])) is None:
            new_feature_vec[str(kenpom_years[x])] = {}
        for k, v in teams.items():
            if v == min_team:
                min_team_id = k
        if not feature_vec.get(str(kenpom_years[x])).get(min_team_id) is None:
            new_feature_vec[str(kenpom_years[x])][min_team_id] = np.concatenate((np.asarray(feature_vec[str(kenpom_years[x])][min_team_id]), np.asarray(new_kenpom_data[x])))
            # print('Year: ', str(kenpom_years[x]))
            # print('Team: ', min_team_id)

train_x = {}
train_y = {}
for i, game in enumerate(season_tuples):
    if not new_feature_vec.get(game[0]).get(game[1]) is None and not new_feature_vec.get(game[0]).get(game[2]) is None:
        train_x[i] = (list(new_feature_vec[game[0]][game[1]]))
        train_x[i].extend(list(new_feature_vec[game[0]][game[2]]))
        train_y[i] = game[3]

# Convert dict to list
train_x = list(train_x.values())
train_y = list(train_y.values())

pickle.dump(train_x, open('train_x.p', 'wb'))
pickle.dump(train_y, open('train_y.p', 'wb'))

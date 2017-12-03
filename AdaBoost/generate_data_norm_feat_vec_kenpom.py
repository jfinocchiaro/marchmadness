import pickle
import numpy as np
import editdistance
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
feature_vec_fname = 'pickled_files/decay_True_normalized_feature_vec.p'
season_tuples_fname = 'pickled_files/season_tuples.p'
kenpom_fname = '../data/kenpom.csv'
teams_fname = '../data/Teams.csv'

with open(feature_vec_fname, 'rb') as f:
    feature_vec = pickle.load(f)
with open(season_tuples_fname, 'rb') as f:
    season_tuples = pickle.load(f)
with open(kenpom_fname, 'rb') as f:
    kenpom_data = genfromtxt(kenpom_fname, delimiter=',', names=True)

# Remove all the inconsequential data and only grab the ranks in the array
new_kenpom_data = []
max_rank = 0
max_ranks = {}
for team in kenpom_data:
    new_team = []
    for i, x in enumerate(team):
        # Only add ranks from the data
        if i == 1 or (i > 8 and i % 2 != 0):
            if str(x) == 'nan':
                new_team.append(0)
            else:
                new_team.append(x)
        # Find the difference of wins to losses
        if i == 4:
            wins = x
        if i == 5:
            new_team.append(wins - x)
    new_kenpom_data.append(new_team)

# Troubleshoot
# print(kenpom_data[0])
# print(kenpom_data.dtype)
# print(new_kenpom_data[0])
# print(new_kenpom_data[1])
# print(new_kenpom_data[500])

# Normalize each individual feature and subtract values from 1 to give a higher score to the better ranking
min_max=MinMaxScaler()
new_kenpom_data = min_max.fit_transform(new_kenpom_data)

for team in new_kenpom_data:
    for i, feature in enumerate(team):
        if i != 1:
            team[i] = 1 - feature

# Import the team names for each of the rows in the kenpom data to cross reference with current feature vec
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

# Populate the train_x and train_y arrays
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

# Split into train and test data
training_data, test_data, training_labels, test_labels = train_test_split(train_x, train_y, test_size=0.2, random_state=None)

# Troubleshoot
print(season_tuples[0])
print(train_x[0])
print(len(train_x[0]))
print(train_y[0])

# Dump data to different pickled files
pickle.dump(new_feature_vec, open('pickled_files/all_feature_vec.p', 'wb'))
pickle.dump(train_x, open('pickled_files/all_train_x.p', 'wb'))
pickle.dump(train_y, open('pickled_files/all_train_y.p', 'wb'))
pickle.dump(training_data, open('pickled_files/train_x.p', 'wb'))
pickle.dump(training_labels, open('pickled_files/train_y.p', 'wb'))
pickle.dump(test_data, open('pickled_files/test_x.p', 'wb'))
pickle.dump(test_labels, open('pickled_files/test_y.p', 'wb'))

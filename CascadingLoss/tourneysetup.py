import numpy as np
import pickle
import random
import sys
sys.path.insert(0, '../')

from bracket_buster import BracketBuster
from log_reg.log_reg import LogReg

def read_in_tourney_slots():
    matchups = {}
    team_list = pickle.load(open('../data/team_list.p', 'rb'))
    seeds = pickle.load(open('../bracket_seeds.p', 'rb'))
    tourney_seeds = '../data/TourneySlots.csv'

    with open(tourney_seeds, 'rb') as fi:
        header = fi.readline()

        for lines in fi:
            # Parse the line
            l = lines.decode('utf-8').rstrip('\r\n').split(',')

            season = l[0]
            game_id = l[1]
            team_1 = l[2]
            team_2 = l[3]

            # Init for new keys
            if season not in matchups:
                matchups[season] = {}



            if game_id.startswith('R1'):
                try:

                    matchups[season][game_id] = (seeds[season][team_1], seeds[season][team_2])
                except:
                    #print(team_2)
                    #print(seeds[season][team_2 + 'a'])

                    matchups[season][game_id] = (seeds[season][team_1], seeds[season][team_2 + 'a'])


            else:
                matchups[season][game_id] = (team_1, team_2)


    pickle.dump(matchups, open('../data/tourney_schedules.p', 'wb'))

def calc_ground_truth():
    #bracket_seeds = pickle.load(open("../bracket_seeds.p", 'rb'))
    bracket_tuples = pickle.load(open("../bracket_tuples.p", 'rb'))
    matchups = pickle.load(open('../data/tourney_schedules.p', 'rb'))

    #print (matchups)


    ground_truth = {}
    for season in matchups.keys():
        if season in bracket_tuples.keys():

            if season not in ground_truth:
                ground_truth[season] = {}


            for game in (bracket_tuples[season]):
                print(game)

                if (bracket_tuples[season][i][2] == 1):
                    winner = bracket_tuples[season][i][0]
                else:
                    winner = bracket_tuples[season][i][1]


                ground_truth[season][game] = winner

    print (ground_truth)

    pickle.dump(ground_truth, open('../data/tourney_gt_by_round.p', 'wb'))

def write_teams():
    tourney_seeds = '../data/TourneySeeds.csv'
    teams = []

    with open(tourney_seeds, 'rb') as fi:
        header = fi.readline()

        for lines in fi:
            # Parse the line
            l = lines.decode('utf-8').rstrip('\r\n').split(',')
            t = l[2]
            if t not in teams:
                teams.append(t)

    pickle.dump(teams, open('../data/team_list.p', 'wb'))


def perform_predictions():
    matchups = pickle.load(open('../data/tourney_schedules.p', 'rb'))
    seeds = pickle.load(open('../bracket_seeds.p', 'rb'))
    team_list = pickle.load(open('../data/team_list.p', 'rb'))
    feature_vec = pickle.load(open("../AdaBoost/pickled_files/all_feature_vec.p", 'rb'))
    ground_truth = pickle.load(open('../data/tourney_gt_by_round.p', 'rb'))

    lr = LogReg()
    lr = pickle.load(open("../log_reg/log_reg.p", 'rb'))


    bracket = {}
    predictions = {}
    print(matchups['2009']['R1Y1'])


    for season in matchups:
        if int(season) > 2002:
            print ('Season: %s' % season)
            N_GAMES = len(matchups[season].values())
            i = 0
            if season not in predictions.keys():
                predictions[season] = {}

            while(i <= N_GAMES):

                print('i:  %s' % i)
                for game, teams in matchups[season].items():
                    team1 = teams[0]
                    team2 = teams[1]
                    game_id = game
                    #print ('Team 1 %s' %team1)
                    #print ('Team 2 %s' % team2)

                    #print('Game number %s' % game)

                    #print(team1 in team_list)

                    #we can actually predict this game
                    if team1 in team_list and team2 in team_list:

                        try:
                            x_vec = list(feature_vec[season][team1]) + list(feature_vec[season][team2])

                            pred_winner = lr.predict([x_vec])[0]
                            #print('Pred winner %s' % teams[pred_winner])
                            #print('Actual winner %s' % ground_truth[season][game_id])
                        except:
                            print('Skipped a game!!')
                            pred_winner = random.randint(0,1)



                        for n_game, n_teams in matchups[season].items():
                            if game_id in n_teams:
                                ind = matchups[season][n_game].index(game_id)
                                game_tup = list(matchups[season][n_game])
                                game_tup[ind] = teams[pred_winner]
                                predictions[season][game_id] = teams[pred_winner]
                                matchups[season][n_game] = tuple(game_tup)
                                i += 1


    print ('2009 Champ:  %s' % predictions['2009']['R6CH'])

    return matchups, predictions


def calc_acc(predictions):
    ground_truth = pickle.load(open('../data/tourney_gt_by_round.p', 'rb'))

    avg_acc = []

    for season in ground_truth.keys():
        if int(season) > 2002:
            total_games = 0
            total_right = 0



            for game in ground_truth[season].keys():
                try:
                    if predictions[season][game] == ground_truth[season][game]:
                        total_right += 1
                    total_games += 1
                except:
                    pass

            avg_acc.append(float(total_right) / total_games)

    print (avg_acc)
    print(np.mean(avg_acc))
    print(np.std(avg_acc))



if __name__ == '__main__':
    #write_teams()
    #read_in_tourney_slots()
    #calc_ground_truth()
    matchups, predictions = perform_predictions()
    #calc_acc(predictions)

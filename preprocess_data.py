import numpy as np
from collections import defaultdict


def readindata(filename):
    data = defaultdict(list)
    gamesbyyear = defaultdict(list)
    with open(filename) as f:
        next(f)
        i = 0
        for line in f:
            lineinfo = line.rstrip('\r\n').split(',')
            year = lineinfo.pop(0)
            data[i] = [year, lineinfo]
            gamesbyyear[year].append(lineinfo)
            #print data[i]
            i += 1
        print gamesbyyear


readindata('data/TourneyCompactResults.csv')

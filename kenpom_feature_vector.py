from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('kenpom.csv', delimiter=',', names=True)
print(my_data)
print(my_data.dtype.names)
my_data.dump('kenpom.p')


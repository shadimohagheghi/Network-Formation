import networkx as nx
from numpy import linalg as LA
import math
import numpy as np 
import matplotlib
from numpy import inf
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
import matplotlib.pyplot as plt
import scipy.stats as st
from NetworkFormation import Formation
from optimality import Optimal

plt.close("all")
    
'''
W_interconnection=[[1, 1, 1, 1 ], \
				   [1, 1, 1, 1 ], \
				   [1, 1, 1, 1 ], \
				   [1, 1, 1, 1 ]]

W_interconnection=[[1   , 0.7 , 0.2 , 0.1 ], \
				   [0.7 , 1   , 0.1 , 0.3 ], \
				   [0.2 , 0.1 , 1   , 0.05   ], \
				   [0.1 , 0.3 , 0.05   , 1   ]]


W_interconnection = [[1    , 0.75 , 0.1 , 0.05, 0.3 ],\
					 [0.75 , 1    , 0.15, 0.2 , 0.1 ],\
					 [0.1  , 0.15 , 1   , 0   , 0.5 ],\
					 [0.05 , 0.2  , 0   , 1   , 0.35],\
					 [0.3  , 0.1  , 0.5 , 0.35, 1   ]]


W_interconnection = [[1   , 0.3 , 0.1 ],\
					 [0.3 , 1   , 0.3 ],\
					 [0.1 , 0.3 , 1   ]]

'''
W_interconnection = [[1   , 0.25 ],\
					 [0.25 , 1   ]]


c=0.2
delta=0.5

Iterations =500
trials=1
clique_sizes=[3, 5]
n=np.sum(clique_sizes)


n_iterations=[]
u_sum=np.zeros(Iterations)
all_U_sum=np.zeros(Iterations)

for i in range(trials):
	G, u_sum=Optimal(clique_sizes, W_interconnection, Iterations, c, delta)

for u in range(Iterations):
	n_iterations.append(u)





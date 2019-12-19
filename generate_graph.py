import networkx as nx
import collections
import math
import numpy as np
import matplotlib.pyplot as plt

def star(center, n):

	M=np.ones((n,n))

	for i in range(n):
		M[center][i]=1
		M[center][i]=1
		M[i][i]=1

	return M

def line(n):

	M=np.ones((n,n))

	for i in range(n-1):
		M[i][i+1]=1
		M[i+1][i]=1
		M[i][i]=1

	return M

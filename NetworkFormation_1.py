import networkx as nx
from numpy import linalg as LA
import math
import collections
import numpy as np 
import matplotlib
from Payoff_1 import payoff_1
from numpy import inf
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
import matplotlib.pyplot as plt
import scipy.stats as st
from networkx.algorithms import community as comm
from community_louvain import best_partition
plt.close("all")

##########################################################

clique_sizes=3

W_interconnection=[]

'''
W_interconnection.append([0   , 0.3 , 0.1 ])
W_interconnection.append([0.3 , 0   , 0.3])
W_interconnection.append([0.1 , 0.3, 0   ])
'''

W_interconnection.append([0   , 0.9 , 0.1 , 0.1 ])
W_interconnection.append([0.9   , 0   , 0.1, 0.1])
W_interconnection.append([0.1 , 0.1, 0   , 0   ])
W_interconnection.append([0.1 , 0.1, 0   , 0   ])

'''
W_interconnection.append([0   , 0.9 , 0.1 , 0.05, 0.3])
W_interconnection.append([0.9 , 0   , 0.15 , 0.2, 0.1])
W_interconnection.append([0.1 , 0.15 , 0   , 0  , 0.5])
W_interconnection.append([0.05 , 0.2 , 0   , 0  , 0.35])
W_interconnection.append([0.3 , 0.1 , 0.5 , 0.35, 0  ])


rowsums=A1.sum(axis=1)
D1_out=np.diag(rowsums, k=0)
A1_EN1= np.dot(LA.inv(D1_out+np.identity(n1)),A1+np.identity(n1))
'''
n_cliques=int(len(W_interconnection))
n=clique_sizes*n_cliques
W=np.zeros((n, n))

#Comembership

for i in range(n_cliques):
    for j in range(n_cliques):
        for k in range(clique_sizes):
            for l in range(clique_sizes):
                W[i*clique_sizes+l][j*clique_sizes+k]=W_interconnection[i][j]


#redundant ties
for i in range(n_cliques):
    '''
    for j in range(n_cliques):
        for k in range(clique_sizes):
            W[i*clique_sizes+k][j*clique_sizes+k]=W_interconnection[i][j]
    '''
    W[i*clique_sizes][i*clique_sizes+1]=1
    W[i*clique_sizes][i*clique_sizes+2]=1

    W[i*clique_sizes+1][i*clique_sizes]=1
    W[i*clique_sizes+2][i*clique_sizes]=1

    W[i*clique_sizes+1][i*clique_sizes+2]=1
    W[i*clique_sizes+2][i*clique_sizes+1]=1

for i in range(n):
    W[i][i]=0

print W-np.transpose(W)

W1=np.kron(W_interconnection, np.ones((clique_sizes,clique_sizes)))
W1=W1+np.kron(np.identity(n_cliques), np.ones((clique_sizes,clique_sizes)))

for i in range(n):
    W1[i][i]=0

def Formation1(N):
    
    G=nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### 

    U=payoff_1(G,W)
    print "U= ", np.around(U, 5)
    np.set_printoptions(precision=5)
    U_updated=[]

    for u in range(500): 
        [i, j]= np.random.choice(N, 2, replace=False)
        i=int(i)
        j=int(j)
        print [i,j]

        if G.has_edge(i,j):
            G.remove_edge(i,j)
            U_updated=payoff_1(G,W)

            #print "U_updated[i], U_updated[j] ", U_updated[i], U_updated[j]
            #print "U[i], U[j] ", U[i], U[j]
            #print "U_updated= ", np.around(U_updated, 5)

            if (U_updated[i]<=U[i] and U_updated[j]<=U[j]):
            #if (sum(U_updated)<=sum(U)):
                G.add_edge(i,j)
                print "didn't remove"
            else:
                print "removed"
                for q in range(len(U_updated)):
                    U[q]=U_updated[q]
            
        else:
            G.add_edge(i,j)
            U_updated=payoff_1(G,W)

            #print "U_updated[i], U_updated[j] ", U_updated[i], U_updated[j]
            #print "U[i], U[j] ", U[i], U[j]
            #print "U_updated= ", np.around(U_updated, 5)

            np.around(U_updated, 5)
            np.set_printoptions(precision=5)

            if (U_updated[i]>=U[i] and U_updated[j]>=U[j]):  
            #if (sum(U_updated)>=sum(U)):              
                print "added"
                for q in range(len(U_updated)):
                    U[q]=U_updated[q]
            else:
                G.remove_edge(i,j)
                print "didn't add"

    U_updated=payoff_1(G,W)

    print "sum(U), sum(U_updated): ", sum(U), sum(U_updated)
    
    A= nx.to_numpy_matrix(G)
    A = np.array(A)

    print W
    print W1
    print np.kron(np.identity(n_cliques), np.ones((clique_sizes,clique_sizes)))
    weights=[]
    allWeights=[]
    '''
    e=G.number_of_edges()
    for i in range(n):
        for j in range(i, n):
            if i != j and A[i][j]:
                weights.append(W[i][j])
            if i != j and W[i][j]:
                allWeights.append(W[i][j])
    '''
    GW = nx.from_numpy_matrix(W, create_using=nx.MultiGraph())
    for (i, j) in GW.edges():
        allWeights.append(W[i][j])

    for (i, j) in G.edges():
        weights.append(W[i][j])

    print allWeights, np.around(U, 3)
    np.set_printoptions(precision=3)

    #partition = best_partition(G) 

    weights=np.multiply(weights, 1)
    allWeights=np.multiply(allWeights, 10)

    plt.figure(1)
    nx.draw(G, pos=nx.shell_layout(G), weight=weights, width=weights, color= weights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show() 
    '''
    plt.figure(2)
    print GW.edges()
    nx.draw(GW, pos=nx.shell_layout(GW), weight=allWeights, width=allWeights, color= allWeights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
    #nx.draw(G, pos=nx.circular_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show() 
    '''
    return G

#G = Formation(6)


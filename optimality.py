import networkx as nx
from numpy import linalg as LA
import math
import collections
import numpy as np 
import matplotlib
from payoff import payoff
from numpy import inf
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools 
from networkx.algorithms import community as comm
#from community_louvain import best_partition
from NetworkFormation import Formation
plt.close("all")

##########################################################


def Graph_Value(clique_sizes, W_interconnection, Iterations, c, delta):

    Iterations =Iterations

    nodes=np.sum(clique_sizes)

    clique_sizes_cum=np.zeros(nodes)
    clique_sizes_cum[0]=0
    for i in range(1, len(clique_sizes)):
        clique_sizes_cum[i]=clique_sizes_cum[i-1]+clique_sizes[i-1]

    n_cliques=int(len(W_interconnection))
    W=np.zeros((nodes, nodes))

    for i in range(n_cliques):
        for j in range(n_cliques):
            for k in range(clique_sizes[i]):
                for l in range(clique_sizes[j]):
                    W[int(clique_sizes_cum[i]+k)][int((clique_sizes_cum[j])+l)]=W_interconnection[i][j]

    for i in range(nodes):
        W[i][i]=0

    U_sum=[]
    perm=[]
    for i in range(nodes):
        for j in range(i):
            perm.append([i,j])
    #print perm

    G_Bridges=[]
    G_EdgeBundles=[]
    G_Comemberss=[]

                
    # Equilibrium Rule Adding Subgroups

    
    G=nx.Graph()
    for i in range(nodes):
        G.add_node(i)
    
    for i in range(n_cliques):
        for k in range(clique_sizes[i]):
            for l in range(clique_sizes[i]):
                G.add_edge(clique_sizes_cum[i]+k, (clique_sizes_cum[i])+l) 
    '''
    plt.figure(0)
    nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show()
    '''
    n_interC=np.zeros(15)

    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    n_interC[0]=0

    # 1 link
    G.add_edge(0, 3)
    n_interC[1]=n_interC[0]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    # 2 link
    G.add_edge(1, 4)
    n_interC[2]=n_interC[1]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    # 3 link
    G.add_edge(2, 5)
    n_interC[3]=n_interC[2]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))


    G.remove_edge(1, 4)
    G.remove_edge(2, 5)
    n_interC[4]=n_interC[3]-2


    # 2 link
    G.add_edge(0, 4)
    n_interC[5]=n_interC[4]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    # 3 link
    G.add_edge(0, 5)
    n_interC[6]=n_interC[5]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    # 4 link
    G.add_edge(0, 6)
    n_interC[7]=n_interC[6]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))
    # 5 link
    G.add_edge(0, 7)
    n_interC[8]=n_interC[7]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    G.remove_edge(0, 7)
    G.remove_edge(0, 6)
    G.remove_edge(0, 5)
    G.remove_edge(0, 4)
    n_interC[9]=n_interC[8]-4

    #3 links
    G.add_edge(0, 4)
    G.add_edge(1, 4)
    n_interC[10]=n_interC[9]+2
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    #4 links
    G.add_edge(2, 5)
    n_interC[11]=n_interC[10]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    #4 links
    G.remove_edge(0, 4)
    G.add_edge(0, 6)
    n_interC[12]=n_interC[11]
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    #5 links
    G.add_edge(1, 7)
    n_interC[13]=n_interC[12]+1
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    #full comem
    #G.add_edge(0, 3)
    G.add_edge(0, 4)
    G.add_edge(0, 5)
    #G.add_edge(0, 6)
    G.add_edge(0, 7)
    G.add_edge(1, 3)
    #G.add_edge(1, 4)
    G.add_edge(1, 5)
    G.add_edge(1, 6)
    #G.add_edge(1, 7)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    #G.add_edge(2, 5)
    G.add_edge(2, 6)
    G.add_edge(2, 7)
    n_interC[14]=n_interC[13]+10
    

    #full comembership


    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    print ["%.4f" % v for v in U_sum]
    print n_interC
    print "%.4f" % max(U_sum), max(n_interC)


    ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### 
    '''
    # Equilibrium Rule Adding Bridges
    
    for i in range(n_cliques):
        for j in range(i):
            if W_interconnection[i][j] >= c/(delta+(clique_sizes[j]-1)*delta**2):
                G.add_edge(clique_sizes_cum[i], clique_sizes_cum[j])
    
    plt.figure(1)
    nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show()
    
    U=payoff(G, W, c, delta)
    U_sum.append(sum(U))

    G=nx.Graph()
    for i in range(nodes):
        G.add_node(i)


    # Equilibrium Rule Adding Redundancy
    
    for i in range(n_cliques):
        for k in range(clique_sizes[i]):
            for l in range(clique_sizes[i]):
                G.add_edge(clique_sizes_cum[i]+k, (clique_sizes_cum[i])+l) 

    for i in range(n_cliques):
        for j in range(i):
            if W_interconnection[i][j] >= c/(delta+(clique_sizes[j]-1)*delta**2):
                G.add_edge(clique_sizes_cum[i], clique_sizes_cum[j])

    for i in range(n_cliques):
        for j in range(i):
            if W_interconnection[i][j] > c/(delta+(clique_sizes[j]-2)*delta**2-(clique_sizes[j]-1)*delta**3):
                G.add_edge(clique_sizes_cum[i]+1, clique_sizes_cum[j]+1)
    
    plt.figure(2)
    nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show()
    
    U=payoff(G,W, c, delta)
    U_sum.append(sum(U))

    
    plt.figure(2)
    print U_sum
    plt.plot(U_sum, linewidth=2)
    plt.show()
    '''
    #return G, U_sum
    return G, sum(U)

#G = Formation(6)


import networkx as nx
import collections
import math
import numpy as np
import matplotlib.pyplot as plt

def payoff_coauthor(G,W, c, delta):


    A= nx.to_numpy_matrix(G)
    A = np.array(A)
    #print A
    for i in range(len(A)):
        A[i][i]=0
    row_sum_A=A.sum(axis=1)


    U=[]
    P=[]
    U_updated=[]
    node_number=int(math.sqrt(np.size(A)))

    A1=np.multiply(W,A)
    row_sum=A1.sum(axis=1)
    nodes=np.zeros(6)

    weights=[]
    for (i, j) in G.edges():
        weights.append(W[int(i)][int(j)])

    weights=np.multiply(weights, 1)

    GW=nx.Graph()
    for i in range(node_number):
        GW.add_node(i)
    for e in G.edges():
        #print e[0], e[1], W[int(e[1])][int(e[0])]
        GW.add_edge(e[1], e[0] ,weight=W[int(e[1])][int(e[0])])

    '''
    plt.figure(1)
    nx.draw(G, pos=nx.shell_layout(G), weight=weights, width=weights, color= weights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show()
    '''
    #print A
    for i in range(node_number):
            y=0
            x=0
            Neighbors=[]
            Neighbors=G.neighbors(i)
            #print Neighbors
            for x in Neighbors:
                #print "x, y ", x, y
                if x!=i:
                    y=y+(1/row_sum_A[i])  +  (1/row_sum_A[int(x)]) + 1/(row_sum_A[i]*row_sum_A[int(x)]) 
                #print "y ",y 
            y=round(y,5)
            U.append(y) 
            #U.append(length1path*delta + length2path*(delta**2) - length1path*c)  

    #np.around(U, 5)
    #print "U= ", np.around(U, 5)
    #np.set_printoptions(precision=5)
    return U

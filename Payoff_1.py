import networkx as nx
import collections
import math
import numpy as np
import matplotlib.pyplot as plt

def payoff_1(G,W):

    '''

    Redundant Ties
    c=0.2, delta=0.7

    W_interconnection.append([0   , 0.7 , 0.1 , 0.1 ])
    W_interconnection.append([0.7 , 0   , 0.5, 0.5])
    W_interconnection.append([0.1 , 0.5, 0   , 0   ])
    W_interconnection.append([0.1 , 0.5, 0   , 0   ])


    Bridging Ties
    c=0.2, delta=0.5

    W_interconnection.append([0   , 0.3 , 0.1 , 0.1 ])
    W_interconnection.append([0.3 , 0   , 0.3, 0.3])
    W_interconnection.append([0.1 , 0.3, 0   , 0   ])
    W_interconnection.append([0.1 , 0.3, 0   , 0   ])

    '''
    c=0.2

    A= nx.to_numpy_matrix(G)
    A = np.array(A)

    delta=0.5
    U=[]
    U_updated=[]
    length1path=0
    length2path=0
    length3path=0
    node_number=int(math.sqrt(np.size(A)))

    A1=np.multiply(W,A)
    row_sum=A1.sum(axis=1)

    #print A
    for i in range(node_number):

            length1path=0
            length2path=0
            length3path=0

            #A=np.multiply(A1,W)
            for j in range(node_number):
                try:
                    if len([p for p in nx.all_shortest_paths(G, source=i, target=j)][0])-1 == 2:
                        x = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][0])
                        y = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][1])
                        z = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][2])
                        length2path=length2path+W[x][y]*A[x][y]*W[y][z]*A[y][z]
                except nx.NetworkXNoPath:
                    pass
                    #print 'No path'
                
                try:
                    if len([p for p in nx.all_shortest_paths(G, source=i, target=j)][0])-1 == 3:
                        x = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][0])
                        y = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][1])
                        z = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][2])
                        w = int([p for p in nx.all_shortest_paths(G, source=i, target=j)][0][3])
                        length3path=length3path+A[x][y]*A[y][z]*A[z][w]*W[x][y]*W[y][z]*W[z][w]
                except nx.NetworkXNoPath:
                    pass
                
                
            length1path=row_sum[i]
            node_degree=G.degree(i)

            #print "length1path*delta ", length1path*delta
            #print "length2path*(delta**2) ", length2path*(delta**2) 
            #print "length3path*(delta**3)", length3path*(delta**3)

            #print "inter_c, inter_c_unweighted, length2_value", inter_c, inter_c_unweighted, length2_value
            U.append(length1path*delta + length2path*(delta**2) + length3path*(delta**3) - node_degree*c) 
            #U.append(length1path*delta + length2path*(delta**2) - length1path*c)  

            length1path=0
            length2path=0
            length2_value=0

    #np.around(U, 5)
    #print "U= ", np.around(U, 5)
    #np.set_printoptions(precision=5)
    return U

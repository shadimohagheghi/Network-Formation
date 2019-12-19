import networkx as nx
import collections
import math
import numpy as np

def payoff(G,W, c, delta):


    A= nx.to_numpy_matrix(G)
    A = np.array(A)
    #print A

    for i in range(len(A)):
        A[i][i]=0

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

            length1path=0
            length2path=0
            length3path=0
            length4path=0
            length5path=0


            #A=np.multiply(A1,W)
            for j in range(node_number):
                try:
                    if len([p for p in nx.all_shortest_paths(GW, source=i, target=j, weight=True)][0])-1 == 2:
                        for x in range(3):
                            nodes[x] = int([p for p in nx.all_shortest_paths(GW, source=i, target=j)][0][x])
                        #length2path=length2path+W[int(nodes[0])][int(nodes[1])]*A[int(nodes[0])][int(nodes[1])]*W[int(nodes[1])][int(nodes[2])]*A[int(nodes[1])][int(nodes[2])]
                        length2path=length2path+W[int(nodes[0])][int(nodes[2])]*A[int(nodes[0])][int(nodes[1])]*A[int(nodes[1])][int(nodes[2])]
                except nx.NetworkXNoPath:
                    pass
                    #print 'No path'
                
                try:
                    if len([p for p in nx.all_shortest_paths(GW, source=i, target=j, weight=True)][0])-1 == 3:
                        for x in range(4):
                            nodes[x] = int([p for p in nx.all_shortest_paths(GW, source=i, target=j)][0][x])
                        #length3path=length3path+W[int(nodes[0])][int(nodes[1])]*A[int(nodes[0])][int(nodes[1])]*W[int(nodes[1])][int(nodes[2])]*A[int(nodes[1])][int(nodes[2])]* \
                        #                        W[int(nodes[2])][int(nodes[3])]*A[int(nodes[2])][int(nodes[3])]
                        length3path=length3path+W[int(nodes[0])][int(nodes[3])]*A[int(nodes[0])][int(nodes[1])]*A[int(nodes[1])][int(nodes[2])]*A[int(nodes[2])][int(nodes[3])]
                except nx.NetworkXNoPath:
                    pass
                
                try:
                    if len([p for p in nx.all_shortest_paths(GW, source=i, target=j, weight=True)][0])-1 == 4:
                        for x in range(5):
                            nodes[x] = int([p for p in nx.all_shortest_paths(GW, source=i, target=j)][0][x])
                        #length4path=length4path+W[int(nodes[0])][int(nodes[1])]*A[int(nodes[0])][int(nodes[1])]*W[int(nodes[1])][int(nodes[2])]*A[int(nodes[1])][int(nodes[2])]* \
                        #                        W[int(nodes[2])][int(nodes[3])]*A[int(nodes[2])][int(nodes[3])]*W[int(nodes[3])][int(nodes[4])]*A[int(nodes[3])][int(nodes[4])]
                        length4path=length4path+W[int(nodes[0])][int(nodes[4])]*A[int(nodes[0])][int(nodes[1])]*A[int(nodes[1])][int(nodes[2])]* \
                                                A[int(nodes[2])][int(nodes[3])]*A[int(nodes[3])][int(nodes[4])]
                except nx.NetworkXNoPath:
                    pass
                
                try:
                    if len([p for p in nx.all_shortest_paths(GW, source=i, target=j, weight=True)][0])-1 == 5:
                        for x in range(6):
                            nodes[x] = int([p for p in nx.all_shortest_paths(GW, source=i, target=j)][0][x])
                        #length5path=length5path+W[int(nodes[0])][int(nodes[1])]*A[int(nodes[0])][int(nodes[1])]*W[int(nodes[1])][int(nodes[2])]*A[int(nodes[1])][int(nodes[2])]* \
                        #                        W[int(nodes[2])][int(nodes[3])]*A[int(nodes[2])][int(nodes[3])]*W[int(nodes[3])][int(nodes[4])]*A[int(nodes[3])][int(nodes[4])]*\
                        #                        W[int(nodes[4])][int(nodes[5])]*A[int(nodes[4])][int(nodes[5])]
                        length5path=length5path+W[int(nodes[0])][int(nodes[5])]*A[int(nodes[0])][int(nodes[1])]*A[int(nodes[1])][int(nodes[2])]* \
                                                A[int(nodes[2])][int(nodes[3])]*A[int(nodes[3])][int(nodes[4])]*A[int(nodes[4])][int(nodes[5])]

                except nx.NetworkXNoPath:
                    pass
                
            length1path=row_sum[i]
            node_degree=G.degree(i)

            row_sum_A=A.sum(axis=1)
            node_degree=row_sum_A[i]
            '''
            print "length1path ", length1path
            print "length2path ", length2path 
            print "length3path ", length3path
            print "node_degree ", node_degree
            '''
            #print "inter_c, inter_c_unweighted, length2_value", inter_c, inter_c_unweighted, length2_value
            #P.append(length1path*delta + length2path*(delta**2) + length3path*(delta**3) + length4path*(delta**4) + length5path*(delta**5) - node_degree*c)
            x=length1path*delta + length2path*(delta**2) + length3path*(delta**3) + length4path*(delta**4) + length5path*(delta**5)- node_degree*c
            x=round(x,5)
            U.append(x) 
            #U.append(length1path*delta + length2path*(delta**2) - length1path*c)  

    #np.around(U, 5)
    #print "U= ", np.around(U, 5)
    #np.set_printoptions(precision=5)
    return U

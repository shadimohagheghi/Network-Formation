import networkx as nx
from numpy import linalg as LA
import math
import collections
import numpy as np 
import matplotlib
from payoff import payoff
#from payoff_coauthor import payoff_coauthor
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
plt.close("all")

##########################################################
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))

def Formation(clique_sizes, W_interconnection, Iterations , c, delta) :
    Iterations =Iterations
    n=np.sum(clique_sizes)

    one_list=np.ones(n)

    clique_sizes_cum=np.zeros(len(clique_sizes))
    clique_sizes_cum[0]=0
    for i in range(1, len(clique_sizes)):
        clique_sizes_cum[i]=clique_sizes_cum[i-1]+clique_sizes[i-1]

    print "clique_sizes_cum: ", clique_sizes_cum

    nodes_by_clique=[]
    '''
    for i in range(len(clique_sizes)):
        for nodes in range(clique_sizes[i]):
            nodes_by_clique.append((nodes))
    '''
    nodes_by_clique = [[0 for j in range(clique_sizes[i])] for i in range(len(clique_sizes))]
    for i in range(len(nodes_by_clique)):
        print i
        for j in range(clique_sizes[i]):
            nodes_by_clique[i][j]=int(clique_sizes_cum[i]+j)


    print "nodes_by_clique ", nodes_by_clique
    #nodes_by_clique = [[0 for j in range(clique_sizes[i])] for i in range(len(clique_sizes))]
    
    print W_interconnection
    n_cliques=int(len(W_interconnection))
    W=np.zeros((n, n))

    #Comembership
    
    for i in range(n_cliques):
        i = int(i)
        for j in range(n_cliques):
            j= int(j)
            for k in range(clique_sizes[i]):
                k=int(k)
                for l in range(clique_sizes[j]):
                    W[int(clique_sizes_cum[i]+k)][int(clique_sizes_cum[j]+l)]=W_interconnection[i][j]

    for i in range(n):
        W[i][i]=0


    [v,w]=LA.eig(np.transpose(W))


    ind = np.unravel_index(np.argmax(v, axis=None), v.shape)
    pi = [row[np.argmax(v)] for row in w]
    pi=np.divide(pi, sum(pi))


    G=nx.Graph()
    
    for i in range(n):
        G.add_node(i)
    '''
    #G=nx.complete_graph(n)
    G.add_edge(0,3)
    G.add_edge(0,4)
    G.add_edge(0,5)
    G.add_edge(0,6)
    G.add_edge(0,7)
    G.add_edge(1,3)
    G.add_edge(1,4)
    G.add_edge(1,5)
    G.add_edge(1,6)
    G.add_edge(1,7)
    G.add_edge(2,3)
    G.add_edge(2,4)
    G.add_edge(2,5)
    G.add_edge(2,6)
    G.add_edge(2,7)
    '''
    '''
    for i in range(n_cliques):
        for k in range(clique_sizes[i]):
            for l in range(clique_sizes[i]):
                G.add_edge(clique_sizes_cum[i]+k, (clique_sizes_cum[i])+l)
    '''
    ### Simulation Resaults Regarding Efficiency:
    
    ##### Star with the same node from central clique: [3,3,4,4,5]
    '''
    W_interconnection=[[1    ,  0.21 , 0.21 , 0.21 , 0.21], \
                   [0.21 , 1     , 0.09 , 0.09 , 0.09], \
                   [0.21 , 0.09  , 1    , 0.09 , 0.09], \
                   [0.21 , 0.09  , 0.09 , 1    , 0.09], \
                   [0.21 , 0.09  , 0.09 , 0.09 , 1   ]]
    for i in range(n_cliques):
        G.add_edge(0, (clique_sizes_cum[i]))  #20.22135, 20.22135, 20.92616, 4th center: 20.92616 , 5th center: 21.537
    
    ########## Line: [3,3,4,4,5]
    ### 5th center:
    ###     same node from this clique: 20.87632, different node from this clique only: 20.79175, different nodes from all cliques: 20.65239
    G.add_edge(14, 6)
    G.add_edge(15, 11)
    G.add_edge(3, 10)
    G.add_edge(0, 7)   
    
    ### 3rd center, same node from this clique: 20.38982
    W_interconnection=[[1    , 0.21  , 0.09 , 0.09 , 0.09  ], \
                   [0.21 , 1     , 0.21 , 0.09 , 0.09], \
                   [0.09 , 0.21  , 1    , 0.21 , 0.09], \
                   [0.09 , 0.09  , 0.21 , 1    , 0.21], \
                   [0.09 , 0.09  , 0.09 , 0.21 , 1   ]]
    G.add_edge(6 , 3)
    G.add_edge(6 , 10)
    G.add_edge(10, 14)
    G.add_edge(0 , 3)
    
    ########## [3,3,4,4,5]: 
    ### 5th center:
    ###     (1,5), (2,5), (3,4), (4,5): same node from this clique: 21.10471

    ########## [3,3,4,4,5]: 
    ### 5th center:
    ###     (1,5), (1,2), (3,5), (4,5): same node from this clique: 20.96375   !!!

    G.add_edge(10, 14)
    G.add_edge(6, 14)
    G.add_edge(3, 0)
    G.add_edge(0, 14)   
    
    

    ### [19]: 102.6 

    ### 34.57399
    G.add_edge(0, 3)
    G.add_edge(3, 9)
    G.add_edge(6, 13)
    G.add_edge(9, 17)
    G.add_edge(13, 22)
    G.add_edge(22, 17)   
    
    
    plt.figure(0)
    nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show()
    '''

    allWeights=[]

    U_sum=[]
    P=[]
    U_updated_list=[]
    n_iterations=[]



    U=payoff(G, W, c, delta)
    #print U
    #print(sum(U))
    '''
    perm=[]
    list1=np.arange(n)
    list2=np.arange(n)
    for i in range(n):
        for j in range(i):
            perm.append([i,j])

    U_all=np.zeros(Iterations, n)
    U_expectedV=[]
    '''
    ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### *** ### 

    U=[]
    U_updated=[]
    P_updated=[]
    potential_f=[]
    U=payoff(G, W, c, delta)
    U_updated=payoff(G, W, c, delta)
    i_updated=0
    j_updated=0

    n_inter_c= np.zeros(Iterations+1)
    n_intra_c= np.zeros(Iterations+1)


    interconnections_num=0
    for u in range(Iterations): 
        
        [i, j]= np.random.choice(n, 2, replace=False)

        #print "[i,j] ", [i,j]

        if G.has_edge(i,j):
            G.remove_edge(i,j)
            U_updated=payoff(G, W, c, delta)
            if (U_updated[i]>U[i] or U_updated[j]>U[j]) :
                '''
                if U_updated[i]>U[i]: 
                    print "U_updated[i], U[i]", U_updated[i], U[i]
                if U_updated[j]>U[j]:
                    print "U_updated[j], U[j]", U_updated[j], U[j]
                '''
                i_updated=i
                j_updated=j
                #print "removed"
                for q in range(len(U_updated)):
                    U[q]=U_updated[q]
                interconnections_num=interconnections_num-1
                '''
                plt.figure()
                nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
                plt.show()
                '''
            elif (U_updated[i]<=U[i] and U_updated[j]<=U[j]) :
                G.add_edge(i,j)
                #print "didn't remove"
            
        elif not(G.has_edge(i,j)):
            G.add_edge(i,j)
            U_updated=payoff(G, W, c, delta)

            if ((U_updated[i]>U[i] and U_updated[j]>=U[j]) or (U_updated[i]>=U[i] and U_updated[j]>U[j])):  
                            
                i_updated=i
                j_updated=j 
                #print "added"
                #print "i, j ", i, j
                print index_2d(nodes_by_clique, i)
                print index_2d(nodes_by_clique, j)

                if index_2d(nodes_by_clique, i)[0]==index_2d(nodes_by_clique, j)[0]:
                    n_intra_c[u]=n_intra_c[u]+1
                else:
                    n_inter_c[u]=n_inter_c[u]+1


                for q in range(len(U_updated)):
                    U[q]=U_updated[q]
                interconnections_num=interconnections_num+1
                #plt.figure(1)
                #nx.draw(G, pos=nx.shell_layout(G), weight=weights, width=weights, color= weights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
                #plt.show()
            else:
                G.remove_edge(i,j)
                #print "didn't add"
        du=0
        for agent in range(n):
            for neighbor in G.neighbors(agent):
                du=du+(U[int(neighbor)]*W[int(agent)][int(neighbor)])
        #print "du", du


        #print "U=", U #, np.around(U, 5)
        potential_f.append(du)#sum(np.dot(U, W))) #np.dot(U, pi))#W.sum(axis=1)))  # U[int(i_updated)]+U[int(j_updated)]) #np.dot(U, W)

        if potential_f[u]<potential_f[u-1]:
                print "\n \n Potential decresing \n "

        U_sum.append(sum(U))
        
        if U_sum[u]<U_sum[u-1]:
            
            #plt.figure(1)
            #nx.draw(G, pos=nx.shell_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
            #plt.show()
            
            print "\n \n U_sum decresing \n \n "

        
            n_iterations.append(u)
            weights=[]
        
            GW = nx.from_numpy_matrix(W, create_using=nx.MultiGraph())

            for (i, j) in G.edges():
                weights.append(W[int(i)][int(j)])

            weights=np.multiply(weights, 1)
            allWeights=np.multiply(allWeights, 10)

        '''
        n_iterations.append(u)
        weights=[]
        
        GW = nx.from_numpy_matrix(W, create_using=nx.MultiGraph())

        for (i, j) in G.edges():
            weights.append(W[int(i)][int(j)])

        weights=np.multiply(weights, 1)
        allWeights=np.multiply(allWeights, 10)

        plt.figure(1)
        nx.draw(G, pos=nx.shell_layout(G), weight=weights, width=weights, color= weights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
        plt.show()
        '''
        n_intra_c[u+1]=n_intra_c[u]
        n_inter_c[u+1]=n_inter_c[u]

    U_updated=payoff(G, W, c, delta)
    #print "sum(U), sum(U_updated): ", sum(U), sum(U_updated)

    #print W
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
        allWeights.append(W[int(i)][int(j)])

    for (i, j) in G.edges():
        weights.append(W[int(i)][int(j)])

    #partition = best_partition(G) 

    #print "sum(U)", sum(U)
    weights=np.multiply(weights, 1)
    allWeights=np.multiply(allWeights, 10)
    
    #plt.figure(2)
    #nx.draw(G, pos=nx.shell_layout(G),  color= weights, node_size=500,cmap=plt.cm.Reds, with_labels = True)

    #plt.figure()
    #plt.plot(potential_f, linewidth=2, color='r')
    #plt.plot(U_sum, linewidth=2)
    #plt.show()
    '''
    plt.figure(2)
    #print U_sum
    plt.plot(n_iterations,U_sum, linewidth=2)
    plt.show()
    '''
    '''
    plt.figure(2)
    print GW.edges()
    nx.draw(GW, pos=nx.shell_layout(GW), weight=allWeights, width=allWeights, color= allWeights, node_size=500,cmap=plt.cm.Reds, with_labels = True)
    #nx.draw(G, pos=nx.circular_layout(G), node_size=500,cmap=plt.cm.Reds, with_labels = True)
    plt.show() 
    '''
    #print G.edges
    perm=[]
    
    for i in range(n):
        for j in range(i):
            perm.append([i,j])
    for x in perm:
        print x
        if G.has_edge(x[0], x[1]):
            G.remove_edge(x[0], x[1])
            U_updated=payoff(G, W, c, delta)
            if (U_updated[x[0]]>U[x[0]] or U_updated[x[1]]>U[x[1]]) :
                print "not stable, removed"
                for q in range(len(U_updated)):
                    U[q]=U_updated[q]

            elif (U_updated[x[0]]<=U[x[0]] and U_updated[x[1]]<=U[x[1]]) :
                G.add_edge(x[0], x[1])
            
        elif not(G.has_edge(x[0], x[1])):
            G.add_edge(x[0], x[1])
            U_updated=payoff(G, W, c, delta)

            if ((U_updated[x[0]]>U[x[0]] and U_updated[x[1]]>=U[x[1]]) or (U_updated[x[0]]>=U[x[0]] and U_updated[x[1]]>U[x[1]])):  
                            
                print "not stable, added"
                for q in range(len(U_updated)):
                    U[q]=U_updated[q]
            else:
                G.remove_edge(x[0], x[1])

    
    #print "sum(U)", sum(U) 
    with open('values2x2_new.txt', "a") as text_file:
        text_file.write(str('%.3f' % W_interconnection[0][1])+'\t'+str('%.4f' % sum(U)) + "\n")
    with open('interconnections_num2x2_new.txt', "a") as text_file:
        text_file.write(str('%.3f' % W_interconnection[0][1])+'\t'+str('%.2f' % interconnections_num) + "\n")
    '''
    with open('values2x2.txt', "a") as text_file:
        text_file.write(str('%.2f' % W_interconnection[0][1])+'\t'+str('%.2f' % sum(U)) + "\n")
    with open('interconnections_num2x2.txt', "a") as text_file:
        text_file.write(str('%.2f' % W_interconnection[0][1])+'\t'+str('%.2f' % interconnections_num) + "\n")

    with open('values3x3.txt', "a") as text_file:
        text_file.write(str(sum(U)) + "\n")
    with open('interconnections_num3x3.txt', "a") as text_file:
        text_file.write(str(sum(U)) + "\n")

    with open('values3x3.txt', "a") as text_file:
        text_file.write(str(sum(U)) + "\n")
    with open('interconnections_num3x3.txt', "a") as text_file:
        text_file.write(str(sum(U)) + "\n")

    
 
    with open('values2x2_varyingSize.txt', "a") as text_file:
        text_file.write(str('%.2f' % clique_sizes[0])+'\t'+str('%.2f' % clique_sizes[1])+'\t'+str('%.2f' % sum(U)) + "\n")
    with open('interconnections_num2x2_varyingSize.txt', "a") as text_file:
        text_file.write(str('%.2f' % clique_sizes[0])+'\t'+str('%.2f' % clique_sizes[1])+'\t'+str('%.2f' % interconnections_num) + "\n")
    
    
    with open('values2x2_varyingSize_20.txt', "a") as text_file:
        text_file.write(str('%.2f' % clique_sizes[0])+'\t'+str('%.2f' % clique_sizes[1])+'\t'+str('%.2f' % sum(U)) + "\n")
    with open('interconnections_num2x2_varyingSize_20.txt', "a") as text_file:
        text_file.write(str('%.2f' % clique_sizes[0])+'\t'+str('%.2f' % clique_sizes[1])+'\t'+str('%.2f' % interconnections_num) + "\n")

    
    print "n_intra_c", n_intra_c
    print "n_inter_c", n_inter_c
    np.savetxt('intra.txt',n_intra_c, fmt='%10.f')
    np.savetxt('inter.txt',n_inter_c, fmt='%10.f')

    plt.figure()
    plt.plot(n_intra_c, linewidth=2, color='b')
    plt.plot(n_inter_c, linewidth=2, color='r')
    plt.show()
    '''
    #return G, U_sum
    return G, sum(U)

#G = Formation(6)


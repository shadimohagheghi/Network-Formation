import networkx as nx
import collections
import numpy as np
import matplotlib.pyplot as plt

def almost_clique(clique_size,clique_num,rep_num):

    #print "clique_size= ", clique_size
    #print "clique_num= ", clique_num

    Gclique = nx.Graph()
    reps = []
    reps_cliquewise = []
    clique_sizes = []
    rep_numbers_cliquewise = []
    reps_ordered_cliquewise = []
    clique_start = []

    cliques = 0
    nodes = 0; 

    for k in range(len(clique_size)):
        i = clique_size[k]
        for j in range(clique_num[k]):
            clique_start.append(nodes)
            cliques = cliques + 1
            Gstn1 = nx.erdos_renyi_graph(i,1)
            comps = nx.number_connected_components(Gstn1)
            while comps > 1:
                #Gstn1.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
                #Gstn1.add_weighted_edges_from([(0,1,1.0),(1,2,1.0),(0,2,1.0)])
                Gstn1 = nx.erdos_renyi_graph(i,1)
                comps = nx.number_connected_components(Gstn1)

            Gclique = nx.disjoint_union(Gclique,Gstn1)
            
            # selecting nodes from the group
            rep_tot = rep_num(clique_size[k])
            rep_numbers_cliquewise.append(rep_tot)
            clique_sizes.append(i)
            x = collections.Counter(nx.degree(Gstn1)).most_common(rep_tot)
            x=np.asarray(x); x = np.transpose(x)
    
            thisrep = x[0]
            
            rep_list_this = thisrep + nodes        
            reps = np.hstack((reps, rep_list_this))
            reps_cliquewise.append(rep_list_this)        
            nodes = nodes + i  
            deg = np.zeros(i)
            for p in range(i):
                deg[p] = len(Gstn1[p])
            degord = sorted(range(len(deg)),key=lambda x:deg[x],reverse=True) 
            reps_ordered_cliquewise.append(degord)

    if nx.number_connected_components(Gclique) != cliques:
        print 'cliques not formed correctly'

  
    print 'reps = ', reps
    print 'reps_ordered_cliquewise = ', reps_ordered_cliquewise
    print 'clique_start = ', clique_start

    return Gclique, reps, reps_cliquewise, clique_sizes, rep_numbers_cliquewise, reps_ordered_cliquewise, clique_start
    
    
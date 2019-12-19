import networkx as nx
import numpy as np 
import matplotlib
import dynetx as dn
import matplotlib.pyplot as plt
plt.close("all")
'''
plt.figure()
G= nx.Graph()
G.add_edge('a','b',timestamp='t1')
G.add_node('c',timestamp='t1')

G.add_edge('a','b',timestamp='t2')
G.add_edge('a','c',timestamp='t2')
G.add_edge('b','c',timestamp='t2')

'''
plt.figure()
G = dn.DynGraph(edge_removal=True)
G.add_path([0,1,2], 0)
dn.nodes(G, t=0)
G.add_edge(2,3, t=1)
dn.interactions(G, t=0)

#G.add_interaction(u=1, v=2, t=0, e=3)
nx.draw(G)
plt.show()


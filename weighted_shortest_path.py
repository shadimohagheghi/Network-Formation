import networkx as nx

G = nx.Graph()
e = [('a', 'b', 2), ('a', 'c', 6), ('b', 'c', 4)]

for i in e:
    G.add_edge(i[1], i[0], weight=i[2])

#print weight
paths = nx.all_shortest_paths(G, source='a', target='c',weight=True)

print list(paths)
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = [(10, 11),(11, 12),(12, 13), (13, 15),(15,10)]

fig = plt.figure()

G = nx.Graph()
G.add_edges_from(g)
G.add_nodes_from(G)

pos = nx.shell_layout(G)
edges = nx.draw_networkx_edges(G, pos, edge_color = 'w')
nodes = nx.draw_networkx_nodes(G, pos, node_color = 'g')

white = (1,1,1,1)
black = (0,0,0,1)

colors = [white for edge in edges.get_segments()]

def update(n):
    global colors

    try:
        idx = colors.index(white)
        colors[idx] = black
    except ValueError:
        colors = [white for edge in edges.get_segments()]

    edges.set_color(colors)
    return edges, nodes

anim = FuncAnimation(fig, update, interval=150, blit = False) 

plt.show()
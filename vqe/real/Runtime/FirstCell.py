import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Generating a graph of 5 nodes

n = 12  # Number of nodes in graph
G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))
elist = [

    (0, 1, 1.19),

    (0, 2, 1.0),

    (0, 3, 1.0),

    (0, 4, 1.19),

    (0, 5, 1.19),

    (0, 6, 1.19),

    (0, 7, 1.0),

    (0, 8, 1.19),

    (0, 9, 1.19),

    (0, 10, 1.19),

    (0, 11, 1.0),

    (1, 0, 1.19),

    (1, 4, 1.19),

    (1, 5, 1.19),

    (1, 6, 1.19),

    (1, 8, 1.19),

    (1, 9, 1.19),

    (1, 10, 1.19),

    (2, 0, 1.0),

    (2, 7, 1.0),

    (3, 0, 1.0),

    (3, 11, 1.0),

    (4, 0, 1.19),

    (4, 1, 1.19),

    (5, 0, 1.19),

    (5, 1, 1.19),

    (6, 0, 1.19),

    (6, 1, 1.19),

    (7, 0, 1.0),

    (7, 2, 1.0),

    (8, 0, 1.19),

    (8, 1, 1.19),

    (9, 0, 1.19),

    (9, 1, 1.19),

    (10, 0, 1.19),

    (10, 1, 1.19),

    (11, 0, 1.0),

    (11, 3, 1.0)]

# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(elist)

colors = ['r' for node in G.nodes()]
pos = nx.spring_layout(G)


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600,
                     alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


draw_graph(G, colors, pos)

# Computing the weight matrix from the random graph
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = temp['weight']
print(w)

best_cost_brute = 0
for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + w[i, j]*x[i]*(1-x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x
    print('case = ' + str(x) + ' cost = ' + str(cost))

colors = ['r' if xbest_brute[i] == 0 else 'c' for i in range(n)]
draw_graph(G, colors, pos)
print('\nBest solution = ' + str(xbest_brute) +
      ' cost = ' + str(best_cost_brute))

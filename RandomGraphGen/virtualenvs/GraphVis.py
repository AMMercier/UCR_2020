import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_network(adj, n, name, sp='o', pos=None):
    G = nx.to_networkx_graph(adj)
    if pos is None:
        pos = nx.spring_layout(G, scale=100)
    fig = plt.figure(figsize=(8, 8))
    weights = nx.get_edge_attributes(G, 'weight')
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}
    # ntwrk = nx.to_networkx_graph(sorted(G.edges(data=True), key=lambda x: x[2]['weight']))
    n1 = list(range(int(np.ceil(.25 * len(weights)))))
    n2 = list(range(int(np.ceil(.25 * len(weights))), int(np.ceil(.5 * len(weights)))))
    n3 = list(range(int(np.ceil(.5 * len(weights))), int(np.ceil(.75 * len(weights)))))
    n4 = list(range(int(np.ceil(.5 * len(weights))), len(weights)))
    l = [n1, n2, n3, n4]
    # colors = ['mediumblue', 'dodgerblue', 'tomato', 'red2']
    cmap1 = mpl.cm.Blues(np.linspace(0, 1, 20))
    cmap1 = mpl.colors.ListedColormap(cmap1[10:, :-1])
    cmap2 = mpl.cm.Purples(np.linspace(0, 1, 20))
    cmap2 = mpl.colors.ListedColormap(cmap2[10:, :-1])
    cmap3 = mpl.cm.Oranges(np.linspace(0, 1, 20))
    cmap3 = mpl.colors.ListedColormap(cmap3[10:, :-1])
    cmap4 = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap4 = mpl.colors.ListedColormap(cmap4[10:, :-1])
    edges = list(weights.keys())
    E = []
    for i in range(4):
        e = list(edges[j] for j in l[i])
        E.append(e)
    G1 = G.edge_subgraph(E[0])
    G2 = G.edge_subgraph(E[1])
    G3 = G.edge_subgraph(E[2])
    G4 = G.edge_subgraph(E[3])
    nx.draw_networkx_edges(G1, pos,
                           width=1.15,
                           alpha=0.7,
                           edge_color=l[0],
                           edge_cmap=cmap1)
    nx.draw_networkx_edges(G2, pos,
                           width=1.1,
                           alpha=0.6,
                           edge_color=l[1],
                           edge_cmap=cmap2)
    nx.draw_networkx_edges(G3, pos,
                           width=1,
                           alpha=.55,
                           edge_color=l[2],
                           edge_cmap=cmap3)
    nx.draw_networkx_edges(G4, pos,
                           width=0.75,
                           alpha=.5,
                           edge_color=l[3],
                           edge_cmap=cmap4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=20,
                           node_color='black',  # list(range(len(adj))),
                           cmap=plt.cm.cividis)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.axis('off')
    plt.savefig(name, transparent=True)
    plt.show()
    return pos


def draw_ran_geo(pos, adj, name):
    G = nx.to_networkx_graph(adj)
    fig = plt.figure(figsize=(8, 8))
    weights = nx.get_edge_attributes(G, 'weight')
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}
    n1 = list(range(int(np.ceil(.25 * len(weights)))))
    n2 = list(range(int(np.ceil(.25 * len(weights))), int(np.ceil(.5 * len(weights)))))
    n3 = list(range(int(np.ceil(.5 * len(weights))), int(np.ceil(.75 * len(weights)))))
    n4 = list(range(int(np.ceil(.5 * len(weights))), len(weights)))
    l = [n1, n2, n3, n4]
    cmap1 = mpl.cm.Blues(np.linspace(0, 1, 20))
    cmap1 = mpl.colors.ListedColormap(cmap1[10:, :-1])
    cmap2 = mpl.cm.Purples(np.linspace(0, 1, 20))
    cmap2 = mpl.colors.ListedColormap(cmap2[10:, :-1])
    cmap3 = mpl.cm.Oranges(np.linspace(0, 1, 20))
    cmap3 = mpl.colors.ListedColormap(cmap3[10:, :-1])
    cmap4 = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap4 = mpl.colors.ListedColormap(cmap4[10:, :-1])
    edges = list(weights.keys())
    E = []
    for i in range(4):
        e = list(edges[j] for j in l[i])
        E.append(e)
    G1 = G.edge_subgraph(E[0])
    G2 = G.edge_subgraph(E[1])
    G3 = G.edge_subgraph(E[2])
    G4 = G.edge_subgraph(E[3])
    nx.draw_networkx_edges(G1, pos,
                           width=1.15,
                           alpha=1,
                           edge_color=l[0],
                           edge_cmap=cmap1)
    nx.draw_networkx_edges(G2, pos,
                           width=1.1,
                           alpha=1,
                           edge_color=l[1],
                           edge_cmap=cmap2)
    nx.draw_networkx_edges(G3, pos,
                           width=1,
                           alpha=1,
                           edge_color=l[2],
                           edge_cmap=cmap3)
    nx.draw_networkx_edges(G4, pos,
                           width=0.75,
                           alpha=1,
                           edge_color=l[3],
                           edge_cmap=cmap4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=20,
                           node_color='black',  # list(range(len(adj))),
                           cmap=plt.cm.cividis)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.savefig(name, transparent=True)
    plt.show()


def draw_ran_latt(pos, adj, name):
    G = nx.to_networkx_graph(adj)
    fig = plt.figure(figsize=(8, 8))
    weights = nx.get_edge_attributes(G, 'weight')
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}
    # ntwrk = nx.to_networkx_graph(sorted(G.edges(data=True), key=lambda x: x[2]['weight']))
    n1 = list(range(int(np.ceil(.25 * len(weights)))))
    n2 = list(range(int(np.ceil(.25 * len(weights))), int(np.ceil(.5 * len(weights)))))
    n3 = list(range(int(np.ceil(.5 * len(weights))), int(np.ceil(.75 * len(weights)))))
    n4 = list(range(int(np.ceil(.5 * len(weights))), len(weights)))
    l = [n1, n2, n3, n4]
    # colors = ['mediumblue', 'dodgerblue', 'tomato', 'red2']
    cmap1 = mpl.cm.Blues(np.linspace(0, 1, 20))
    cmap1 = mpl.colors.ListedColormap(cmap1[10:, :-1])
    cmap2 = mpl.cm.Purples(np.linspace(0, 1, 20))
    cmap2 = mpl.colors.ListedColormap(cmap2[10:, :-1])
    cmap3 = mpl.cm.Oranges(np.linspace(0, 1, 20))
    cmap3 = mpl.colors.ListedColormap(cmap3[10:, :-1])
    cmap4 = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap4 = mpl.colors.ListedColormap(cmap4[10:, :-1])
    edges = list(weights.keys())
    E = []
    for i in range(4):
        e = list(edges[j] for j in l[i])
        E.append(e)
    G1 = G.edge_subgraph(E[0])
    G2 = G.edge_subgraph(E[1])
    G3 = G.edge_subgraph(E[2])
    G4 = G.edge_subgraph(E[3])
    nx.draw_networkx_edges(G1, pos,
                           width=1.15,
                           alpha=1,
                           edge_color=l[0],
                           edge_cmap=cmap1)
    nx.draw_networkx_edges(G2, pos,
                           width=1.1,
                           alpha=1,
                           edge_color=l[1],
                           edge_cmap=cmap2)
    nx.draw_networkx_edges(G3, pos,
                           width=1,
                           alpha=1,
                           edge_color=l[2],
                           edge_cmap=cmap3)
    nx.draw_networkx_edges(G4, pos,
                           width=0.75,
                           alpha=1,
                           edge_color=l[3],
                           edge_cmap=cmap4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=20,
                           node_color='black',  # list(range(len(adj))),
                           cmap=plt.cm.cividis)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.axis('off')
    plt.savefig(name, transparent=True)
    plt.show()

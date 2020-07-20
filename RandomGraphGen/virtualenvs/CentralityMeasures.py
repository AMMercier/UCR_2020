import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_Sparse import Spl_EffRSparse


def hamming_distance(string1, string2):
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter


def EigenVecCen(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    e_G = nx.eigenvector_centrality(nx.to_networkx_graph(adj), max_iter=10000, weight="weight")
    e_G = np.array(list(e_G.values()))
    e_G = list(zip(e_G, list(range(n))))
    e_G.sort(key=lambda x: x[0], reverse=True)
    key = list(range(0, int(0.25 * n)))
    e_G = [e_G[i] for i in key]
    node1 = [n[1] for n in e_G]
    y_list = []
    x_list = list(range(min_q, max_q, 100000))
    print(max_q)
    for i in range(len(x_list)):
        r = x_list[i]
        L = []
        for j in range(k):
            H = Spl_EffRSparse(adj, r, R)
            e_H = nx.eigenvector_centrality(nx.to_networkx_graph(H), max_iter=10000, weight="weight")
            e_H = np.array(list(e_H.values()))
            e_H = list(zip(e_H, list(range(n))))
            e_H.sort(key=lambda x: x[0], reverse=True)
            e_H = [e_H[i] for i in key]
            node2 = [n[1] for n in e_H]
            L.append(1 - (hamming_distance(node1, node2) / n))
        y_list.append(np.average(L))
        print(r)
    x_list.pop(0)
    y_list.pop(0)
    plt.plot(x_list, y_list, 'ro', color="Purple")
    plt.title("Eigenvector Centrality | Over 10 Averages")
    plt.xlabel("Number of Samples")
    plt.ylabel("EigVec Cen Corr for top 25% of Nodes")
    plt.savefig("Eig.png", transparent=False)
    plt.show()
    return x_list, y_list


def BetweennessCen(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    e_G = nx.betweenness_centrality(nx.to_networkx_graph(adj), weight="weight")
    e_G = np.array(list(e_G.values()))
    e_G = list(zip(e_G, list(range(n))))
    e_G.sort(key=lambda x: x[0], reverse=True)
    key = list(range(0, int(0.25 * n)))
    e_G = [e_G[i] for i in key]
    node1 = [n[1] for n in e_G]
    y_list = []
    x_list = list(range(min_q, max_q, 100000))
    print(max_q)
    for i in range(len(x_list)):
        r = x_list[i]
        L = []
        for j in range(k):
            H = Spl_EffRSparse(adj, r, R)
            e_H = nx.betweenness_centrality(nx.to_networkx_graph(H), weight="weight")
            e_H = np.array(list(e_H.values()))
            e_H = list(zip(e_H, list(range(n))))
            e_H.sort(key=lambda x: x[0], reverse=True)
            e_H = [e_H[i] for i in key]
            node2 = [n[1] for n in e_H]
            L.append(1 - (hamming_distance(node1, node2) / n))
        y_list.append(np.average(L))
        print(r)
    x_list.pop(0)
    y_list.pop(0)
    plt.plot(x_list, y_list, 'ro', color="Purple")
    plt.title("Betweenness Centrality | Over 20 Averages")
    plt.xlabel("Number of Samples")
    plt.ylabel("Betweenness Cen Corr for top 25% of Nodes")
    plt.savefig("Between.png", transparent=False)
    plt.show()
    return x_list, y_list


def ClosenessCen(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    e_G = nx.closeness_centrality(nx.to_networkx_graph(adj))
    e_G = np.array(list(e_G.values()))
    e_G = list(zip(e_G, list(range(n))))
    e_G.sort(key=lambda x: x[0], reverse=True)
    key = list(range(0, int(0.25 * n)))
    e_G = [e_G[i] for i in key]
    node1 = [n[1] for n in e_G]
    y_list = []
    x_list = list(range(min_q, max_q, 100000))
    print(max_q)
    for i in range(len(x_list)):
        r = x_list[i]
        L = []
        for j in range(k):
            H = Spl_EffRSparse(adj, r, R)
            e_H = nx.closeness_centrality(nx.to_networkx_graph(H))
            e_H = np.array(list(e_H.values()))
            e_H = list(zip(e_H, list(range(n))))
            e_H.sort(key=lambda x: x[0], reverse=True)
            e_H = [e_H[i] for i in key]
            node2 = [n[1] for n in e_H]
            L.append(1 - (hamming_distance(node1, node2) / n))
        y_list.append(np.average(L))
        print(r)
    x_list.pop(0)
    y_list.pop(0)
    plt.plot(x_list, y_list, 'ro', color="Purple")
    plt.title("Betweenness Centrality | Over 20 Averages")
    plt.xlabel("Number of Samples")
    plt.ylabel("Betweenness Cen Corr for top 25% of Nodes")
    plt.savefig("Between.png", transparent=False)
    plt.show()
    return x_list, y_list


def DegreeCen(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    e_G = nx.degree_centrality(nx.to_networkx_graph(adj))
    e_G = np.array(list(e_G.values()))
    e_G = list(zip(e_G, list(range(n))))
    e_G.sort(key=lambda x: x[0], reverse=True)
    key = list(range(0, int(0.25 * n)))
    e_G = [e_G[i] for i in key]
    node1 = [n[1] for n in e_G]
    y_list = []
    x_list = list(range(min_q, max_q, 100000))
    print(max_q)
    for i in range(len(x_list)):
        r = x_list[i]
        L = []
        for j in range(k):
            H = Spl_EffRSparse(adj, r, R)
            e_H = nx.degree_centrality(nx.to_networkx_graph(H))
            e_H = np.array(list(e_H.values()))
            e_H = list(zip(e_H, list(range(n))))
            e_H.sort(key=lambda x: x[0], reverse=True)
            e_H = [e_H[i] for i in key]
            node2 = [n[1] for n in e_H]
            L.append(1 - (hamming_distance(node1, node2) / n))
        y_list.append(np.average(L))
        print(r)
    x_list.pop(0)
    y_list.pop(0)
    plt.plot(x_list, y_list, 'ro', color="Purple")
    plt.title("Degree Centrality | Over 20 Averages")
    plt.xlabel("Number of Samples")
    plt.ylabel("Degree Cen Corr for top 25% of Nodes")
    plt.savefig("Degree.png", transparent=False)
    plt.show()
    return x_list, y_list


def MasterCen(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    e_G = nx.eigenvector_centrality(nx.to_networkx_graph(adj), max_iter=10000, weight="weight")
    c_G = nx.closeness_centrality(nx.to_networkx_graph(adj))
    b_G = nx.betweenness_centrality(nx.to_networkx_graph(adj), weight="weight")
    d_G = nx.current_flow_betweenness_centrality(nx.to_networkx_graph(adj), weight='weight')
    e_G = np.array(list(e_G.values()))
    c_G = np.array(list(c_G.values()))
    b_G = np.array(list(b_G.values()))
    d_G = np.array(list(d_G.values()))
    e_G = list(zip(e_G, list(range(n))))
    c_G = list(zip(c_G, list(range(n))))
    b_G = list(zip(b_G, list(range(n))))
    d_G = list(zip(d_G, list(range(n))))
    e_G.sort(key=lambda x: x[0], reverse=True)
    c_G.sort(key=lambda x: x[0], reverse=True)
    b_G.sort(key=lambda x: x[0], reverse=True)
    d_G.sort(key=lambda x: x[0], reverse=True)
    key = list(range(0, int(0.25 * n)))
    e_G = [e_G[i] for i in key]
    c_G = [c_G[i] for i in key]
    b_G = [b_G[i] for i in key]
    d_G = [d_G[i] for i in key]
    e_node1 = [n[1] for n in e_G]
    c_node1 = [n[1] for n in c_G]
    b_node1 = [n[1] for n in b_G]
    d_node1 = [n[1] for n in d_G]
    y_list_e = []
    y_list_c = []
    y_list_b = []
    y_list_d = []
    x_list = list(range(min_q, max_q, 100000))
    print(max_q)
    for i in range(len(x_list)):
        r = x_list[i]
        e_L = []
        c_L = []
        b_L = []
        d_L = []
        for j in range(k):
            H = Spl_EffRSparse(adj, r, R)
            e_H = nx.eigenvector_centrality(nx.to_networkx_graph(H), max_iter=10000, weight="weight")
            c_H = nx.closeness_centrality(nx.to_networkx_graph(H))
            b_H = nx.betweenness_centrality(nx.to_networkx_graph(H), weight="weight")
            d_H = nx.current_flow_betweenness_centrality(nx.to_networkx_graph(H), weight='weight')
            e_H = np.array(list(e_H.values()))
            c_H = np.array(list(c_H.values()))
            b_H = np.array(list(b_H.values()))
            d_H = np.array(list(d_H.values()))
            e_H = list(zip(e_H, list(range(n))))
            c_H = list(zip(c_H, list(range(n))))
            b_H = list(zip(b_H, list(range(n))))
            d_H = list(zip(d_H, list(range(n))))
            e_H.sort(key=lambda x: x[0], reverse=True)
            c_H.sort(key=lambda x: x[0], reverse=True)
            b_H.sort(key=lambda x: x[0], reverse=True)
            d_H.sort(key=lambda x: x[0], reverse=True)
            e_H = [e_H[i] for i in key]
            c_H = [c_H[i] for i in key]
            b_H = [b_H[i] for i in key]
            d_H = [d_H[i] for i in key]
            e_node2 = [n[1] for n in e_H]
            c_node2 = [n[1] for n in c_H]
            b_node2 = [n[1] for n in b_H]
            d_node2 = [n[1] for n in d_H]
            e_L.append(1 - (hamming_distance(e_node1, e_node2) / n))
            c_L.append(1 - (hamming_distance(c_node1, c_node2) / n))
            b_L.append(1 - (hamming_distance(b_node1, b_node2) / n))
            d_L.append(1 - (hamming_distance(d_node1, d_node2) / n))
        y_list_e.append(np.average(e_L))
        y_list_c.append(np.average(c_L))
        y_list_b.append(np.average(b_L))
        y_list_d.append(np.average(d_L))
        print(r)
    x_list.pop(0)
    y_list_e.pop(0)
    y_list_c.pop(0)
    y_list_b.pop(0)
    y_list_d.pop(0)
    return x_list, y_list_e, y_list_c, y_list_b, y_list_d

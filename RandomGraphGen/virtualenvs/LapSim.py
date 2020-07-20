import matplotlib.pyplot as plt
import numpy as np
import random as ran
from scipy import sparse

from PycharmProjects.RandomGraphGen.virtualenvs.NetworkDynamics import Hamming
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist, Spl_EffR
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_Sparse import Spl_EffRSparse, SSEdge, UniEdge


# Create plot of 2-norm of Laplacian
# Input:
# adj - Adj matrix
# R - Matrix of effective resistances
# k - Number of S-S sparsified networks to generate for each number of q
# Output:
# Plot
#   x-axis: # samples from ceil(0.22 * |E|) to O(nlogn) by 1
#   y-axis: 2-norm ||L_G - L_H||_2 from
#   Points: [box] plot with points of 2-norm where line is average
def PlotLapSim(adj, R):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log(n)) / (0.1 ** 2)))
    Lap_G = sparse.csgraph.laplacian(adj)
    x_list = []
    y_list = list(range(min_q, len(E), 50))
    for i in range(len(y_list)):
        r = y_list[i]
        avg = []
        for j in range(10):
            H = Spl_EffRSparse(adj, r, R)
            Lap_H = sparse.csgraph.laplacian(H)
            avg.append(np.linalg.norm(Lap_G - Lap_H, 2) / n)
        x_list.append(np.average(avg))
    y_list.pop(0)
    x_list.pop(0)
    plt.plot(y_list, x_list, 'ro')
    # plt.xticks(np.arange(0, int(np.ceil(np.max(y_list)))), 100)
    # plt.yticks(np.arange(0, int(np.ceil(np.max(x_list)))), 1)
    plt.title("2-Norm ||L_G - L_H||_2")
    plt.xlabel("Number of Samples")
    plt.ylabel("||L_G - L_H||_2 / n")
    plt.savefig("2Norm.png", transparent=True)
    plt.show()
    return x_list


def PlotEpsSim(adj, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    Lap_G = sparse.csgraph.laplacian(adj)
    z_list = []
    y_list = list(range(min_q, max_q, 10000))
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        for j in range(5):
            H = Spl_EffRSparse(adj, r, R)
            Lap_H = sparse.csgraph.laplacian(H)
            Q = np.array(ran.sample(range(n), n))
            Q[Q <= 18] = 0
            Q[Q > 18] = 1
            base = np.transpose(Q) @ Lap_G @ Q
            L.append((np.transpose(Q) @ Lap_H @ Q) / base)
        z_list.append(np.average(L))
        print(r)
    y_list.pop(0)
    z_list.pop(0)
    z_list_norm = [np.abs(z - 1) for z in z_list]
    plt.plot(y_list, z_list, 'ro')
    plt.title("Epsilon | Over 10 Averages")
    plt.xlabel("Number of Samples")
    plt.ylabel("Epsilon")
    plt.savefig("Ep.png", transparent=True)
    plt.show()
    return z_list_norm, z_list


def EffRElen(adj, R, k, t):
    E = Mtrx_Elist(adj)[0]
    y_list = list(range(0, len(E), 10))
    ss_mx = []
    ss_mn = []
    ss_avg = []
    un_mx = []
    un_mn = []
    un_avg = []
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        for j in range(k):
            H = SSEdge(adj, R, r)
            for m in range(t):
                R_H = Spl_EffR(H, 0.1)
                L.append(np.max(R_H - R))
        ss_mx.append(np.max(L))
        ss_mn.append(np.min(L))
        print("1", r)
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        for j in range(k):
            H = UniEdge(adj, r)
            for m in range(t):
                R_H = Spl_EffR(H, 0.1)
                L.append(np.max(R_H - R))
        un_mx.append(np.max(L))
        un_mn.append(np.min(L))
        print("2", r)
    return ss_mx, ss_mn, ss_avg, un_mx, un_mn, un_avg, y_list

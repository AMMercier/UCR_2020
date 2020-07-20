import random as ran
import numpy as np
import pandas as df

from PycharmProjects.RandomGraphGen.virtualenvs.AdaptiveAlgo import Adapt1
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist


# Normalize probs such that sum(probs)=1
# Input:
# P - list of probs
# Output:
# P_n - list of probs' that sum to 1
def normprobs(P):
    prob_fac = 1 / sum(P)
    P_n = [prob_fac * p for p in P]
    return P_n


# Create a list of edge R_eff
# Input:
# R - Array of R_effs
# adj - Adj matrix
# Output:
# R_list - list of edge R_eff
def EffR_List(R, adj):
    R_list = []
    adj = np.triu(adj)
    R = np.triu(R)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] > 0:
                R_list.append(R[i][j])
    return R_list


# Create a effective resistance sparsifer
# From Spielman and Srivastava 2011
# Input:
# adj - Adj matrix
# q - number of samples
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# Output:
# H - effective resistance sparsifer adj matrix
def Spl_EffRSparse(adj, q, R):
    n = len(adj)
    E_list = Mtrx_Elist(adj)[0]
    W_list = Mtrx_Elist(adj)[1]
    R_list = EffR_List(R, adj)
    P = []
    for i in range(len(E_list)):
        w_e = W_list[i]
        R_e = R_list[i]
        P.append((w_e * R_e) / (n - 1))
    Pn = normprobs(P)
    # E = df.DataFrame({'e': E_list, 'w': W_list, 'p': Pn})
    C = ran.choices(list(zip(E_list, W_list, Pn)), Pn, k=q)
    H = np.zeros(shape=(n, n))
    for x in range(q):
        e = C[x][0]
        w_e = C[x][1]
        p_e = C[x][2]
        H[e[0]][e[1]] += w_e / (q * p_e)
    return H + np.transpose(H)


def EList_Mtrx(data, n):
    A = np.zeros(shape=(n, n))
    for i in range(len(data)):
        n1 = data[0][i]
        n2 = data[1][i]
        A[n1][n2] = 1
    return A


# Create a random uniform sparsifier
# Input:
# adj - Adj matrix
# q - number of samples
# Output:
# H
def UniSampleSparse(adj, q):
    n = len(adj)
    E_list = Mtrx_Elist(adj)[0]
    W_list = Mtrx_Elist(adj)[1]
    Pn = [1 / len(E_list)] * len(E_list)
    C = ran.choices(list(zip(E_list, W_list, Pn)), Pn, k=q)
    H = np.zeros(shape=(n, n))
    for x in range(q):
        e = C[x][0]
        w_e = C[x][1]
        p_e = C[x][2]
        H[e[0]][e[1]] += w_e / (q * p_e)
    return H + np.transpose(H)


# Create a S-S sparsifier with a specificed number of edges.
# Input:
# adj - Adj matrix
# e - number of edges
# Output:
# H - effective resistance sparsifer adj matrix
def SSEdge(adj, R, e):
    H = np.zeros(shape=(len(adj), len(adj)))
    r_tick = int(1.25 * e)
    while len(Mtrx_Elist(H)[0]) != e:
        H = Spl_EffRSparse(adj, r_tick, R)
        if len(Mtrx_Elist(H)[0]) > e:
            r_tick = r_tick - (len(Mtrx_Elist(H)[0]) - e)
        if len(Mtrx_Elist(H)[0]) < e:
            r_tick = r_tick + (e - len(Mtrx_Elist(H)[0]))
        print(len(Mtrx_Elist(H)[0]))
    return H


# Create a Uni sparsifier with a specificed number of edges.
# Input:
# adj - Adj matrix
# e - number of edges
# Output:
# H - uni sparsifer adj matrix
def UniEdge(adj, e):
    H = np.zeros(shape=(len(adj), len(adj)))
    r_tick = int(0.9 * e)
    while len(Mtrx_Elist(H)[0]) != e:
        H = UniSampleSparse(adj, r_tick)
        if len(Mtrx_Elist(H)[0]) > e:
            r_tick = r_tick - (len(Mtrx_Elist(H)[0]) - e)
        if len(Mtrx_Elist(H)[0]) < e:
            r_tick = r_tick + (e - len(Mtrx_Elist(H)[0]))
        print(len(Mtrx_Elist(H)[0]))
    return H


def AdaptEdge(adj, R, T, e):
    H = np.zeros(shape=(len(adj), len(adj)))
    r_tick = int(1.5 * e)
    while len(Mtrx_Elist(H)[0]) != e:
        H = Adapt1(adj, r_tick, R, T)
        if len(Mtrx_Elist(H)[0]) > e:
            r_tick = int(r_tick - (len(Mtrx_Elist(H)[0]) - (e * 1/T)))
        if len(Mtrx_Elist(H)[0]) < e:
            r_tick = int(r_tick + (e - (len(Mtrx_Elist(H)[0]) * (1/T))))
        print(len(Mtrx_Elist(H)[0]))
    return H

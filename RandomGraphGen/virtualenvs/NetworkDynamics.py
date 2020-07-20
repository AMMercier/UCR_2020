import random as ran
import numpy as np
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist


# Calculate hamming distance
# Input:
# string1 - matrix of size t by n
# string2 - matrix of size t by n
# Output:
# C - list of integers
def Hamming(string1, string2):
    if len(string1) != len(string2):
        raise Exception("Cannot compute for strings of unequal length.")
    C = []
    for i in range(np.shape(string1)[0]):
        dist = 0
        for j in range(np.shape(string1)[1]):
            if string1[i][j] != string2[i][j]:
                dist += 1
        C.append(dist)
    return C


# Perform SI dynamics on any given network
# Input:
# adj - Adj matrix
# k - integer, first infected node
# t - number of time steps to run
# Output:
# I - a t by n matrix of 1s and 0s indicating I and S respectivally
def SI_model(adj, prob, k, t):
    E = Mtrx_Elist(adj)
    edges = E[0]
    weights = E[1]
    Infected = [k]
    I = [np.zeros(len(adj))]
    I[0][k] = 1
    for i in range(t):
        It = np.zeros(shape=(len(adj)))
        for j in range(len(Infected)):
            n1 = Infected[j]
            for x in range(len(edges)):
                if n1 == edges[x][0]:
                    n2 = edges[x][1]
                    w = weights[x]
                    infec_p = 1 - (1 - prob) ** w
                    p = ran.uniform(0, 1)
                    if p < infec_p:
                        if n2 not in Infected:
                            Infected.append(n2)
                elif n1 == edges[x][1]:
                    n2 = edges[x][0]
                    w = weights[x]
                    infec_p = 1 - (1 - prob) ** w
                    p = ran.uniform(0, 1)
                    if p < infec_p:
                        if n2 not in Infected:
                            Infected.append(n2)
        for y in range(len(Infected)):
            index = Infected[y]
            It[index] = 1
        I.append(It)
    return np.array(I)


# Perform Create Adj Matrix from Edge List
# Input:
# E - Edge list
# n - number of nodes
# Output:
# adj - Adj matrix
def Elist_Mtrx(E, n):
    adj = np.zeros(shape=(n, n))
    for i in range(len(E)):
        x = E[i][0]
        y = E[i][1]
        adj[x][y] = 1
    return adj


# Create SI tree
# Input:
# adj - Adj matrix
# k - integer, first infected node
# t - number of time steps to run
# Output:
# I - a t by n matrix of 2s, 1s, and 0s indicating R, I, and S respectivally
def SI_Tree(adj, prob, k):
    E = Mtrx_Elist(adj)
    edges = E[0]
    weights = E[1]
    Infected = [k]
    Edges = []
    while len(Infected) < len(adj):
        for j in range(len(Infected)):
            n1 = Infected[j]
            for x in range(len(edges)):
                if n1 == edges[x][0] or n1 == edges[x][1]:
                    n2 = edges[x][0]
                    n2_2 = edges[x][1]
                    w = weights[x]
                    infec_p = 1 - (1 - prob) ** w
                    p = ran.uniform(0, 1)
                    if p < infec_p:
                        if n2 not in Infected:
                            Infected.append(n2)
                            Edges.append(edges[x])
                        if n2_2 not in Infected:
                            Infected.append(n2_2)
                            Edges.append(edges[x])
    return Edges


def SIEdgeImportance(adj, prob, t):
    A = np.zeros(shape=(len(adj),len(adj)))
    for i in range(t):
        for j in range(len(adj)):
            Tree = SI_Tree(adj, prob, j)
            for x in Tree:
                A[x[0], x[1]] += 1
        print(i)
    A = (1/sum(sum(A))) * A
    return A + np.transpose(A)


# Perform SIR dynamics on any given network
# Input:
# adj - Adj matrix
# k - integer, first infected node
# t - number of time steps to run
# Output:
# I - a t by n matrix of 2s, 1s, and 0s indicating R, I, and S respectivally
def SIR_model(adj, prob, k, t, gamma):
    E = Mtrx_Elist(adj)
    edges = E[0]
    weights = E[1]
    Infected = [k]
    Recovered = []
    I = [np.zeros(len(adj))]
    I[0][k] = 1
    for i in range(t):
        It = np.zeros(shape=(len(adj)))
        for j in range(len(Infected)):
            n1 = Infected[j]
            for x in range(len(edges)):
                if n1 == edges[x][0] or n1 == edges[x][1]:
                    n2 = edges[x][0]
                    n2_2 = edges[x][1]
                    w = weights[x]
                    infec_p = 1 - (1 - prob) ** w
                    p = ran.uniform(0, 1)
                    if p < infec_p:
                        if n2 not in Infected:
                            Infected.append(n2)
                        if n2_2 not in Infected:
                            Infected.append(n2_2)
            p = ran.uniform(0, 1)
            if p < gamma:
                Recovered.append(n1)
        for y in range(len(Infected)):
            index1 = Infected[y]
            It[index1] = 1
        for y in range(len(Recovered)):
            index2 = Recovered[y]
            It[index2] = 2
        I.append(It)
    return np.array(I)

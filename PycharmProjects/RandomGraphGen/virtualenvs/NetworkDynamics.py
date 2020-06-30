import random as ran
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math as m
from scipy.sparse.linalg import cg
from scipy import stats as stat
from scipy import sparse
from Spielman_EffR import Mtrx_Elist

#Calculate hamming distance
#Input:
# string1 - matrix of size t by n
# string2 - matrix of size t by n
#Output:
# C - list of integers
def Hamming(string1, string2):
    if len(string1) != len(string2):
        raise ("Cannot compute for strings of unequal length.")
    C = []
    for i in range(np.shape(string1)[0]):
        dist = 0
        for j in range(np.shape(string1)[1]):
            if string1[i][j] != string2[i][j]:
                dist += 1
        C.append(dist)
    return C

#Perform SI dynamics on any given network
#Input:
# adj - Adj matrix
# k - integer, first infected node
# t - number of time steps to run
#Output:
# I - a t by n matrix of 1s and 0s indicating I and S respectivally
def SI_model(adj,k,t):
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
                if n1 == edges[x][0] or n1 == edges[x][1]:
                    n2 = edges[x][0]
                    n2_2 = edges[x][1]
                    w = weights[x]
                    p = ran.uniform(0,1)
                    if p < w:
                        if n2 not in Infected:
                            Infected.append(n2)
                        if n2_2 not in Infected:
                            Infected.append(n2_2)
        for y in range(len(Infected)):
            index = Infected[y]
            It[index] = 1
        I.append(It)
    return np.array(I)

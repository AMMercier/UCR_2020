import random as ran
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math as m
import scipy.sparse as sci
import pandas as df
from scipy import stats as stat
from Spielman_EffR import Spl_EffR
from Spielman_EffR import Mtrx_Elist
from scipy.sparse import csgraph

#Normalize probs such that sum(probs)=1
#Input:
# P - list of probs
#Output:
# P_n - list of probs' that sum to 1
def normprobs(P):
    prob_fac = 1/sum(P)
    P_n = [prob_fac * p for p in P]
    return P_n

#Create a effective resistance sparsifer
#From Spielman and Srivastava 2011
#Input:
# adj - Adj matrix
# q - number of samples
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
#Output:
# A_spar - effective resistance sparsifer adj matrix
def Spl_EffRSparse(adj,q,R):
    n = len(adj)
    P = []
    P_t = 0
    u_adj = np.triu(adj)
    E_list = Mtrx_Elist(adj)[0]
    W_list = Mtrx_Elist(adj)[1]
    for i in range(n):
        for j in range(n):
            if u_adj[i][j] > 0:
                w_e = adj[i][j]
                R_e = R[i][j]
                P.append((w_e*R_e)/(n-1))
                P_t = (w_e*R_e)/(n-1) + P_t
    Pn = normprobs(P)
    E = df.DataFrame({'e':E_list,'w':W_list,'p':Pn})
    E = E.sort_values(by='p')
    C = ran.choices(list(range(len(E))),E['w'],k=q)
    H = np.zeros(shape=(n,n))
    for i in range(q):
        x = C[i]
        e = E['e'][x]
        w_e = E['w'][x]
        p_e = E['p'][x]
        if p_e != 0:
            H[e[0]][e[1]] += w_e/(q*p_e)
    return H + np.transpose(H)



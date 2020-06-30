import random as ran
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math as m
import itertools
from RanGraphGen import ER_gen
from GraphVis import show_graph

def LapMatrix(adj):
    n = len(adj)
    L = np.zeros(shape = (n,n))
    for i in range(len(adj)):
        d = adj[:,i].sum()
        L[i][i] = d
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i != j:
                if adj[i][j] > 0:
                    L[i][j] = (-1)*adj[i][j]
    return L



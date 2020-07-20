import itertools as it
import random as ran
import numpy as np
from scipy import stats as stt


# Generate an Erdos-Renyi Model
# Input:
# n - the number of nodes
# p - probability of adding an edge
# Output:
# A - adj matrix of ER network
def ER_gen(n, p):
    A = np.zeros(shape=(n, n))  # Generate all zero adj matrix
    for i in range(0, n):
        for j in range(0, n):
            if i <= j:
                theta = ran.uniform(0, 1)
                if theta < p:  # With prob p add an edge between two verticies
                    if i != j:
                        A[i][j] += stt.uniform.rvs()
                    # else:  # Self edges have an added degree of two
                    #    A[i][i] += 2 * stt.uniform.rvs()
    C = np.transpose(A)
    C = np.tril(C, -1)
    return A + C


# Generate a Configuration Model
# Input:
# k - degree list
# Output:
# A - adj matrix of Configuration network
def ConfigGraphGen(k):
    if sum(k) % 2 != 0:  # The sum of the degrees must be even
        raise Exception("Please use a degree distribution that sums two an even number.")
    L = []
    A = np.zeros(shape=(len(k), len(k)))  # Create an all 0 adj matrix
    for i in range(0, len(k)):
        for j in range(0, k[i]):
            L.append(i)  # Create stub list
    while len(L) > 0:
        ran.shuffle(L)  # Randomly shuffle stublist
        i = L.pop()
        j = L.pop()
        if i == j:
            A[i][i] += 2  # Self edges have an added degree of two
        else:
            A[i][j] = A[j][i] = 1 + A[i][j]
    return A


# Create a random partition of nodes with given community sizes
# Input:
# L - nodelist
# s - list of community sizes
# Output:
# P - list of lists; a partition based on L and s
def l_g_partition(L, s):
    # ran.shuffle(L) #randomly shuffle nodelist
    P = []
    for i in range(len(s)):
        Tp = []
        for j in range(s[i]):
            l = L.pop()
            Tp.append(l)  # Add partition to a list of partitions
        P.append(Tp)
    return P


# Create a SBM
# Input:
# s - list of community sizes
# W - matrix of probabilities where W_ij is the prob of connecting community i and j
# Output:
# A - adj matrix of Configuration network
def SBM(s, W):
    n = sum(s)  # Find total number of nodes
    A = np.zeros(shape=(n, n))  # Create 0 matrix of size nxn
    nodelist = list(range(n))  # Create a nodelist
    P = l_g_partition(nodelist, s)  # Create partition of nodelist
    L = []
    for i in range(len(P)):
        for j in range(len(P)):
            C1 = P[i]
            C2 = P[j]
            if i <= j:
                for x in range(len(C1)):
                    for y in range(len(C2)):
                        if x <= y:
                            pr = W[i][j]
                            theta = ran.uniform(0, 1)
                            if theta < pr:
                                n = C1[x]
                                m = C2[y]
                                if n != m:
                                    A[n][m] += ran.uniform(0, 2000)
                                    if i == j:
                                        A[n][m] += ran.uniform(1000, 2000)
                                else:
                                    A[n][n] += 0
                                    if i == j:
                                        A[n][n] += 0
    C = np.transpose(A)
    C = np.triu(C, 1)
    return A + C


# Measure euclidean distance between two vectors
# Input:
# x - vector 1
# y - vector 2
# Output:
# dis - distance between x and y
def euclid(x, y):
    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    return dist


# Generate random undirected geometric graph
# Input:
# n - the number of nodes
# r - minimum dist to add edge
# Output:
# A - the adj matrix of the graph
# pos - dict of lists, each entry the x, y coordinate
def GeoRan(n, r):
    pos = {}
    for i in range(n):
        x = ran.uniform(0, 1)
        y = ran.uniform(0, 1)
        pos[i] = [x, y]
    A = np.zeros(shape=(n, n))
    for a, b in it.combinations(pos, 2):
        if euclid(pos[a], pos[b]) <= r:
            A[a][b] = A[b][a] = stt.uniform.rvs(0, 0.5)  # Change this when needed
    return [A, pos]


# Generate random undirected lattice
# Input:
# n - the number of nodes
# Output:
# A - the adj matrix of the graph
# pos - dict of lists, each entry the x, y coordinate
def Lattice(n, shape):
    if shape[0] * shape[1] != n:
        raise Exception("Your dim must equal the number of nodes.")
    pos = {}
    N = list(range(n))
    for m in range(n):
        tick = 0
        for i in range(shape[0]):
            # for j in range(shape[1]):
            x = tick
            y = i
            pos[N[m]] = [x, y]
            tick += 1
    A = np.zeros(shape=(n, n))
    for a, b in it.combinations(pos, 2):
        if euclid(pos[a], pos[b]) <= np.sqrt(2):
            A[a][b] = A[b][a] = stt.uniform.rvs(0, 0.5)  # Change this when needed
    return [A, pos]


W = [[0.4, 0.01, 0.03, 0.01, 0.01],
     [0.01, 0.3, 0.02, 0.01, 0.01],
     [0.03, 0.02, 0.5, 0.01, 0.01],
     [0.01, 0.01, 0.0, 0.6, 0.01],
     [0.01, 0.01, 0.01, 0.01, 0.3]]

A = [[0, 1, 1, 1, 0, 0, 0, 0],
     [1, 0, 1, 1, 0, 0, 0, 0],
     [1, 1, 0, 1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 1, 0, 1, 1],
     [0, 0, 0, 0, 1, 1, 0, 1],
     [0, 0, 0, 0, 1, 1, 1, 0]]

import math as m
import random as ran
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


# Compute weight diagonal matrix
# Input:
# weights - weights matrix of a graph
# Output:
# W - matrix with weights on the diagonal
def WDiag(adj):
    E = Mtrx_Elist(adj)  # Generate edge and weight list
    weights = E[1]
    m = len(weights)
    row = list(range(m))
    col = list(range(m))
    W = sparse.csr_matrix((weights, (row, col)), shape=(m, m))
    # W = np.zeros(shape=(m, m))
    # for i in range(m):
    #    W[i][i] = weights[i] #Add weights on diag
    return W


# Compute vertex incidence matrix
# Input:
# Adj - Adj matrix of a graph
# Output:
# B - vertex incidence matrix made up of vertex incidence vectors
def sVIM(adj):
    E = Mtrx_Elist(adj)
    E_list = E[0]
    m = len(E_list)
    n = len(adj)
    # B=np.zeros(shape = (m,n))
    data = []
    row = []
    col = []
    ran.seed(30)
    for i in range(m):
        p = ran.uniform(0, 1)
        edges = E_list[i]
        n1 = edges[0]  # Find head/tail 1
        n2 = edges[1]  # Find head/tail 1
        if p < 0.5:  # Randomly determine head/tail orientation
            if n1 != n2:
                row.append(i)
                row.append(i)
                col.append(n1)
                col.append(n2)
                data.append(-1)
                data.append(1)
        else:
            if n1 != n2:
                row.append(i)
                row.append(i)
                col.append(n1)
                col.append(n2)
                data.append(1)
                data.append(-1)
    B = sparse.csr_matrix((data, (row, col)), shape=(m, n))
    return B


def VIM(adj):
    E = Mtrx_Elist(adj)
    E_list = E[0]
    m = len(E_list)
    n = len(adj)
    B = np.zeros(shape=(m, n))
    ran.seed(30)
    for i in range(m):
        p = ran.uniform(0, 1)
        edges = E_list[i]
        n1 = edges[0]  # Find head/tail 1
        n2 = edges[1]  # Find head/tail 1
        if p < 0.5:  # Randomly determine head/tail orientation
            if n1 != n2:
                B[i][n1] = -1
                B[i][n2] = 1
            else:  # Self edge case
                B[i][n1] = 0
        else:
            if n1 != n2:
                B[i][n1] = -1
                B[i][n2] = 1
            else:  # Self edge case
                B[i][n1] = 0
    return sparse.csr_matrix(B)


# Compute Laplacian via BGB^T
# Input:
# Adj - Adj matrix
# Output:
# L - Laplacian
def Lplcn(adj):
    B = VIM(adj)
    W = WDiag(adj)
    BT = np.transpose(B)
    L = BT @ W @ B
    return sparse.csr_matrix(L)


# Compute and order Eigen values and vectors
# Input:
# L - Laplican matrix
# Output:
# L_i - Pseudoinverse
def NonZEigValVec(L):
    E = np.linalg.eig(L)
    E_val = E[0]
    E_vec = E[1]
    Eig = []
    for i in range(0, len(E_vec)):
        u = E_vec[:, i]
        lm = round(E_val[i], 15)
        Eig.append([lm, u])
    Eig.sort(key=lambda x: x[0])  # Order eigen values with corresponding eigen vectors
    return Eig


# Compute Moore-Penrose Pseudoinverse of Laplican
# Input:
# L - Laplican matrix
# Output:
# L_i - Pseudoinverse
def MPPI(L):
    Eig = NonZEigValVec(L)
    Eig.pop(0)  # Remove 0 eigen value and vector
    L_ps = 0
    for i in range(len(Eig)):
        lm = Eig[i][0]
        u = Eig[i][1]
        u = np.transpose(u[np.newaxis])  # Make sure u has dim for transpose
        uut = u @ np.transpose(u)
        L_ps = (1 / lm) * uut + L_ps
    return sparse.csr_matrix(L_ps)


# Edge list to adj matrix
# Input:
# elist - List of lists giving edges
# weights - List of weights (default none)
# Output:
# adj - Adj matrix

# Adj matrix to edge list
# Input:
# adj - Adj matrix
# Output:
# Elist - List of lists giving edges
def Mtrx_Elist(adj):
    n = len(adj)
    elist = []
    weights = []
    u_adj = np.triu(adj)
    for j in range(n):
        for i in range(n):
            if u_adj[i][j] > 0:
                elist.append([j, i])
                weights.append(u_adj[i, j])
    return elist, weights


# Compute random \pm 1\sqrt(k) Bernoulli matrix where k=24log(n)/epsilon^2
# From Spielman and Srivastava 2011
# Input:
# adj - Adj matrix
# epsilon - List of weights (default none)
# Output:
# Q - Random \pm 1/sqrt(k) Bernoulli matrix
def QGen(n, epsilon):
    k = ((24 * np.log(n)) / (epsilon ** 2))  # Define k
    k_D = m.ceil(k)
    # = np.ceil(Np)
    # data = [1] *
    # tick = 0
    # row = []
    # col = []
    Q = np.zeros(shape=(k_D, n))
    C = ran.sample(range(k_D * n), k_D * n)
    p = 0.5 * (k_D * n)
    t = 0
    for i in range(k_D):
        for j in range(n):
            Q[i][j] = C[t]
            t += 1
    Q = sparse.csr_matrix(Q)
    x = 1 / np.sqrt(k)
    # Q = sparse.csr_matrix((data,(row,col)),shape=(k_D))
    # Q = np.array(stat.bernoulli.rvs(size=(k_D,n),p=0.5), dtype=float) #Gen random \pm k Bernoulli matrix
    Q[Q <= p] = x
    Q[Q > p] = -x
    return Q


# Approximately find effective resistances for edges of a graph
# Based on alogrithm from Spielman-Srivastava (2011)
# Input:
# elist - list of edges to find effective resistance
# e - edges of the graph
# w - weights of edges
# tol - relative error in the SLE solution
# epsilon - parameter to control accuracy
# Output:
# T - vector of effective resistances for the edges in elist
def Spl_EffR(adj, epsilon):
    B = sVIM(adj)
    A = sparse.csr_matrix(adj)
    L = sparse.csgraph.laplacian(A)
    W = WDiag(adj)
    Q = QGen(n=sparse.csr_matrix.get_shape(B)[0], epsilon=epsilon)
    Y = Q @ (np.sqrt(W)) @ B
    del B, W, Q
    Z = []
    for i in range(sparse.csr_matrix.get_shape(Y)[0]):
        Yr = Y[i, :]
        nZ = cg(L, np.transpose(Yr.toarray()))
        Z.append(np.transpose(nZ[0]))
    Z = np.array(Z)
    # NOTE ## WE CAN COMPUTE THIS WITH scipy.sparse.linalg.cg(L,Y[:,i]) ##
    # THIS WILL GIVE ROWS OF Z, WHICH THEN CAN BE USED BELOW FOR NORM ##
    # MAY BE FASTER IN THE END - CAN TEST LATER ##
    R = np.zeros(shape=(len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i != j:
                R[i][j] = abs(sum((Z[:, i] - Z[:, j]) ** 2))
    return R


# Inverse Method
def Spl_EffRSlow(adj, epsilon):
    B = sVIM(adj)
    # A = sparse.csr_matrix(adj)
    L = sparse.csgraph.laplacian(adj)
    W = WDiag(adj)
    Q = QGen(n=sparse.csr_matrix.get_shape(B)[0], epsilon=epsilon)
    L_t = MPPI(L)
    Z = Q @ (np.sqrt(W)) @ B @ L_t
    del B, W, Q
    # for i in range(sparse.csr_matrix.get_shape(Y)[0]):
    #    Yr = Y[i, :]
    #    nZ = cg(L, np.transpose(Yr.toarray()))
    #    Z.append(np.transpose(nZ[0]))
    # NOTE ## WE CAN COMPUTE THIS WITH scipy.sparse.linalg.cg(L,Y[:,i]) ##
    # THIS WILL GIVE ROWS OF Z, WHICH THEN CAN BE USED BELOW FOR NORM ##
    # MAY BE FASTER IN THE END - CAN TEST LATER ##
    R = np.zeros(shape=(len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i != j:
                R[i][j] = abs(sum((Z[:, i] - Z[:, j]) ** 2))
    return R

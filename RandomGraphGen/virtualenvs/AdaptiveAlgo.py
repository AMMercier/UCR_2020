import random as ran
import numpy as np
import pandas as df
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist
from PycharmProjects.RandomGraphGen.virtualenvs.NetworkDynamics import Hamming, SI_model
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_Sparse import normprobs, EffR_List, UniEdge, UniSampleSparse


# Create a effective resistance sparsifer
# From Spielman and Srivastava 2011
# Input:
# adj - Adj matrix
# q - number of samples
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# Output:
# A_spar - effective resistance sparsifer adj matrix
def Spl_EffRSparse2(adj, q, R):
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
                P.append((w_e * R_e) / (n - 1))
                P_t = (w_e * R_e) / (n - 1) + P_t
    Pn = normprobs(P)
    E = df.DataFrame({'e': E_list, 'w': W_list, 'p': Pn})
    E = E.sort_values(by='p')
    C = ran.choices(list(range(len(E))), E['p'], k=q)
    H = np.zeros(shape=(n, n))
    for i in range(q):
        x = C[i]
        e = E['e'][x]
        w_e = E['w'][x]
        p_e = E['p'][x]
        if p_e != 0:
            H[e[0]][e[1]] += w_e / (q * p_e)
    H = H + np.transpose(H)
    return H, Pn


# Create a Boltzmann distribution from a list of probabilites
# P(i,j) ~ e^(H(i,J)/T) where H(i,j) = -log(p(i,j))
# Input:
# p - List of probabilites
# T - Temperature
# Output:
# P - List of probabilties
def BoltzProb(p, T):
    P = []
    for i in range(len(p)):
        H = -1 * np.log(p[i])
        Pi = np.exp(H / T)
        P.append(Pi)
    return normprobs(P)


# Create a Boltzmann distribution from a matrix of R_eff
# P(i,j) ~ e^(H(i,J)/T) where H(i,j) = -log(R_eff(i,j))
# Input:
# R - List of probabilites
# T - Temperature
# Output:
# P - List of probabilties
def BoltzEffR(R, T):
    R_B = np.zeros(shape=(len(R), len(R)))
    R = np.triu(R)
    for i in range(len(R)):
        for j in range(len(R)):
            if i < j:
                Pi = R[i][j] ** (1 / T)
                R_B[i][j] = Pi
    return R_B + np.transpose(R_B)


# Create a effective resistance sparsifer on adaption-based algorithm using temp-based idea
# Input:
# adj - Adj matrix
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# T - Temperature or Boltzmann distribution
# Output:
# A_spar - effective resistance sparsifer adj matrix
def Adapt1(adj, q, R, T):
    n = len(adj)
    E_list = Mtrx_Elist(adj)[0]
    W_list = Mtrx_Elist(adj)[1]
    R_B = BoltzEffR(R, T)
    R_list = EffR_List(R_B, adj)
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


# Create a effective resistance sparsifer on adaption-based algorithm
# Criterion: Performance of dynamics
# Input:
# adj - Adj matrix
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# Output:
# A_spar - effective resistance sparsifer adj matrix
def Adapt2(adj, R, T, prob, prop_remove, Boltz="R_eff"):
    n = len(adj)
    u_adj = np.triu(adj)
    E_list = Mtrx_Elist(adj)[0]
    W_list = Mtrx_Elist(adj)[1]
    P = []
    if Boltz == "R_eff":
        R_B = BoltzEffR(R, T)
        for i in range(n):
            for j in range(n):
                if u_adj[i][j] > 0:
                    w_e = adj[i][j]
                    R_e = R_B[i][j]
                    P.append((w_e * R_e) / (n - 1))
    P_B = normprobs(P)
    E = df.DataFrame({'e': E_list, 'w': W_list, 'p': P_B})
    E = E.sort_values(by='p')
    H = Hg = np.zeros(shape=(n, n))
    base = []
    for i in range(n):
        base.append(SI_model(adj, prob, i, 25))
    f = SI_model(H, prob, ran.randint(0, n - 1), 25)
    case = np.average(Hamming(f, base[0]))
    q = 0
    while len(Mtrx_Elist(Hg)[0]) < (1 - prop_remove) * len(E_list):
        H = Hg
        C = ran.choices(list(range(len(E))), E['p'])[0]
        e = E['e'][C]
        w_e = E['w'][C]
        p_e = E['p'][C]
        if p_e != 0:
            H[e[0]][e[1]] += w_e / p_e
        tdym = SI_model(H, prob, ran.randint(0, n - 1), 25)
        val = []
        for i in range(n):
            if np.average(Hamming(tdym, base[i])) < case:
                Hg = H
                val.append(np.average(Hamming(tdym, base[i])))
                q += 1
        if len(val) != 0:
            case = np.min(val)
            print(case)
    Hg = (Hg + np.transpose(Hg)) * 1 / q
    return Hg


# Create a effective resistance sparsifer on adaption-based algorithm
# Criterion: Gentic "fitness"
# Input:
# adj - Adj matrix
# R - Matrix of effective resistances (other types of edge importance in the future, possibly)
# pop - Number of population to generate
# Output:
# A_spar - effective resistance sparsifer adj matrix
def Adapt3(adj, R, pop):
    while True:
        I = []
        base = SI_model(adj, 10, 50)
        for k in range(5):
            I_k = np.average(Hamming(SI_model(adj, 10, 50), base))
            I.append(I_k)
        base_avg = np.average(I)
        ne = len(Mtrx_Elist(adj)[0])
        landscape = []
        for i in range(pop):
            var = ran.uniform(0, 1)
            Creature = list(Spl_EffRSparse2(adj, int(np.ceil(var * ne)), R))
            Creature.append(0)
            landscape.append(Creature)
        Comp = []
        for j in range(pop):
            F = []
            for k in range(5):
                F_k = np.average(Hamming(SI_model(landscape[j][0], 10, 50), base))
                F.append(F_k)
            Comp.append((np.average(F), landscape[j][0]))
        Comp.append(base_avg)
        return Comp


def UniSimElenVar(adj, prob, R, k, t, time):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    y_list = list(range(10, len(E), 50))
    x_listl = []
    x_listm = []
    x_list = []
    x_listu = []
    ssl_mx = []
    ssl_mn = []
    ssl_avg = []
    ssm_mx = []
    ssm_mn = []
    ssm_avg = []
    ss_mx = []
    ss_mn = []
    ss_avg = []
    un_mx = []
    un_mn = []
    un_avg = []
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        X = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = Adapt1(adj, r, R, 0.1)
            X.append(len(Mtrx_Elist(H)[0]))
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        ssl_avg.append(np.average(L))
        ssl_mx.append(np.max(L))
        ssl_mn.append(np.min(L))
        print("1", r)
        x_listl.append(np.average(X))
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        X = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = Adapt1(adj, r, R, 0.5)
            X.append(len(Mtrx_Elist(H)[0]))
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        ssm_avg.append(np.average(L))
        ssm_mx.append(np.max(L))
        ssm_mn.append(np.min(L))
        print("2", r)
        x_listm.append(np.average(X))
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        X = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = Adapt1(adj, r, R, 1)
            X.append(len(Mtrx_Elist(H)[0]))
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        ss_avg.append(np.average(L))
        ss_mx.append(np.max(L))
        ss_mn.append(np.min(L))
        print("3", r)
        x_list.append(np.average(X))
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        X = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = UniSampleSparse(adj, r)
            X.append(len(Mtrx_Elist(H)[0]))
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        un_mx.append(np.max(L))
        un_mn.append(np.min(L))
        un_avg.append(np.average(L))
        print("4", r)
        x_listu.append(np.average(X))
    return ss_mx, ss_mn, ss_avg, ssl_mx, ssl_mn, ssl_avg, ssm_mx, ssm_mn, ssm_avg, un_mx, un_mn, un_avg, y_list, x_listl, x_listm, x_list, x_listu

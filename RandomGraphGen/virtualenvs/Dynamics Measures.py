import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random as ran
from scipy import sparse
from PycharmProjects.RandomGraphGen.virtualenvs.AdaptiveAlgo import Adapt1
from PycharmProjects.RandomGraphGen.virtualenvs.NetworkDynamics import SI_model, Hamming
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_EffR import Mtrx_Elist
from PycharmProjects.RandomGraphGen.virtualenvs.Spielman_Sparse import UniSampleSparse, SSEdge, UniEdge

from math import log


def variation_of_information(X, Y):
    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)


def SISim(adj, prob, R, k):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    min_q = int(np.ceil(0.22 * len(E)))
    max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
    z_list_1 = []
    z_list_2 = []
    z_list_3 = []
    y_list = list(range(min_q, len(E), 100))
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
        for j in range(k):
            H = Adapt1(adj, r, R, 0.2)
            I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
            L.append(np.abs(np.average(Hamming(I_G, I_H))))
        z_list_1.append(np.average(L))
        print("1", r)
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
        for j in range(k):
            H = Adapt1(adj, r, R, 1)
            I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
            L.append(np.abs(np.average(Hamming(I_G, I_H))))
        z_list_2.append(np.average(L))
        print("2", r)
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
        for j in range(k):
            H = Adapt1(adj, r, R, 100)
            I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
            L.append(np.abs(np.average(Hamming(I_G, I_H))))
        z_list_3.append(np.average(L))
        print("3", r)
    y_list.pop(0)
    z_list_1.pop(0)
    z_list_2.pop(0)
    z_list_3.pop(0)

    return z_list_1, z_list_2, z_list_3, y_list


def UniSimElen(adj, prob, R, k, t, time):
    E = Mtrx_Elist(adj)[0]
    n = len(adj)
    y_list = list(range(0, len(E), 1))
    ss_mx = []
    ss_mn = []
    ss_avg = []
    un_mx = []
    un_mn = []
    un_avg = []
    base = []
    for i in range(k):
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        I_G2 = SI_model(adj=adj, prob=prob, k=infec, t=time)
        base.append(np.abs(np.average(Hamming(I_G, I_G2))))
    base = np.average(base)
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = SSEdge(adj, R, r)
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        ss_avg.append(np.average(L))
        ss_mx.append(np.max(L))
        ss_mn.append(np.min(L))
        print("1", r)
    for i in range(len(y_list)):
        r = y_list[i]
        L = []
        infec = ran.randint(0, n - 1)
        I_G = SI_model(adj=adj, prob=prob, k=infec, t=time)
        for j in range(k):
            H = UniEdge(adj, r)
            for m in range(t):
                I_H = SI_model(adj=H, prob=prob, k=infec, t=time)
                L.append(np.abs(np.average(Hamming(I_G, I_H))))
        un_mx.append(np.max(L))
        un_mn.append(np.min(L))
        un_avg.append(np.average(L))
        print("2", r)
    return [ss_mx, ss_mn, ss_avg], [un_mx, un_mn, un_avg], y_list, base


flat = lambda l: [item for sublist in l for item in sublist]


def SISimVI(adj, prob, R, k, dim="all"):
    if dim == "all":
        E = Mtrx_Elist(adj)[0]
        n = len(adj)
        min_q = int(np.ceil(0.22 * len(E)))
        max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
        z_list_1 = []
        z_list_2 = []
        z_list_3 = []
        y_list = list(range(min_q, len(E), 100))
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            I_G = [flat(I_G)]
            for j in range(k):
                H = Adapt1(adj, r, R, 0.2)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                I_H = [flat(I_H)]
                L.append(variation_of_information(I_G, I_H))
            z_list_1.append(np.average(L))
            print("1", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            I_G = [flat(I_G)]
            for j in range(k):
                H = Adapt1(adj, r, R, 1)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                I_H = [flat(I_H)]
                L.append(variation_of_information(I_G, I_H))
            z_list_2.append(np.average(L))
            print("2", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            I_G = [flat(I_G)]
            for j in range(k):
                H = Adapt1(adj, r, R, 100)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                I_H = [flat(I_H)]
                L.append(variation_of_information(I_G, I_H))
            z_list_3.append(np.average(L))
            print("3", r)
        y_list.pop(0)
        z_list_1.pop(0)
        z_list_2.pop(0)
        z_list_3.pop(0)
    if dim == "column":
        E = Mtrx_Elist(adj)[0]
        n = len(adj)
        min_q = int(np.ceil(0.22 * len(E)))
        max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
        z_list_1 = []
        z_list_2 = []
        z_list_3 = []
        y_list = list(range(min_q, len(E), 100))
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = np.transpose(I_G)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 0.2)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = np.transpose(I_H)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_1.append(np.average(L))
            print("1", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = np.transpose(I_G)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 1)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = np.transpose(I_H)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_2.append(np.average(L))
            print("2", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = np.transpose(I_G)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 100)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = np.transpose(I_H)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_3.append(np.average(L))
            print("3", r)
        y_list.pop(0)
        z_list_1.pop(0)
        z_list_2.pop(0)
        z_list_3.pop(0)
    if dim == "row":
        E = Mtrx_Elist(adj)[0]
        n = len(adj)
        min_q = int(np.ceil(0.22 * len(E)))
        max_q = int(np.floor((24 * n * np.log10(n)) / (0.1 ** 2)))
        z_list_1 = []
        z_list_2 = []
        z_list_3 = []
        y_list = list(range(min_q, len(E), 100))
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 0.2)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_1.append(np.average(L))
            print("1", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 1)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_2.append(np.average(L))
            print("2", r)
        for i in range(len(y_list)):
            r = y_list[i]
            L = []
            infec = ran.randint(0, n - 1)
            I_G = SI_model(adj=adj, prob=prob, k=infec, t=50)
            I_G = I_G.tolist()
            for j in range(k):
                H = Adapt1(adj, r, R, 100)
                I_H = SI_model(adj=H, prob=prob, k=infec, t=50)
                I_H = I_H.tolist()
                L.append(variation_of_information(I_G, I_H))
            z_list_3.append(np.average(L))
            print("3", r)
    y_list.pop(0)
    z_list_1.pop(0)
    z_list_2.pop(0)
    z_list_3.pop(0)
    return z_list_1, z_list_2, z_list_3, y_list

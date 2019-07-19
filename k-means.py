import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import randrange
from random import choice

players = pd.read_excel("C:/Users/Jack/Desktop/NHL_players.xlsx")
# print(players)
#
# plt.plot(players.G, players.A, "o")
# plt.title("NHL players")
# plt.xlabel("Goals")
# plt.ylabel("Assists")
# plt.show()

GA = np.array([players.G, players.A])
GA = np.transpose(GA)

# this function randomly selects K points as an initial guess for cluster centroids


def RandomInit(K, Points):
    init_centers = np.zeros((K, len(Points[0, :])))
    s = ([*range(0, len(Points))])
    for i in range(0, K):
        val = choice(s)
        init_centers[i] = Points[val, :]
        s.remove(val)
    return init_centers


def findClosestCentroid(Points, centroids):
    K = len(centroids)
    m = len(Points[:, 0])
    n = len(Points[0, :])
    idx = np.zeros(len(Points))
    for i in range(0, m):
        min_val = 1000000
        l = np.zeros(K)
        for k in range(0, K):
            l[k] = np.sqrt(np.sum((Points[i, :] - centroids[k, :])**2))
            if l[k] < min_val:
                idx[i] = k
                min_val = l[k]
    return idx


def computeCentroids(Points, idx, K):
    centroids = np.zeros((K, np.size(Points, axis=1)))
    for k in range(0, K):
        bool_ind = (idx == k)
        # print(bool_ind)
        temp = Points[idx == k]
        centroids[k, :] = np.sum(temp, axis=0)/len(temp)
        # print(Points[idx == k])
    return centroids


def kmeans(Points, K):
    centroids = RandomInit(K, Points)
    isSame = False
    while not isSame:
        idx = findClosestCentroid(Points, centroids)
        centroids = computeCentroids(Points, idx, K)
        if np.all(findClosestCentroid(Points, centroids) == idx):
            isSame = True
    return idx


# init_c = RandomInit(2, GA)
# closest_p = findClosestCentroid(GA, init_c)
# print(RandomInit(2, GA))
# print(closest_p)
# print(computeCentroids(GA, closest_p, 2))

# print(kmeans(GA, 2))

testG = [21, 14, 15, 36]
testA = [28, 25, 45, 26]
res = kmeans(GA, 2)
for i in range(0, len(GA)):
    if res[i] == 0:
        plt.plot(players.G[i], players.A[i], "o", color='g')
    else:
        plt.plot(players.G[i], players.A[i], "x", color='b')


# plt.plot(players.G, players.A, "o")
# plt.plot(testG, testA, "x")
plt.title("NHL players")
plt.xlabel("Goals")
plt.ylabel("Assists")
plt.show()



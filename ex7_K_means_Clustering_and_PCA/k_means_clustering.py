import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random


def find_closet_centroids(X, centroids):
    """
    寻找所属簇
    :param X: ndarray,所有点
    :param centroids: ndarray,上一步计算出或初始化的簇中心
    :return: ndarray,每个点所属于的簇
    """
    res = np.zeros((1,))
    for x in X:
        res = np.append(res, np.argmin(np.sqrt(np.sum((centroids - x) ** 2, axis=1))))
    return res[1:]


def compute_centroids(X, idx):
    """
    计算新的簇中心
    :param X: ndarray,所有点
    :param idx: ndarray,每个点对应的簇号
    :return: ndarray,所有新簇中心
    """
    K = int(np.max(idx)) + 1
    m = X.shape[0]
    n = X.shape[-1]
    centroids = np.zeros((K, n))
    counts = np.zeros((K, n))
    for i in range(m):
        centroids[int(idx[i])] += X[i]
        counts[int(idx[i])] += 1
    centroids = centroids / counts
    return centroids


def random_initialization(X, K):
    """
    随机选择K组数据，作为簇中心
    :param X: ndarray,所有点
    :param K: int,聚类的类数
    :return: ndarray,簇中心
    """
    res = np.zeros((1, X.shape[-1]))
    m = X.shape[0]
    rl = []
    while True:
        index = random.randint(0, m)
        if index not in rl:
            rl.append(index)
        if len(rl) >= K:
            break
    for index in rl:
        res = np.concatenate((res, X[index].reshape(1, -1)), axis=0)
    return res[1:]


def cost(X, idx, centrodis):
    c = 0
    for i in range(len(X)):
        c += np.sum((X[i] - centrodis[int(idx[i])]) ** 2)
    c /= len(X)
    return c


def k_means(X, K):
    """
    k-means聚类算法
    :param X: ndarray,所有的数据
    :param K: int,聚类的类数
    :return: tuple,(idx, centroids_all)
                idx,ndarray为每个数据所属类标签
                centroids_all,[ndarray,...]计算过程中每轮的簇中心
    """
    centroids = random_initialization(X, K)
    centroids_all = [centroids]
    idx = np.zeros((1,))
    last_c = -1
    now_c = -2
    # iterations = 200
    # for i in range(iterations):
    while now_c != last_c:  # 当收敛时结束算法，或者可以利用指定迭代轮数
        idx = find_closet_centroids(X, centroids)
        last_c = now_c
        now_c = cost(X, idx, centroids)
        centroids = compute_centroids(X, idx)
        centroids_all.append(centroids)
    return idx, centroids_all


def visualizing(X, idx, centroids_all):
    """
    可视化聚类结果和簇中心的移动过程
    :param X: ndarray,所有的数据
    :param idx: ndarray,每个数据所属类标签
    :param centroids_all: [ndarray,...]计算过程中每轮的簇中心
    :return: None
    """
    plt.scatter(X[..., 0], X[..., 1], c=idx)
    xx = []
    yy = []
    for c in centroids_all:
        xx.append(c[..., 0])
        yy.append(c[..., 1])
    plt.plot(xx, yy, 'rx--')
    plt.show()


if __name__ == "__main__":
    data = sio.loadmat("data\\ex7data2.mat")
    X = data['X']  # (300,2)
    init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closet_centroids(X, init_centroids)
    print(idx[0:3])
    print(compute_centroids(X, idx))
    idx, centroids_all = k_means(X, 3)
    visualizing(X, idx, centroids_all)

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def data_preprocess(X):
    """
    数据归一化
    :param X: ndarray,原始数据
    :return: (ndarray.ndarray,ndarray),处理后的数据,每个特征均值，每个特征方差
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)  # 默认ddof=0, 这里一定要修改
    return (X - mean) / std, mean, std


def pca(X):
    sigma = X.T.dot(X) / len(X)  # (n,m)x(m,n) (n,n)
    u, s, v = np.linalg.svd(sigma)  # u(n,n) s(n,), v(n,n)
    return u, s, v


# X(m,n), U(n,n)
def project_data(X, U, K):
    """
    数据降维
    :param X: ndarray,原始数据
    :param U: ndarray,奇异值分解后的U
    :param K: int,目标维度
    :return: ndarray,降维后的数据
    """
    return X.dot(U[..., :K])


# Z(m,K), U(n,n)
def reconstruct_data(Z, U, K):
    """
    数据升维
    :param Z: ndarray,降维后的数据
    :param U: ndarray,奇异值分解后的U
    :param K: int,降维的维度
    :return: ndarray,原始数据
    """
    return Z.dot(U[..., :K].T)


if __name__ == "__main__":
    data = sio.loadmat("data\\ex7data1.mat")
    X = data['X']  # (50,2)
    normalized_X, _, _ = data_preprocess(X)
    u, _, _ = pca(normalized_X)  # (2,2)
    Z = project_data(normalized_X, u, 1)
    print(Z[0])
    rec_X = reconstruct_data(Z, u, 1)
    print(rec_X[0])
    plt.scatter(normalized_X[..., 0], normalized_X[..., 1], marker='x', c='b', label='normalized x')
    plt.scatter(rec_X[..., 0], rec_X[..., 1], marker='x', c='r', label='reconstructed x')
    plt.title("Visualizing the projections")
    for i in range(len(normalized_X)):
        plt.plot([normalized_X[i][0], rec_X[i][0]], [normalized_X[i][1], rec_X[i][1]], 'k--')
    plt.xlim((-3, 2))
    plt.ylim((-3, 2))
    plt.legend()
    plt.show()

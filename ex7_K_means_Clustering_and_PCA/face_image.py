import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import ex7_K_means_Clustering_and_PCA.PCA as pca


def visualizing_images(X, d):
    """
    可视化图片
    :param X: ndarray,图片
    :param d: int,一行展示多少张图片
    :return: None
    """
    m = len(X)
    n = X.shape[-1]
    s = int(np.sqrt(n))
    for i in range(1, m + 1):
        plt.subplot(m / d, d, i)
        plt.axis('off')
        plt.imshow(X[i - 1].reshape(s, s).T, cmap='Greys_r')  # 要把脸摆正需要转置
    plt.show()


if __name__ == "__main__":
    data = sio.loadmat("data\\ex7faces.mat")
    X = data['X']  # (5000,1024)
    visualizing_images(X[:25], 5)
    nor_X, _, _ = pca.data_preprocess(X)
    u, _, _ = pca.pca(nor_X)
    Z = pca.project_data(nor_X, u, 36)
    rec_X = pca.reconstruct_data(Z, u, 36)
    visualizing_images(rec_X[:25], 5)

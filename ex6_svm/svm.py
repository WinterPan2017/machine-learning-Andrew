import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


def plot_scatter(x1, x2, y):
    """
    绘制散点图
    :param x1: ndarray,横坐标数据
    :param x2: ndarray,纵坐标数据
    :param y: ndarray,标签
    :return: None
    """
    plt.scatter(x1, x2, c=y.flatten())
    plt.xlabel("x1")
    plt.ylabel("X2")


def plot_boundary(model, X, title):
    """
    绘制决策边界
    :param model: <class 'sklearn.svm._classes.SVC'>,训练好的模型
    :param X: ndarray,训练数据
    :param title: str,图片的题目
    :return: None
    """
    x_max, x_min = np.max(X[..., 0]), np.min(X[..., 0])
    y_max, y_min = np.max(X[..., 1]), np.min(X[..., 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    p = model.predict(np.concatenate((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)), axis=1))
    plt.contour(xx, yy, p.reshape(xx.shape))
    plt.title(title)


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum(np.power(x1 - x2, 2)) / (2 * sigma ** 2))


if __name__ == "__main__":
    data1 = sio.loadmat("data\\ex6data1.mat")
    X = data1["X"]
    y = data1["y"]
    plot_scatter(X[..., 0], X[..., 1], y)
    model = svm.SVC(C=100, kernel='linear')
    model.fit(X, y.ravel())
    plot_boundary(model, X, "SVM Decision Boundary with C = 100 (Example Dataset 1)")
    plt.show()
    print(gaussian_kernel(np.array([1, 2, 1]), np.array([0, 4, -1]), 2.))

    data2 = sio.loadmat("data\\ex6data2.mat")
    X = data2['X']
    y = data2['y']
    sigma = 0.1
    gamma = 1 / (2 * np.power(sigma, 2))
    plot_scatter(X[..., 0], X[..., 1], y)
    model = svm.SVC(C=1, kernel='rbf', gamma=gamma)
    model.fit(X, y.ravel())
    plot_boundary(model, X, "SVM (Gaussian Kernel) Decision Boundary (Example Dataset 2)")
    plt.show()

    data3 = sio.loadmat("data\\ex6data3.mat")
    X = data3['X']
    y = data3['y']
    Xval = data3['Xval']
    yval = data3['yval']
    xx, yy = np.meshgrid(np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]), np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]))
    parameters = np.concatenate((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)), axis=1)
    score = np.zeros(1)
    for C, sigma in parameters:
        gamma = 1 / (2 * np.power(sigma, 2))
        model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(X, y.ravel())
        score = np.append(score, model.score(Xval, yval.ravel()))
    res = np.concatenate((parameters, score[1:].reshape(-1, 1)), axis=1)
    index = np.argmax(res, axis=0)[-1]
    print("the best choice of parameters:C=", res[index][0], ",sigma=", res[index][1], ",score=", res[index][2])
    C = res[index][0]
    sigma = res[index][1]
    gamma = 1 / (2 * np.power(sigma, 2))
    plot_scatter(X[..., 0], X[..., 1], y)
    model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(X, y.ravel())
    plot_boundary(model, X, "SVM (Gaussian Kernel) Decision Boundary (Example Dataset 3)")
    plt.show()

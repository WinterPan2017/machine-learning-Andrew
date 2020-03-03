import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset(X):
    plt.scatter(X[..., 0], X[..., 1], marker='x', label='point')


def gaussian_distribution(X, mu, sigma2):
    """
    根据高斯模型参数，计算概率
    :param X: ndarray,数据
    :param mu: ndarray,均值
    :param sigma2: ndarray,方差
    :return: ndarray,概率
    """
    p = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(X - mu) ** 2 / (2 * sigma2))
    return np.prod(p, axis=1)  # 横向累乘


def estimate_parameters_for_gaussian_distribution(X):
    """
    估计数据估计参数
    :param X: ndarray,数据
    :return: (ndarray,ndarray),均值和方差
    """
    mu = np.mean(X, axis=0)  # 计算方向因该是沿着0，遍历每组数据
    sigma2 = np.var(X, axis=0)  # N-ddof为除数,ddof默认为0
    return mu, sigma2


def visualize_contours(mu, sigma2):
    """
    画出高斯分布的等高线
    :param mu: ndarray,均值
    :param sigma2: ndarray,方差
    :return: None
    """
    x = np.linspace(5, 25, 100)
    y = np.linspace(5, 25, 100)
    xx, yy = np.meshgrid(x, y)
    X = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    z = gaussian_distribution(X, mu, sigma2).reshape(xx.shape)
    cont_levels = [10 ** h for h in range(-20, 0, 3)]  # 当z为当前列表的值时才绘出等高线
    plt.contour(xx, yy, z, cont_levels)


# yp预测的y,yt真实的y,两个参数都为(m,)
def error_analysis(yp, yt):
    """
    计算误差分析值F1-score
    :param yp: ndarray,预测值
    :param yt: ndarray,实际值
    :return: float,F1-score
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(yp)):
        if yp[i] == yt[i]:
            if yp[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if yp[i] == 1:
                fp += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0  # 防止除以0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return f1


# 传入的两个参数都为(m,)
def select_threshold(yval, pval):
    """
    根据预测值和真实值确定最好的阈值
    :param yval: ndarray,真实值（这里是0或1）
    :param pval: ndarray,预测值（这里是[0,1]的概率）
    :return: (float,float),阈值和F1-score
    """
    epsilons = np.linspace(min(pval), max(pval), 1000)
    l = np.zeros((1, 2))
    for e in epsilons:
        ypre = (pval < e).astype(float)
        f1 = error_analysis(ypre, yval)
        l = np.concatenate((l, np.array([[e, f1]])), axis=0)
    index = np.argmax(l[..., 1])
    return l[index, 0], l[index, 1]


def detection(X, e, mu, sigma2):
    """
    根据高斯模型检测出异常数据
    :param X: ndarray,需要检查的数据
    :param e: float,阈值
    :param mu: ndarray,均值
    :param sigma2: ndarray,方差
    :return: ndarray,异常数据
    """
    p = gaussian_distribution(X, mu, sigma2)
    anomaly_points = np.array([X[i] for i in range(len(p)) if p[i] < e])
    return anomaly_points


def circle_anomaly_points(X):
    plt.scatter(X[..., 0], X[..., 1], s=80, facecolors='none', edgecolors='r', label='anomaly point')


if __name__ == "__main__":
    data = sio.loadmat("data\\ex8data1.mat")
    X = data['X']  # (307,2)
    visualize_dataset(X)
    mu, sigma2 = estimate_parameters_for_gaussian_distribution(X)
    p = gaussian_distribution(X, mu, sigma2)
    visualize_contours(mu, sigma2)

    Xval = data['Xval']  # (307,2)
    yval = data['yval']  # (307,1)
    e, f1 = select_threshold(yval.ravel(), gaussian_distribution(Xval, mu, sigma2))
    print('best choice of epsilon is ', e, ',the F1 score is ', f1)
    anomaly_points = detection(X, e, mu, sigma2)
    circle_anomaly_points(anomaly_points)
    plt.title('anomaly detection')
    plt.legend()
    plt.show()

    # High dimensional dataset
    data2 = sio.loadmat("data\\ex8data2.mat")
    X = data2['X']
    Xval = data2['Xval']
    yval = data2['yval']
    mu, sigma2 = estimate_parameters_for_gaussian_distribution(X)
    e, f1 = select_threshold(yval.ravel(), gaussian_distribution(Xval, mu, sigma2))
    anomaly_points = detection(X, e, mu, sigma2)
    print('\n\nfor this high dimensional dataset \nbest choice of epsilon is ', e, ',the F1 score is ', f1)
    print('the number of anomaly points is', anomaly_points.shape[0])
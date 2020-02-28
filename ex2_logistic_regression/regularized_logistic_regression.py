from ex2_logistic_regression.logistic_regression import read_dataset, separate_dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, l):
    m = X.shape[0]
    part1 = np.mean(-y * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))
    part2 = (l / (2 * m)) * np.sum(np.delete((theta * theta), 0, axis=0))
    return part1 + part2


def gradient(theta, X, y, l):
    m = X.shape[0]
    part1 = X.T.dot((sigmoid(X.dot(theta)) - y)) / m
    part2 = (l / m) * theta
    part2[0] = 0
    return part1 + part2


def predict(theta, X):
    return [1 if i > 0.5 else 0 for i in sigmoid(X.dot(theta))]


def features_mapping(x1, x2, power):
    m = len(x1)
    features = np.zeros((m, 1))
    for sum_power in range(power):
        for x1_power in range(sum_power + 1):
            x2_power = sum_power - x1_power
            features = np.concatenate(
                (features, (np.power(x1, x1_power) * np.power(x2, x2_power)).reshape(m, 1)),
                axis=1)
    return np.delete(features, 0, axis=1)


if __name__ == "__main__":
    # 画出散点图
    data = read_dataset("ex2data2.txt", (float, float, float), separator=',')
    data0, data1 = separate_dataset(data, -1, 0.5)
    plt.subplot(1, 1, 1)
    test1 = plt.scatter(data0[..., 0], data0[..., 1], marker='x', c='b', label='reject')
    test2 = plt.scatter(data1[..., 0], data1[..., 1], marker='x', c='y', label='accepted')
    # plt.legend(handles=[test1, test2], loc='upper right')
    plt.xlim((-1, 1.2))
    plt.ylim((-1, 1.2))
    plt.xlabel('microchips test 1')
    plt.xlabel('microchips test 2')
    plt.title('Plot of training data')

    # 特征映射
    features = features_mapping(data[..., 0], data[..., 1], 6)

    y = data[..., -1]
    theta = np.zeros(features.shape[-1])
    # 测试损失函数
    print(cost(theta, features, y, 1))

    # 优化
    res = opt.minimize(fun=cost, x0=theta, args=(features, y, 1), method='TNC', jac=gradient)
    print(classification_report(y, predict(res.x, features)))

    # 画出决策边界
    x = np.linspace(-1, 1.2, 100)
    x1, x2 = np.meshgrid(x, x)
    z = features_mapping(x1.ravel(), x2.ravel(), 6)
    z = z.dot(res.x).reshape(x1.shape)
    db = plt.contour(x1, x2, z, 0, colors=['r'])
    plt.legend(loc='upper right')
    plt.show()

    # 自行修改lambda的值去观察，完成额外的练习

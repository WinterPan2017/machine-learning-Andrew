import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


def read_dataset(filename, type_tuple, separator=','):
    """
    从文件中读入数据，文件的数据存储应该是每组数据存在一行并用分隔符隔开
    返回：ndarray
    eg:
        1.1,2.1,3.1
        1.2,2.2,3.2

    parameters:
    ----------
    filename : str
            (包括路径的）文件名
    type_tuple : tuple
            每一行数据的类型
    separator : str
            分隔符，默认为','
    """
    f = open(filename, 'r')
    lines = f.readlines()

    data = []
    if len(type_tuple) != len(lines[0]) and len(type_tuple) == 1:
        for line in lines:
            line = line[:-1]
            line = line.split(sep=separator)
            row = []
            for col in line:
                row.append(type_tuple[0](col))
            data.append(row)

    elif len(type_tuple) == len(lines[0].split(sep=separator)):
        for line in lines:
            line = line[:-1]
            line = line.split(sep=separator)
            row = []
            for i in range(len(line)):
                row.append(type_tuple[i](line[i]))
            data.append(row)
    else:
        data = None
    return np.array(data)


def separate_dataset(data, col, boundary):
    """
    将数据按照某列进行二分类

    parameters:
    ----------
    data : ndarray
            一组数据存在一行
    col : int
            分类标准应用到的列号
    boundary : double
            分类边界
    """
    data0 = np.array(data)
    data1 = np.array(data)
    dc0 = 0
    dc1 = 0
    for i in range(data.shape[0]):
        if data[i][col] < boundary:
            data1 = np.delete(data1, i - dc1, axis=0)
            dc1 += 1
        else:
            data0 = np.delete(data0, i - dc0, axis=0)
            dc0 += 1
    return data0, data1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    return np.mean((-y) * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))


def gradient(theta, X, y):
    return X.T.dot(sigmoid(X.dot(theta)) - y) / X.shape[0]


# 普通的梯度下降实现
def gradient_descent(theta, X, y, alpha, iterations):
    for i in range(iterations):
        theta -= alpha * gradient(theta, X, y)
    return theta


def predict(theta, X):
    return [1 if i > 0.6 else 0 for i in sigmoid(X.dot(theta))]


if __name__ == "__main__":
    # 读取数据画出散点图
    data = read_dataset("ex2data1.txt", (float, float, float), separator=',')
    data0, data1 = separate_dataset(data, -1, 0.5)
    # plt.subplot(2, 2, 1)
    plt.title("raw data scatter")
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.xlim((20, 110))
    plt.ylim((20, 110))
    na = plt.scatter(data0[..., 0], data0[..., 1], marker='x', c='b', label='not admitted')
    a = plt.scatter(data1[..., 0], data1[..., 1], marker='x', c='y', label='admitted')

    # 测试损失函数
    data = np.array(data)
    X = np.insert(data[..., :2], 0, 1, axis=1)  # 记得添加x0
    y = data[..., -1]
    theta = np.zeros((3,))
    # print(cost(theta, X, y))  # 0.6931471805599453

    # 特征归一化
    mean = np.mean(X[..., 1:], axis=0)
    std = np.std(X[..., 1:], axis=0, ddof=1)
    X[..., 1:] = (X[..., 1:] - mean) / std

    # 传统梯度下降
    alpha = 0.2
    iterations = 10000
    theta = gradient_descent(theta, X, y, alpha, iterations)
    print("使用梯度下降,最后的Theta:", theta)

    # # 使用scipy中的高级优化算法
    # res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
    # theta = res.x
    # print("使用scipy.optimize.minimize,最后的Theta:", res.x)

    # 画出决策边界
    # plt.subplot(2, 2, 2)
    x1 = np.arange(20, 110, 0.1)
    # 因为进行了特征缩放，所以计算y时需要还原特征缩放
    x2 = mean[1] - std[1] * (theta[0] + theta[1] * (x1 - mean[0]) / std[0]) / theta[2]
    db = plt.plot(x1, x2, c='r', label="decision boundary")
    plt.legend(loc="upper right")
    plt.show()

    # 测试优化结果
    test_x = np.array([45, 85])
    test_x = (test_x - mean) / std
    test_x = np.insert(test_x, 0, 1)
    print(sigmoid(test_x.dot(theta)))  # 0.7763928918272246
    # 评价
    print(classification_report(y, predict(theta, X)))

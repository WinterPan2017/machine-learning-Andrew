import scipy.io as sio
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from ex3_neural_network.Multi_class_Classification import mapping,convert


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def serialize(thetas):
    """
    将多个参数多维数组，映射到一个向量上
    :param thetas: tuple or list,按顺序存储每层的theta参数，每个为ndarray
    :return: ndarray，一维化的参数向量
    """
    res = np.array([0])
    for t in thetas:
        res = np.concatenate((res, t.ravel()), axis=0)
    return res[1:]


def deserialize(theta):
    """
    将向量还原为多个参数(只适用当前网络)
    :param theta: ndarray,一维化的参数向量
    :return: tuple ,按顺序存储每层的theta参数，每个为ndarray
    """
    return theta[:25 * 401].reshape(25, 401), theta[25 * 401:].reshape(10, 26)


def not_regularized_cost(thetas, X, y):
    """
    计算非正则化的损失值
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :param y: ndarray,数据的标记
    :return: float,损失值
    """
    for t in deserialize(thetas):
        X = np.insert(X, 0, 1, axis=1)
        X = X.dot(t.T)
        X = sigmoid(X)
    return np.mean(np.sum((-y) * np.log(X) - (1 - y) * np.log(1 - X), axis=1))


def regularized_cost(theta, X, y, l):
    """
    计算正则化的损失值
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :param y: ndarray,数据的标记
    :param l: float,惩罚参数
    :return: float,损失值
    """
    m = X.shape[0]
    part2 = 0.0
    for t in deserialize(theta):
        X = np.insert(X, 0, 1, axis=1)
        X = X.dot(t.T)
        X = sigmoid(X)
        t = t[..., 1:]  # 要去掉bias unit
        part2 += (l / (2 * m)) * np.sum(t * t)
    part1 = np.mean(np.sum((-y) * np.log(X) - (1 - y) * np.log(1 - X), axis=1))
    return part1 + part2


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def random_initialize_weights(shape, e=0.12):
    """
    随机初始化参数，范围为[-e, e]
    :param shape: tuple or list，需要初始化的参数的规格
    :param e: float,边界
    :return: ndarray,参数矩阵
    """
    return (np.random.rand(shape[0], shape[1]) - 0.5) * 2 * e


def feedforward(thetas, X):
    """
    前向传播
    :param thetas: ndarray，一维参数向量
    :param X: 输入层的输入值
    :return: 所有的a组成的列表，所有的Z组成的列表
    """
    A = []
    Z = []
    a = X
    for t in deserialize(thetas):
        a = np.insert(a, 0, 1, axis=1)
        A.append(a)
        z = a.dot(t.T)
        Z.append(z)
        a = sigmoid(z)
    A.append(a)
    # 返回：[a1, a2, a3], [z2, z3]
    return A, Z


def back(theta, X, y, l):
    """
    反向传播算法
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :param y: ndarray,数据的标签
    :param l: float,惩罚参数
    :return: ndarray，下降后一维参数向量
    """
    A, Z = feedforward(theta, X)
    a1, a2, a3 = A  # a1(5000,401), a2(5000,26), a3(5000,10)
    z2, z3 = Z  # z2(5000,25), z3(5000,10)
    theta1, theta2 = deserialize(theta)  # theta1(25,401), theta2(10,26)
    m = X.shape[0]

    d3 = a3 - y  # d3(5000,10)
    d2 = d3.dot(theta2)[..., 1:] * sigmoid_gradient(z2)  # d2(5000,25)

    theta1 = np.insert(np.delete(theta1, 0, axis=1), 0, 0, axis=1)
    theta2 = np.insert(np.delete(theta2, 0, axis=1), 0, 0, axis=1)
    D1 = (1 / m) * d2.T.dot(a1) + (l / m) * theta1  # D1(25,401)
    D2 = (1 / m) * d3.T.dot(a2) + (l / m) * theta2  # D2(10,26)
    return serialize((D1, D2))


def gradient_checking(theta, X, y, l, e=10 ** -4):
    """
    检测反向传播算法是否正确运行
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :param y: ndarray,数据的标签
    :param l: float,惩罚参数
    :param e: float，微小扰动
    :return: ndarray，下降后一维参数向量
    """
    res = np.zeros(theta.shape)
    for i in range(len(theta)):
        left = np.array(theta)
        left[i] -= e
        right = np.array(theta)
        right[i] += e
        gradient = (regularized_cost(right, X, y, l) - regularized_cost(left, X, y, l)) / (2 * e)
        res[i] = gradient
    return res


def predict(theta, X):
    """
    利用训练好的参数进行预测
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :return: ndarray,网络计算结果，分类结果
    """
    a3 = feedforward(theta, X)[0][-1]
    p = np.zeros((1, 10))
    for i in a3:
        index = np.argmax(i)
        temp = np.zeros((1, 10))
        temp[0][index] = 1
        p = np.concatenate((p, temp), axis=0)
    return p[1:]


def visualizing_the_hidden_layer(theta, X):
    """
    可视化显示隐藏层的输入和输出
    :param theta: ndarray，一维参数向量
    :param X: ndarray,输入层的输入值
    :return: None
    """
    A, _ = feedforward(theta, X)
    a1, a2, a3 = A
    # 要去掉bias unit
    input = a1[..., 1:][:25]
    output = a2[..., 1:][:25]
    input = mapping(input, 5)
    output = mapping(output, 5)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(input.T)
    plt.title("hidden layer input")

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(output)
    plt.title("hidden layer output")
    plt.show()


if __name__ == "__main__":
    data = sio.loadmat("..\\ex3_neural_network\\ex3data1.mat")
    theta = sio.loadmat("..\\ex3_neural_network\\ex3weights.mat")
    X = data['X']
    y = convert(data['y'])
    # 这里用于训练的训练集对y的处理是
    # 1 2 3 ... 0
    # [0,0,0,...,0]
    # 而convert处理中时
    # 0 1 2 ... 9
    # [0,0,0,...,0]
    # 因此需要转换
    y0 = y[..., 0].reshape(y.shape[0], 1)
    y = np.concatenate((y[..., 1:], y0), axis=1)
    theta1 = theta['Theta1']
    theta2 = theta['Theta2']
    theta = serialize((theta1, theta2))
    print(X.shape, y.shape, theta1.shape, theta2.shape)
    a1 = X  # (5000,400)
    a1 = np.insert(a1, 0, 1, axis=1)  # (5000,401)
    z2 = a1.dot(theta1.T)  # (5000,25)
    a2 = sigmoid(z2)  # (5000,25)
    a2 = np.insert(a2, 0, 1, axis=1)  # (5000,26)
    z3 = a2.dot(theta2.T)  # (5000,10)
    a3 = sigmoid(z3)  # (5000,10)
    cost = np.mean(np.sum((-y) * np.log(a3) - (1 - y) * np.log(1 - a3), axis=1))
    print(cost)
    print("not_regularized_cost", not_regularized_cost(theta, X, y))
    print("regularized_cost", regularized_cost(theta, X, y, 1))
    print(sigmoid_gradient(0))
    print(random_initialize_weights((2, 2)))
    # print(back(theta, X, y, 1).shape, gradient_checking(theta, X, y, 1).shape)

    # 以上是为了测试写得是否正确，所以按照提供数据更改了y，下面我们将使用最自然的y表示方式
    y = convert(data['y'])
    theta1 = random_initialize_weights((25, 401))
    theta2 = random_initialize_weights((10, 26))
    theta = serialize((theta1, theta2))
    # 参数初始化得不同，有时会导致溢出
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, 1), method="TNC", jac=back)
    print(res)
    theta1, theta2 = deserialize(res.x)
    sio.savemat("parametersWeights.mat", {"theta1": theta1, "theta2": theta2})
    print(classification_report(y, predict(res.x, X)))
    visualizing_the_hidden_layer(res.x, X)

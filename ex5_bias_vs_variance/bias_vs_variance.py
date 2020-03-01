import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def cost(theta, X, y, l):
    m = X.shape[0]
    part1 = np.mean(np.power(X.dot(theta) - y.ravel(), 2)) / 2
    part2 = (l / (2 * m)) * np.sum(np.delete(theta * theta, 0, axis=0))
    return part1 + part2


def gradient(theta, X, y, l):
    m = X.shape[0]
    part1 = X.T.dot(X.dot(theta) - y.ravel()) / m
    part2 = (l / m) * theta
    part2[0] = 0
    return part1 + part2


def poly_features(X, power_max):
    """
    添加多次项，增加特征
    :param X: ndarray,原始特征
    :param power_max: int,最高次
    :return: ndarray,增加特征后的特征
    """
    _X = X.reshape(-1, 1)
    res = np.ones((X.shape[0], 1))
    for i in range(power_max):
        res = np.concatenate((res, np.power(_X, i + 1)), axis=1)
    return res[..., 1:]


def normalize_features(X, means, stds):
    return (X - means) / stds


def randomly_select(data, n):
    """
    从数据集中随机取出n组
    :param data: ndarray,数据
    :param n: int,选择数量
    :return: ndarray,随机选择的数据
    """
    res = np.array(data)
    m = data.shape[0]
    for i in range(m - n):
        index = np.random.randint(0, res.shape[0] - 1)
        res = np.delete(res, index, axis=0)
    return res


if __name__ == "__main__":
    data = sio.loadmat("ex5data1.mat")
    X = data["X"]
    y = data["y"]
    Xval = data["Xval"]
    yval = data["yval"]
    Xtest = data["Xtest"]
    ytest = data["ytest"]
    # 可视化训练集
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.title("linear regression")
    plt.xlim((-50, 40))
    plt.ylim((-10, 40))

    # 线性拟合
    theta = np.ones((2,))
    X = np.insert(X, 0, 1, axis=1)
    print(cost(theta, X, y, 1))
    print(gradient(theta, X, y, 1))
    res = opt.minimize(fun=cost, x0=theta, args=(X, y, 0), method="TNC", jac=gradient)
    plt.plot([i for i in range(-50, 40, 1)], [res.x[0] + res.x[1] * i for i in range(-50, 40, 1)])

    # 画出learning curve
    Xval = np.insert(Xval, 0, 1, axis=1)  # 为了计算误差
    error_train = []
    error_validation = []
    for i in range(X.shape[0]):
        subX = X[:i + 1]
        suby = y[:i + 1]
        res = opt.minimize(fun=cost, x0=theta, args=(subX, suby, 1), method="TNC", jac=gradient)
        t = res.x
        error_train.append(cost(t, subX, suby, 0))
        error_validation.append(cost(t, Xval, yval, 0))
    plt.subplot(2, 2, 2)
    plt.plot([i for i in range(X.shape[0])], error_train, label="training set error")
    plt.plot([i for i in range(X.shape[0])], error_validation, label="cross validation set error")
    plt.legend(loc="upper right")
    plt.xlabel("m(numbers of training set)")
    plt.title("learning curves")

    # 多项式拟合
    # 若选取作业上的8，由于优化方法不同，会导致曲线不同
    power_max = 6
    l = 0  # 参数lambda, 取0时优化报错但不影响使用（过拟合），取100（欠拟合）
    features = poly_features(data['X'], power_max)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0, ddof=1)
    normalized_features = normalize_features(features, means, stds)
    normalized_X = np.insert(normalized_features, 0, 1, axis=1)
    res = opt.minimize(fun=cost, x0=np.ones((power_max + 1,)), args=(normalized_X, y, l), method="TNC", jac=gradient)
    plt.subplot(2, 2, 3)
    plt.scatter(data['X'], y, marker='x', c='r')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.title("polynomial(8) regression")
    plt.xlim((-100, 100))
    plt.ylim((-10, 40))
    X = np.linspace(-100, 100, 50)
    normalized_X = normalize_features(poly_features(X, power_max), means, stds)
    normalized_X = np.insert(normalized_X, 0, 1, axis=1)
    plt.plot(X, normalized_X.dot(res.x))

    # 训练集变化对应的learn curve
    error_train = []
    error_validation = []
    # 注意坑！！！
    # 这里需要直接利用全部训练集的归一化参数，直接将训练集和验证集数据全部归一化，以后直接在里面取即可
    # 而不是对原始训练集取后，重新选择归一化参数
    train_features = poly_features(data["X"], power_max)
    train_normalized_features = normalize_features(train_features, means, stds)
    train_normalized_X = np.insert(train_normalized_features, 0, 1, axis=1)
    val_features = poly_features(data["Xval"], power_max)
    val_normalized_features = normalize_features(val_features, means, stds)
    val_normalized_X = np.insert(val_normalized_features, 0, 1, axis=1)
    yval = data["yval"]
    for i in range(1, train_normalized_X.shape[0]):
        subX = train_normalized_X[:i + 1]
        suby = y[:i + 1]
        res = opt.minimize(fun=cost, x0=np.ones((power_max + 1,)), args=(subX, suby, l),
                           method="TNC", jac=gradient)
        t = res.x
        error_train.append(cost(t, subX, suby, 0))  # 计算error时不需要正则化
        error_validation.append(cost(t, val_normalized_X, yval, 0))
    plt.subplot(2, 2, 4)
    plt.plot([i for i in range(1, train_normalized_X.shape[0])], error_train, label="training set error")
    plt.plot([i for i in range(1, train_normalized_X.shape[0])], error_validation, label="cross validation set error")
    plt.legend(loc="upper right")
    plt.xlabel("m(numbers of training set)")
    plt.title("learning curves")
    plt.show()

    # lambda变化对应的learn curve
    ls = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_validation = []
    for l in ls:
        res = opt.minimize(fun=cost, x0=np.ones((power_max + 1,)), args=(train_normalized_X, y, l),
                           method="TNC", jac=gradient)
        error_train.append(cost(res.x, train_normalized_X, y, 0))
        error_validation.append(cost(res.x, val_normalized_X, yval, 0))
    plt.plot([i for i in ls], error_train, label="training set error")
    plt.plot([i for i in ls], error_validation, label="cross validation set error")
    plt.legend(loc="upper right")
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.title("Selecting λ using a cross validation set")
    plt.show()

    # 计算测试集的损失值
    test_features = poly_features(Xtest, power_max)
    test_normalized_features = normalize_features(test_features, means, stds)
    test_normalized_X = np.insert(test_normalized_features, 0, 1, axis=1)
    res = opt.minimize(fun=cost, x0=np.ones((power_max + 1,)), args=(train_normalized_X, y, 3),
                       method="TNC", jac=gradient)
    print(cost(res.x, test_normalized_X, ytest, 0))

    # 随机选择数据
    error_train = []
    error_validation = []
    print(train_normalized_X.shape, val_normalized_X.shape)
    for i in range(X.shape[0]):
        Xy = randomly_select(np.concatenate((train_normalized_X, y), axis=1), i + 1)
        subtrainX = Xy[..., :-1]
        subtrainy = Xy[..., -1]
        res = opt.minimize(fun=cost, x0=np.ones((power_max + 1,)), args=(subtrainX, subtrainy, 0.01), method="TNC",
                           jac=gradient)
        t = res.x
        error_train.append(cost(t, subtrainX, subtrainy, 0.01))
        Xy = randomly_select(np.concatenate((val_normalized_X, yval), axis=1), i + 1)
        subvalX = Xy[..., :-1]
        subvaly = Xy[..., -1]
        error_validation.append(cost(t, subvalX, subvaly, 0.01))
    plt.plot([i for i in range(X.shape[0])], error_train, label="training set error")
    plt.plot([i for i in range(X.shape[0])], error_validation, label="cross validation set error")
    plt.legend(loc="upper right")
    plt.xlabel("m(numbers of training set)")
    plt.title("learning curves(randomly select)")
    plt.show()

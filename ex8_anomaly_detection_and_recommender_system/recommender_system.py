import scipy.io as sio
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error


def serialize(X, theta):
    """
    参数一维向量化
    :param X: ndarray,电影特征
    :param theta: ndarray,用户特征
    :return: ndarray,一维化向量
    """
    return np.concatenate((X.flatten(), theta.flatten()), axis=0)


def deserialize(params, nm, nu, nf):
    """
    将一维化参数向量还原
    :param params: ndarray,一维化的用户特征和电影特征
    :param nm: int,电影数量
    :param nu: int,用户数量
    :param nf: int,特征数量
    :return: (ndarray,ndarray) 电影特征,用户特征
    """
    X = params[:nm * nf].reshape(nm, nf)
    theta = params[nm * nf:].reshape(nu, nf)
    return X, theta


def collaborative_filtering_cost(params, Y, R, nm, nu, nf, l=0.0):
    """
    协同过滤算法目标函数
    :param params: ndarray,一维化的用户特征和电影特征
    :param Y: ndarray,表明用户的评分
    :param R: ndarray,表明哪些用户评价了哪些电影
    :param nm: int,电影数量
    :param nu: int,用户数量
    :param nf: int,特征数量
    :param l: float,惩罚参数
    :return: float,损失值
    """
    X, theta = deserialize(params, nm, nu, nf)
    part1 = np.sum(((X.dot(theta.T) - Y) ** 2) * R) / 2
    part2 = l * np.sum(theta ** 2) / 2
    part3 = l * np.sum(X ** 2) / 2
    return part1 + part2 + part3


def collaborative_filtering_gradient(params, Y, R, nm, nu, nf, l=0.0):
    """
    协同过滤梯度下降
    :param params: ndarray,一维化的用户特征和电影特征
    :param Y: ndarray,表明用户的评分
    :param R: ndarray,表明哪些用户评价了哪些电影
    :param nm: int,电影数量
    :param nu: int,用户数量
    :param nf: int,特征数量
    :param l: float,惩罚参数
    :return: ndarray,跟新后的一维化的用户特征和电影特征
    """
    X, theta = deserialize(params, nm, nu, nf)
    g_X = ((X.dot(theta.T) - Y) * R).dot(theta) + l * X
    g_theta = ((X.dot(theta.T) - Y) * R).T.dot(X) + l * theta
    return serialize(g_X, g_theta)


def check_gradient(params, Y, R, nm, nu, nf):
    # X, theta = deserialize(params, nm, nu, nf)
    e = 0.0001
    m = len(params)
    g_params = np.zeros((m,))
    for i in range(m):
        temp = np.zeros((m,))
        temp[i] = e
        g_params[i] = (collaborative_filtering_cost(params + temp, Y, R, nm, nu, nf) -
                       collaborative_filtering_cost(params - temp, Y, R, nm, nu, nf)) / (2 * e)
    return g_params


def normalizeRatings(Y, R):
    """
    归一化评分，只对有评分的进行操作
    :param Y: ndarray,表明用户的评分
    :param R: ndarray,表明哪些用户评价了哪些电影
    :return: (ndarray,ndarray)
    """
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)
    #     Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    Ynorm = (Y - Ymean) * R  # 这里也要注意不要归一化未评分的数据
    return Ynorm, Ymean


if __name__ == "__main__":
    data1 = sio.loadmat("data\\ex8_movies.mat")
    Y = data1["Y"]  # (1682,943)
    R = data1["R"]  # (1682,943)
    data2 = sio.loadmat("data\\ex8_movieParams.mat")
    X = data2["X"]  # (1682,10)
    theta = data2["Theta"]  # (943,10)
    nu = data2["num_users"][0][0]  # (1,1) 943
    nm = data2["num_movies"][0][0]  # (1,1) 1682
    nf = data2["num_features"][0][0]  # (1,1) 10

    # 题目中计算数据不是全部数据，取nm=5,nu=4,nf=3,值为22.224603725685675
    # nu = 4
    # nm = 5
    # nf = 3
    # X = X[:nm, :nf]
    # theta = theta[:nu, :nf]
    # Y = Y[:nm, :nu]
    # R = R[:nm, :nu]
    print(collaborative_filtering_cost(serialize(X, theta), Y, R, nm, nu, nf))
    # 正则化时选择的lambda为1.5
    print(collaborative_filtering_cost(serialize(X, theta), Y, R, nm, nu, nf, 1.5))

    # 梯度下降检测运行得太慢，没有跑出结果
    # print(collaborative_filtering_gradient(serialize(X, theta), Y, R, nm, nu, nf)[10],
    #       check_gradient(serialize(X, theta), Y, R, nm, nu, nf)[10])

    # 读入电影标签
    f = open("data\\movie_ids.txt", "r")
    movies = []
    for line in f.readlines():
        movies.append(line.split(' ', 1)[-1][:-1])

    # 训练模型
    # 先添加一组自定义的用户数据
    my_ratings = np.zeros((1682, 1))
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    Y = np.concatenate((Y, my_ratings), axis=1)
    R = np.concatenate((R, my_ratings > 0), axis=1)
    nu += 1

    # params = serialize(np.random.random((nm, nf)), np.random.random((nu, nf)))
    # res = opt.minimize(fun=collaborative_filtering_cost, x0=params, args=(Y, R, nm, nu, nf, 10),
    #                    method='TNC',
    #                    jac=collaborative_filtering_gradient)
    # print(res)
    # sio.savemat("parameters.mat", {"params": res.x})
    # trained_X, trained_theta = deserialize(res.x, nm, nu, nf)
    trained_X, trained_theta = deserialize(sio.loadmat("parameters.mat")["params"].ravel(), nm, nu, nf)
    predict = trained_X.dot(trained_theta.T)
    my_predict = predict[..., -1]

    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(my_ratings[i], movies[i])
    # 从预测结果中选10个最优推荐
    # 由于训练初始化参数的不同会导致最后的结果不同
    print("Top recommendations for you:")
    for i in range(10):
        index = int(np.argmax(my_predict))
        print("Predicting rating ", my_predict[index], " for movie ", movies[index])
        my_predict[index] = -1

    # 用均方误差来评价
    Y = Y.flatten()
    R = R.flatten()
    predict = predict.flatten()
    true_y = []
    pre_y = []
    for i in range(len(Y)):
        if R[i] == 1:
            true_y.append(Y[i])
            pre_y.append(predict[i])
    print("当前训练对岳原始数据集的均方误差", mean_squared_error(true_y, pre_y))

import numpy as np
import matplotlib.pyplot as plt


def cost(X, theta, y):
    m = X.shape[0]
    temp = X.dot(theta) - y
    return temp.T.dot(temp) / (2 * m)


def gradient_descent(X, theta, y, alpha, iterations):
    m = X.shape[0]
    c = []  # 存储计算损失值
    for i in range(iterations):
        theta -= (alpha / m) * X.T.dot(X.dot(theta) - y)
        c.append(cost(X, theta, y))
    return theta, c


def normal_equation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


if __name__ == "__main__":
    # 读入数据
    f = open("ex1data2.txt", 'r')
    house_size = []
    bedroom_number = []
    house_price = []
    for line in f.readlines():
        col1 = float(line.split(",")[0])
        col2 = float(line.split(",")[1])
        col3 = float(line.split(",")[2].split("\n")[0])
        house_size.append(col1)
        bedroom_number.append(col2)
        house_price.append(col3)

    # 特征归一化
    x1 = np.array(house_size).reshape(-1, 1)
    x2 = np.array(bedroom_number).reshape(-1, 1)
    y = np.array(house_price).reshape(-1, 1)
    data = np.concatenate((x1, x2, y), axis=1)  # 放在一个ndarray中便于归一化处理

    # mean = np.mean(data, axis=0)
    # std = np.std(data, axis=0, ddof=1)
    # nor_data = (data - mean) / std
    mean = np.mean(data, axis=0)  # 计算每一列的均值
    ptp = np.ptp(data, axis=0)  # 计算每一列的最大最小差值
    nor_data = (data - mean) / ptp  # 归一化
    X = np.insert(nor_data[..., :2], 0, 1, axis=1)  # 添加x0=1
    y = nor_data[..., -1]

    # 梯度下降
    theta = np.zeros((3,))
    alpha = 0.1
    iterations = 10000
    # print(cost(X, theta, y))
    # print(X)
    theta, c = gradient_descent(X, theta, y, alpha=alpha, iterations=iterations)

    # 可视化下降过程
    plt.plot()
    plt.title("Visualizing J(θ)")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.plot([i for i in range(iterations)], c, color="red")
    plt.show()
    print("use gradient descent:", theta)
    print("use normal equation", normal_equation(X, y))

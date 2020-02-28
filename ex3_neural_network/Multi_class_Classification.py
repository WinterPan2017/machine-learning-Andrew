import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.optimize as opt
from sklearn.metrics import classification_report, accuracy_score


def randomly_select(images, numbers):
    """
    从images中选择numbers张图片

    parameters:
    ----------
    images : ndarray
            多干张图片
    numbers : int
            随机选择的图片数量
    """
    m = images.shape[0]
    n = images.shape[1]
    flags = np.zeros((m,), bool)
    res = False
    for i in range(numbers):
        index = random.randint(0, m - 1)
        while flags[index]:
            index = random.randint(0, m)
        if type(res) == bool:
            res = images[index].reshape(1, n)
        else:
            res = np.concatenate((res, images[index].reshape(1, n)), axis=0)
    return res


def mapping(images, images_dimension):
    """
    将若干张图片，组成一张图片

    parameters:
    ----------
    images : ndarray
            多干张图片
    images_dimension : int
            新的正方形大图片中一边上有多少张图片
    """
    image_dimension = int(np.sqrt(images.shape[-1]))
    image = False
    im = False
    for i in images:
        if type(image) == bool:
            image = i.reshape(image_dimension, image_dimension)
        else:
            if image.shape[-1] == image_dimension * images_dimension:
                if type(im) == bool:
                    im = image
                else:
                    im = np.concatenate((im, image), axis=0)
                image = i.reshape(image_dimension, image_dimension)
            else:
                image = np.concatenate((image, i.reshape(image_dimension, image_dimension)), axis=1)
    return np.concatenate((im, image), axis=0)


def convert(y):
    """
    将y的每个值变化为向量，来表示数字

    parameters:
    ----------
    y : ndarray
            表示图片对应额数字
    """
    n = len(np.unique(y))
    res = False
    for i in y:
        temp = np.zeros((1, n))
        temp[0][i[0] % 10] = 1
        if type(res) == bool:
            res = temp
        else:
            res = np.concatenate((res, temp), axis=0)
    return res


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# theta shape (1,n+1)
# X shape (m,n+1)
# y shape (m,)
def cost(theta, X, y, l):
    m = X.shape[0]
    part1 = np.mean((-y) * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))
    part2 = (l / (2 * m)) * np.sum(theta * theta)
    return part1 + part2


def gradient(theta, X, y, l):
    m = X.shape[0]
    part1 = X.T.dot(sigmoid(X.dot(theta)) - y)
    part2 = (l / m) * theta
    part2[0] = 0
    return part1 + part2


def predict(theta, X):
    p = sigmoid(X.dot(theta.T))
    res = False
    for i in p:
        index = np.argmax(i)
        temp = np.zeros((1, 10))
        temp[0][index] = 1
        if type(res) == bool:
            res = temp
        else:
            res = np.concatenate((res, temp), axis=0)
    return res


if __name__ == "__main__":
    data = sio.loadmat("ex3data1.mat")
    X = data['X']
    y = data['y']
    print(X.shape, y.shape)

    im = mapping(randomly_select(X, 100), 10)
    # print(im.shape)
    plt.imshow(im.T)  # 图片是镜像的需要转置让它看起来更更正常
    plt.axis('off')
    plt.show()

    y = convert(y)
    X = np.insert(X, 0, 1, axis=1)
    m = X.shape[0]
    n = X.shape[1] - 1
    theta = np.zeros((n + 1,))
    trained_theta = False
    for i in range(y.shape[-1]):
        res = opt.minimize(fun=cost, x0=theta, args=(X, y[..., i], 1), method="TNC", jac=gradient)
        if type(trained_theta) == bool:
            trained_theta = res.x.reshape(1, n + 1)
        else:
            trained_theta = np.concatenate((trained_theta, res.x.reshape(1, n + 1)), axis=0)
    # print(trained_theta, trained_theta.shape)
    # print(predict(trained_theta, X).shape, y.shape)
    print(classification_report(y, predict(trained_theta, X), target_names=[str(i) for i in range(10)], digits=4))

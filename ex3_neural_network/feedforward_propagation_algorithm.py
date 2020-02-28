import scipy.io as sio
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from ex3_neural_network.Multi_class_Classification import convert


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(theta, X):
    labels = theta[-1].shape[0]
    for t in theta:
        X = np.insert(X, 0, 1, axis=1)
        X = sigmoid(X.dot(t.T))
    p = X
    res = np.zeros((1, labels))
    print(p)
    for i in p:
        index = np.argmax(i)
        temp = np.zeros((1, labels))
        temp[0][index] = 1
        res = np.concatenate((res, temp), axis=0)
    return res[1:]


if __name__ == "__main__":
    data = sio.loadmat("ex3data1.mat")
    theta = sio.loadmat("ex3weights.mat")
    # print(theta.keys())
    theta1 = theta["Theta1"]  # (25,401)
    theta2 = theta["Theta2"]  # (10,26)
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
    X = data['X']  # (5000,400)
    print(X.shape)
    print(classification_report(y, predict((theta1, theta2), X), digits=3))

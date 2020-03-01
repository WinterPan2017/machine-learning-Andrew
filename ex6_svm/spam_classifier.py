import scipy.io as sio
import numpy as np
from sklearn import svm
import re
from nltk.stem.porter import PorterStemmer


def process_email(content):
    """
    处理邮件文本
    :param content: str,邮件文本
    :return: list,单词列表
    """
    content = content.lower()
    content = re.sub(r'<.*>', '', content)  # 移除html标签
    content = re.sub(r'http[s]?://.+', 'httpaddr', content)  # 移除url
    content = re.sub(r'[\S]+@[\w]+.[\w]+', 'emailaddr', content)  # 移除邮箱
    content = re.sub(r'[\$][0-9]+', 'dollar number', content)  # 移除$,解决dollar和number连接问题
    content = re.sub(r'\$', 'dollar number', content)  # 移除单个$
    content = re.sub(r'[0-9]+', 'number', content)  # 移除数字
    content = re.sub(r'[\W]+', ' ', content)  # 移除字符
    words = content.split(' ')
    if words[0] == '':
        words = words[1:]  # 分开时会导致开始空格处多出一个空字符
    porter_stemmer = PorterStemmer()
    for i in range(len(words)):
        words[i] = porter_stemmer.stem(words[i])  # 提取词干
    return words


def mapping(word, vocab):
    """
    单词映射为编号
    :param word: str,单词
    :param vocab: list,编号 表
    :return: int,编号
    """
    for i in range(len(vocab)):
        if word == vocab[i]:
            return i
    return None


def email_features(email, vocab):
    """
    邮件单词列表转化为特征向量
    :param email: list,邮件的单词列表
    :param vocab: list,编号表
    :return: ndarray,特征向量
    """
    features = np.zeros((len(vocab, )))
    for word in email:
        index = mapping(word, vocab)
        if index is not None:
            features[index] = 1
    return features


if __name__ == "__main__":
    f = open("data\\vocab.txt")
    vocab = []  # python中的下标从0起，因此与作业中有所不同
    for l in f.readlines():
        s = l.split('\t')[-1].split('\n')[0]
        vocab.append(s)
    print(mapping('anyon', vocab))
    print(email_features("aa hello", vocab))

    train_data = sio.loadmat("data\\spamTrain.mat")
    train_X = train_data['X']  # (4000,1899)
    train_y = train_data['y']  # (4000,1)
    test_data = sio.loadmat("data\\spamTest.mat")
    test_X = test_data['Xtest']  # (1000,1899)
    test_y = test_data['ytest']  # (1000,1)
    model = svm.SVC(kernel='linear')  # 这里的n比较大，选用线性核函数效果好
    model.fit(train_X, train_y.ravel())
    print(model.score(train_X, train_y.ravel()), model.score(test_X, test_y.ravel()))

    x = email_features(process_email(open("data\\emailSample2.txt").read()), vocab)
    print(model.predict(x.reshape(1, -1)))

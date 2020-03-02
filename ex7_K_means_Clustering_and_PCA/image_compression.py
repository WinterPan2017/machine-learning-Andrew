import matplotlib.pyplot as plt
import numpy as np
import ex7_K_means_Clustering_and_PCA.k_means_clustering as k_means


def compress(image, colors_num):
    """
    压缩图片
    :param image: ndarray,原始图片
    :param colors_num: int,压缩后的颜色数量
    :return: (ndarray,ndarray),第一个每个像素点存储一个值，第二个为颜色矩阵
    """
    d1, d2, _ = image.shape
    raw_image = image.reshape(d1 * d2, -1)  # 展开成二维数组
    idx, centroids_all = k_means.k_means(raw_image, colors_num)
    colors = centroids_all[-1]
    compressed_image = np.zeros((1, 1))  # 构造压缩后的图片格式
    for i in range(d1 * d2):
        compressed_image = np.concatenate((compressed_image, idx[i].reshape(1, -1)), axis=0)
    compressed_image = compressed_image[1:].reshape(d1, d2, -1)
    return compressed_image, colors


def compressed_format_to_normal_format(compressed_image, colors):
    """
    将压缩后的图片转为正常可以显示的图片格式
    :param compressed_image: ndarray,压缩后的图片,存储颜色序号
    :param colors: ndarray,颜色列表
    :return: ndarray,正常的rgb格式图片
    """
    d1, d2, _ = compressed_image.shape
    normal_format_image = np.zeros((1, len(colors[0])))
    compressed_image = compressed_image.reshape(d1 * d2, -1)
    for i in range(d1 * d2):
        normal_format_image = np.concatenate((normal_format_image, colors[int(compressed_image[i][0])].reshape(1, -1)),
                                             axis=0)
    normal_format_image = normal_format_image[1:].reshape(d1, d2, -1)
    return normal_format_image


if __name__ == "__main__":
    image = plt.imread("data\\bird_small.png")  # (128,128,3)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("raw image")
    plt.subplot(1, 2, 2)
    compressed_image, colors = compress(image, 16)
    print(compressed_image.shape, colors.shape)
    plt.imshow(compressed_format_to_normal_format(compressed_image, colors))
    plt.axis('off')
    plt.title("compressed image")
    plt.show()

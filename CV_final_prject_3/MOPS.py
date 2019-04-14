import cv2
import numpy as np

import math
from scipy import ndimage


def MPOSdescribtor(image, keypoints):
#MOPS：Multi-Scale Oriented Patches
#basic version
# Input：images RGB 3-channel
#步骤:
    # 从特征点周围的40×40像素区域子采样的8×8定向图像块。
    # 提供一个变换矩阵，它将围绕特征点的40×40窗口
    # 变换为8×8图像块，使其特征点方向指向右侧。
    # 将8×8图像块规范化为零均值和单位方差。如果方差非常接近零(幅度小于10 -5)，
    # 则返回 一个全零向量以避免除零错误。

    image = image.astype(np.float32)
    image /= 255.

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)
    height, width = grayImage.shape[:2]

    desc = np.zeros((len(keypoints), 8 * 8))

    newpd = np.zeros((height + 40 - 1, width + 40 - 1), dtype=float)
    newpd[20:20 + height, 20:20 + width] = grayImage

    for i, f in enumerate(keypoints):

        # transMx = np.zeros((2, 3))

        x, y = f.pt
        #依次处理 角度，rotation，scale，40*40变成8*8之后的位移
        angle = -f.angle * (2 * np.pi) / 360

        trans_mx1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])

        rot_mx = np.array([[math.cos(angle), -math.sin(angle), 0],
                           [math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]])

        scale_mx = np.array([[1 / 5, 0, 0],
                             [0, 1 / 5, 0],
                             [0, 0, 1]])

        trans_mx2 = np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]])
        #把这些变换矩阵都乘一起
        transMx = np.dot(trans_mx2, np.dot(scale_mx, np.dot(rot_mx, trans_mx1)))[0:2, 0:3]


        # Normalize the descriptor to have zero mean and unit variance.
        #对特征描述子进行归一化

        destImage = cv2.warpAffine(grayImage, transMx,
                                   (8, 8), flags=cv2.INTER_LINEAR)

        destImage = destImage - np.mean(destImage)
        if (np.std(destImage) < 10 ** -5):
            destImage = np.zeros((1, 8 * 8))
            #去除variance为0的部分.
        else:
            destImage = destImage / np.std(destImage)
            destImage = np.reshape(destImage, (1, 8 * 8))

        desc[i] = destImage

    return np.array(desc, dtype='float32')
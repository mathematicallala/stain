import cv2
import numpy as np
import cv2
import os
# region 辅助函数
# RGB2XYZ空间的系数矩阵
M = np.array([[0.5141, 0.3239, 0.1604],
              [0.2651, 0.6702, 0.0641],
              [0.0241, 0.1228, 0.8444]])


# im_channel取值范围：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787


# endregion


# region RGB 转 Lab
# 像素值RGB转XYZ空间，pixel格式:(B,G,R)
# 返回XYZ空间下的值
def __rgb2xyz__(pixel):
    r, g, b = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
    XYZ空间转Lab空间
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab


# endregion

# region Lab 转 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb


# endregion

if __name__ == '__main__':
    img_t0 = cv2.imread('.\.')
    img_t0 = cv2.resize(img_t0, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_t = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    w = img_t.shape[0]
    h = img_t.shape[1]
    lab_t = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            Lab = RGB2Lab(img_t[i, j])
            lab_t[i, j] = (Lab[0], Lab[1], Lab[2])
    L_t, a_t, b_t = cv2.split(lab_t)


    num = 1
    path1 = os.path.join('.\.')
    root = '.\.'
    path_list = os.listdir(root)
    for file in path_list:
        file_name = os.path.join(root, file)
        img_s0 = cv2.imread(file_name, 1)
        img_s0 = cv2.resize(img_s0, (256, 256), interpolation=cv2.INTER_CUBIC)
        img_s = cv2.cvtColor(img_s0, cv2.COLOR_BGR2RGB)
        lab_s = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                Lab = RGB2Lab(img_s[i, j])
                lab_s[i, j] = (Lab[0], Lab[1], Lab[2])
        L_s, a_s, b_s = cv2.split(lab_s)
        L_s = (np.std(L_t) / np.std(L_s)) * (L_s - np.mean(L_s)) + np.mean(L_t)
        a_s = (np.std(a_t) / np.std(a_s)) * (a_s - np.mean(a_s)) + np.mean(a_t)
        b_s = (np.std(b_t) / np.std(b_s)) * (b_s - np.mean(b_s)) + np.mean(b_t)
        L_s_ = np.zeros([w, h], np.int32)
        a_s_ = np.zeros([w, h], np.int32)
        b_s_ = np.zeros([w, h], np.int32)
        for i in range(w):
            for j in range(h):
                L_s_[i, j] = int(L_s[i, j])
                a_s_[i, j] = int(a_s[i, j])
                b_s_[i, j] = int(b_s[i, j])
        Lab_N = cv2.merge([L_s_, a_s_, b_s_])
        img_new = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                rgb = Lab2RGB(Lab_N[i, j])
                img_new[i, j] = (rgb[2], rgb[1], rgb[0])
        img_new = img_new.astype(np.uint8)
        cv2.imwrite(path1 + '\\' + str(num) + '.png', img_new)
        num = num + 1

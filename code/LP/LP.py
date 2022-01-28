import numpy as np
import math
from sklearn.cluster import KMeans
import cv2
from collections import defaultdict
import os
import time
def rgbtohsi(B, G, R):
    rows = int(B.shape[0])
    cols = int(B.shape[1])
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    HSI_H = np.zeros([rows, cols], np.int32)
    HSI_S = np.zeros([rows, cols], np.float32)
    HSI_I = np.zeros([rows, cols], np.int32)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = float(np.arccos(num / den))
            else:
                H = 2 * np.pi - float(np.arccos(num / den))
            min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
            sum = B[i, j] + G[i, j] + R[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum
            I = sum / 3.0
            HSI_H[i, j] = np.round(H * 180 / np.pi)
            HSI_S[i, j] = S
            HSI_I[i, j] = int(I * 255)
    return HSI_H, HSI_S, HSI_I

def hue(h, s, h0, w0):#
    hist1 = np.zeros([1, 361], np.float32)
    for i in range(h0):
        for j in range(w0):
          thet = h[i, j]
          hist1[0, thet] = hist1[0, thet] + s[i, j]

    return hist1
def train(V, W, H, r, k, e):
    m, n = np.shape(V)
    for x in range(k):
        V_pre = np.dot(W, H)
        E = V - V_pre
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]
        mi = np.min(W)
        if err < e or mi < 1e-6:
            break

        a = np.dot(W.T, V)  # Hkj
        b = np.dot(W.T, np.dot(W, H))
        H[b != 0] = (H * a / b)[b != 0]
        # for i_1 in range(r):
        #     for j_1 in range(n):
        #         if b[i_1, j_1] != 0:
        #             H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = np.dot(V, H.T)
        d = np.dot(W, np.dot(H, H.T))
        W[d != 0] = (W * c / d)[d != 0]
        # for i_2 in range(m):
        #     for j_2 in range(r):
        #         if d[i_2, j_2] != 0:
        #             W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]


    return W, H


def s_v(centroid_set, c, h, s, ii, h0, w0):
    vv = []
    ss = []
    for i in range(c):
        a = int(centroid_set[i, 0])

        b1 = 0
        b2 = 0
        b3 = 0
        for j in range(h0):
            for k in range(w0):
                if a == h[j, k]:
                    b1 = b1 + s[j, k]
                    b2 = b2 + s[j, k] * ii[j, k]
                    b3 = b3 + s[j, k] ** 2
        for mm in range(1, 100):
            if b1 == 0:
                for j in range(h0):
                    for k in range(w0):
                        if a - mm == h[j, k]:
                            b1 = b1 + s[j, k]
                            b2 = b2 + s[j, k] * ii[j, k]
                            b3 = b3 + s[j, k] ** 2
            if b1 == 0:
                for j in range(h0):
                    for k in range(w0):
                        if a + mm == h[j, k]:
                            b1 = b1 + s[j, k]
                            b2 = b2 + s[j, k] * ii[j, k]
                            b3 = b3 + s[j, k] ** 2
            if b1 != 0:
                break
        if b1 != 0:
            vv.append(b2 / b1)
            ss.append(b3 / b1)
    return ss, vv

def hsi_rgb(centroid_set, ss, vv, c):
    I_i_bgr = np.zeros([3, c], np.float32)

    for x in range(c):
        h = centroid_set[x, 0] * np.pi / 180
        s = ss[x]
        i = vv[x]
        if 0 <= h < (2 / 3) * np.pi:
            b = i * (1 - s)
            r = i * (1 + s * np.cos(h) / np.cos((1 / 3) * np.pi - h))
            g = 3 * i - (r + b)
        elif (2 / 3) * np.pi <= h < (4 / 3) * np.pi:
            h = h - (2 / 3) * np.pi
            r = i * (1 - s)
            g = i * (1 + s * np.cos(h) / np.cos((1 / 3) * np.pi - h))
            b = 3 * i - (r + g)
        else:
            h = h - (4 / 3) * np.pi
            g = i * (1 - s)
            b = i * (1 + s * np.cos(h) / np.cos((1 / 3) * np.pi - h))
            r = 3 * i - (g + b)
        I_i_bgr[0, x] = b
        I_i_bgr[1, x] = g
        I_i_bgr[2, x] = r


    return I_i_bgr


def spectrum_matrix(I_i_bgr, c, I0):

    W = np.zeros([3, c], np.float32)
    for i in range(3):
        for j in range(c):
            if I_i_bgr[i, j] != 0:
                W[i, j] = np.log(I0 / I_i_bgr[i, j])
            else:
                W[i, j] = 0
    return W
def density(b, g, r, h0, w0, W):
    BGR_I = np.zeros([3, h0 * w0], np.float32)
    for i in range(h0):  # 768
        for j in range(w0):  # 896
            if b[i, j] != 0:
                BGR_I[0][j + i * w0] = np.log(255 / b[i, j])
            else:
                BGR_I[0][j + i * w0] = 0

            if g[i, j] != 0:
                BGR_I[1][j + i * w0] = np.log(255 / g[i, j])
            else:
                BGR_I[1][j + i * w0] = 0

            if r[i, j] != 0:
                BGR_I[2][j + i * w0] = np.log(255 / r[i, j])
            else:
                BGR_I[2][j + i * w0] = 0
    W_ni = np.dot(W.T, np.linalg.inv(np.dot(W, W.T)))
    H = np.dot(W_ni, BGR_I)

    mi = np.min(H)
    H = H + (-mi)
    return H
def cluster(hist):
    KMEANS = KMeans(n_clusters=2).fit(hist)
    clusters = defaultdict(list)
    for ind, label in enumerate(KMEANS.labels_):
        clusters[label].append(ind)
    mu = []
    for inds in clusters.values():
        partial_data = hist[inds]
        mu.append(partial_data.mean(axis=0))
    mu = np.array(mu)
    return mu

def t_optimal(I_i_bgr, b, g, r, c, h0, w0, I0):
    W = spectrum_matrix(I_i_bgr, c, I0)
    H = density(b, g, r, h0, w0, W)
    BGR_I = np.zeros([3, h0 * w0], np.float32)
    for i in range(h0):
        for j in range(w0):
            if b[i, j] != 0:
                BGR_I[0][j + i * w0] = np.log(I0 / b[i, j])
            else:
                BGR_I[0][j + i * w0] = 0

            if g[i, j] != 0:
                BGR_I[1][j + i * w0] = np.log(I0 / g[i, j])
            else:
                BGR_I[1][j + i * w0] = 0

            if r[i, j] != 0:
                BGR_I[2][j + i * w0] = np.log(I0 / r[i, j])
            else:
                BGR_I[2][j + i * w0] = 0
    W1, H1 = train(BGR_I, W, H, c, 100, 1e-5)
    return W1, H1
def normalization(M_t, M_s,H_s, img, h0, w0, I0):
    img_normal = np.multiply(255, np.exp(-M_t.dot(H_s)))
    img_normal[img_normal > 255] = 255
    img_normal = np.reshape(img_normal.T, (h0, w0, 3)).astype(np.uint8)

    return img_normal
def template():
    img00 = cv2.imread('.\.', 1)
    img00 = cv2.resize(img00, (256, 256), interpolation=cv2.INTER_CUBIC)
    h0 = 256
    w0 = 256
    I0 = np.max(img)
    b, g, r = cv2.split(img00)
    h, s, v = rgbtohsi(b, g, r)
    hist = hue(h, s, h0, w0)
    hist = hist.T / sum(sum(hist))

    data01 = []
    for i in range(361):
        if hist[i, 0] != 0:
            for j in range(int(np.around(256 * 256 * hist[i, 0]))):
                data01.append(i)
    data_s = np.zeros([len(data01), 1], np.float32)
    for i in range(len(data01)):
        data_s[i, 0] = data01[i]
    mu = cluster(data_s)
    c = len(mu)
    ss, vv = s_v(mu, c, h, s, v, h0, w0)
    I_i_bgr = hsi_rgb(mu, ss, vv,c)
    M_t, H_t = t_optimal(I_i_bgr, b, g, r, c, h0, w0, I0)
    return M_t, img00

def source(img):
    h0, w0, a = img.shape
    I0 = np.max(img)
    b, g, r = cv2.split(img)
    b = 255 * (b / np.max(b))
    b = b.astype(np.uint8)
    g = 252 * (g / np.max(g))
    g = g.astype(np.uint8)
    r = 255 * (r / np.max(r))
    r = r.astype(np.uint8)
    h, s, v = rgbtohsi(b, g, r)
    hist = hue(h, s, h0, w0)
    hist = hist.T / sum(sum(hist))

    data01 = []
    for i in range(361):
        if hist[i, 0] != 0:
            for j in range(int(np.around(256 * 256 * hist[i, 0]))):
                data01.append(i)
    data_s = np.zeros([len(data01), 1], np.float32)
    for i in range(len(data01)):
        data_s[i, 0] = data01[i]
    mu = cluster(data_s)
    c = len(mu)
    ss, vv = s_v(mu, c, h, s, v, h0, w0)
    I_i_bgr = hsi_rgb(mu, ss, vv, c)
    M_s, H_s = t_optimal(I_i_bgr, b, g, r, c, h0, w0, I0)
    M_t, img00 = template()
    Inorm = normalization(M_t, M_s, H_s, img, h0, w0, I0)

    return Inorm
if __name__ == '__main__':
    file_name = '.\.'
    img = cv2.imread(file_name, 1)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    re = source(img)
    cv2.imshow('h', re)
    cv2.imshow('e', img)
    cv2.waitKey(0)


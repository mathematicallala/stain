import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from numpy import *
import cv2
import math
import os
import time

def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    B, G, R = cv2.split(rgb_lwpImg)
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



def achromatic_saturation_weighted(stain_achromatic_T, h, s, v, h0, w0):
    achromatic_h, achromatic_s = stain_achromatic_partition(stain_achromatic_T, h, s, v, h0, w0)
    hist2 = np.zeros([1, 361], np.float32)
    for i in range(len(achromatic_h)):
        theta = achromatic_h[i]
        hist2[0, theta] = hist2[0, theta] + achromatic_s[i]
    hist2 = hist2 / sum(sum(hist2))
    return hist2

def saturation_weighted_hue(h, s, h0, w0):
    hist1 = np.zeros([1, 361], np.float32)
    for i in range(h0):
        for j in range(w0):
          thet = h[i, j]
          hist1[0, thet] = hist1[0, thet] + s[i, j]
    hist1 = hist1 / sum(sum(hist1))
    return hist1

def stain_achromatic_partition(stain_achromatic_T, h, s, I, h0, w0):
    achromatic_h = []
    achromatic_s = []
    for i in range(h0):
        for j in range(w0):
            if I[i, j] > stain_achromatic_T:
                achromatic_h.append(h[i, j])
                achromatic_s.append(s[i, j])
    return achromatic_h, achromatic_s

def achromatic_u(hist, stain_achromatic_T, h, s, v, h0, w0):
    achromatic_hist = achromatic_saturation_weighted(stain_achromatic_T, h, s, v, h0, w0)
    achromatic_cos_sum = 0
    achromatic_sin_sum = 0
    for i in range(len(hist.T)):
        achromatic_cos_sum = achromatic_cos_sum + achromatic_hist[0, i] * np.cos(i * np.pi / 180)
        achromatic_sin_sum = achromatic_sin_sum + achromatic_hist[0, i] * np.sin(i * np.pi / 180)
    qq = math.atan(achromatic_cos_sum / achromatic_sin_sum)
    return qq


def fuzzy_membership_v(u, c, m, hist):
    n = len(hist.T)
    fuzzy_membership_vij = np.zeros([c, n], np.float32)
    for i in range(c):
        for j in range(n):
            dij = 1 - np.cos(j * np.pi / 180 - u[i])
            d1j = 1 - np.cos(j * np.pi / 180 - u[0])
            d2j = 1 - np.cos(j * np.pi / 180 - u[1])
            d3j = 1 - np.cos(j * np.pi / 180 - u[2])
            fuzzy_membership_vij[i, j] = ((dij / d1j) ** (1 / (m - 1)) + (dij / d2j) ** (1 / (m - 1)) +
                                          (dij / d3j) ** (1 / (m - 1))) ** (-1)
    return fuzzy_membership_vij
def sort(fuzzy_membership_vij, hist, la):
    lower_approximation0 = []
    lower_approximation1 = []
    lower_approximation2 = []
    fuzzy_boundary0 = []
    fuzzy_boundary1 = []
    fuzzy_boundary2 = []
    fuzzy_max_set = []
    for j in range(len(hist.T)):
        sss = sorted(fuzzy_membership_vij[:, j])
        first_max = sss[2]
        second_max = sss[1]
        qwe = np.argmax(fuzzy_membership_vij[:, j])
        qwe1 = fuzzy_membership_vij[:, j].tolist()
        qwe3 = qwe1.index(second_max)
        fuzzy_max_set.append(first_max)
        if first_max - second_max > la:
            if qwe == 0:
                lower_approximation0.append(j)
            elif qwe == 1:
                lower_approximation1.append(j)
            else:
                lower_approximation2.append(j)
        else:
            if qwe == 0:
                fuzzy_boundary0.append(j)
                if qwe3 == 1:
                    fuzzy_boundary1.append(j)
                else:
                    fuzzy_boundary2.append(j)
            elif qwe == 1:
                fuzzy_boundary1.append(j)
                if qwe3 == 0:
                    fuzzy_boundary0.append(j)
                else:
                    fuzzy_boundary2.append(j)
            elif qwe == 2:
                fuzzy_boundary2.append(j)
                if qwe3 == 0:
                    fuzzy_boundary0.append(j)
                else:
                    fuzzy_boundary1.append(j)
    lower_approximation = [lower_approximation0, lower_approximation1, lower_approximation2]
    fuzzy_boundary = [fuzzy_boundary0, fuzzy_boundary1, fuzzy_boundary2]
    return lower_approximation, fuzzy_boundary, fuzzy_membership_vij, fuzzy_max_set
def centroid_u(c, hist, fuzzy_membership_vij, lower_approximation, fuzzy_boundary, m):

    centroid_set = []
    for i in range(c):
        p1 = 0
        q1 = 0
        for k in range(len(lower_approximation[i])):
            nn = lower_approximation[i][k]
            p1 = p1 + np.sin(nn * np.pi / 180) * hist[0, nn]
            q1 = q1 + np.cos(nn * np.pi / 180) * hist[0, nn]
        p2 = 0
        q2 = 0
        for l in range(len(fuzzy_boundary[i])):
            n2 = fuzzy_boundary[i][l]
            p2 = p2 + (fuzzy_membership_vij[i, n2] ** m) * np.sin(n2 * np.pi / 180) * hist[0, n2]
            q2 = q2 + (fuzzy_membership_vij[i, n2] ** m) * np.cos(n2 * np.pi / 180) * hist[0, n2]

        qq = math.atan((0.55 * p1 + 0.45 * p2 + 0.0001)/(0.55 * q1 + 0.45 * q2 + 0.001))
        centroid_set.append(qq)
    return centroid_set


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

        a = np.dot(W.T, V)
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
        a = int(centroid_set[i] * 180 / np.pi)

        b1 = 0
        b2 = 0
        b3 = 0
        for j in range(h0):
            for k in range(w0):
                if a == h[j, k]:
                    b1 = b1 + s[j, k]
                    b2 = b2 + s[j, k] * ii[j, k]
                    b3 = b3 + s[j, k] ** 2
        for mm in range(1, 300):
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
        h = centroid_set[x]
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

def spectrum_matrix(I_i_bgr, c):
    W = np.zeros([3, c], np.float32)
    for i in range(3):
        for j in range(c):
            if I_i_bgr[i, j] != 0:
                W[i, j] = np.log(255 / I_i_bgr[i, j])
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
    if np.max(H) > 50:
        mi = np.min(H)
        H = (H + (-mi))/np.max(H)
    else:
        mi = np.min(H)
        H = (H + (-mi))
    return H

def t_optimal(I_i_bgr, b, g, r, c, h0, w0):
    W = spectrum_matrix(I_i_bgr, c)
    H = density(b, g, r, h0, w0, W)
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
    W1, H1 = train(BGR_I, W, H, c, 100, 1e-5)
    return W1, H1

def normalization(M, D, h0, w0):
    img_normal = np.multiply(255, np.exp(-M.dot(D)))
    img_normal[img_normal > 255] = 255
    img_normal = np.reshape(img_normal.T, (h0, w0, 3)).astype(np.uint8)

    H = np.multiply(255, np.exp(np.expand_dims(-M[:, 0], axis=1).dot(np.expand_dims(D[0, :], axis=0))))
    H[H > 255] = 255
    H = np.reshape(H.T, (h0, w0, 3)).astype(np.uint8)

    E = np.multiply(255, np.exp(np.expand_dims(-M[:, 1], axis=1).dot(np.expand_dims(D[1, :], axis=0))))
    E[E > 255] = 255
    E = np.reshape(E.T, (h0, w0, 3)).astype(np.uint8)


    return img_normal, H, E


def trans(centroid_set):
    ce = []
    if 0 < centroid_set[0] <= np.pi / 2:
        s1 = (360 - centroid_set[0] * 180 / np.pi) * np.pi / 180
        ce.append(s1)
    else:
        s1 = (360 + centroid_set[0] * 180 / np.pi) * np.pi / 180
        ce.append(s1)

    if 0 < centroid_set[1] <= np.pi / 2:
        s2 = (360 - centroid_set[1] * 180 / np.pi) * np.pi / 180
        ce.append(s2)
    else:
        s2 = (360 + centroid_set[1] * 180 / np.pi) * np.pi / 180
        ce.append(s2)


    s3 = abs(centroid_set[2])
    ce.append(s3)
    return ce
def ac(achromatic_ui):
    if 0 < achromatic_ui <= np.pi / 2:
        s1 = achromatic_ui
    else:
        s1 = (360 + achromatic_ui * 180 / np.pi) * np.pi / 180

    return s1

def template():
    img00 = cv2.imread('.\.', 1)
    img00 = cv2.resize(img00, (256, 256), interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(img00)
    h0, w0, alpha = img00.shape
    h, s, v = rgbtohsi(img00)
    stain_achromatic_T = 220
    c = 3
    m = 2
    la = 0.1
    hist = saturation_weighted_hue(h, s, h0, w0)
    achromatic_ui = achromatic_u(hist, stain_achromatic_T, h, s, v, h0, w0)
    achromatic_ui = ac(achromatic_ui)
    u = [4.6, 5.0, achromatic_ui]
    fuzzy_membership_vij = fuzzy_membership_v(u, c, m, hist)
    first_max_set1 = []
    for i in range(len(hist.T)):
        first_max_set1.append(0)
    for ij in range(100):
        lower_approximation, fuzzy_boundary, fuzzy_membership_vij1, fuzzy_max_set = sort(fuzzy_membership_vij, hist, la)
        centroid_set = centroid_u(c, hist, fuzzy_membership_vij1, lower_approximation, fuzzy_boundary, m)
        centroid_set = trans(centroid_set)
        first_max_set2 = fuzzy_max_set
        vij_difference = []
        for i in range(len(fuzzy_max_set)):
            vij_difference.append(abs(first_max_set2[i] - first_max_set1[i]))
        fuzzy_membership_vij = fuzzy_membership_v(centroid_set, c, m, hist)
        first_max_set1 = first_max_set2
        if np.max(vij_difference) < 0.01:
            break
    ss, vv = s_v(centroid_set, c, h, s, v, h0, w0)
    I_i_bgr = hsi_rgb(centroid_set, ss, vv, c)
    W, H = t_optimal(I_i_bgr, b, g, r, c, h0, w0)
    return W, img00

if __name__ == '__main__':
    file_name = '.\.'
    img = cv2.imread(file_name, 1)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    h0, w0, alpha = img.shape
    h_t, s_t, v_t = rgbtohsi(img)
    b1, g1, r1 = cv2.split(img)
    stain_achromatic_T = 220
    c = 3
    m = 2
    la = 0.1
    hist = saturation_weighted_hue(h_t, s_t, h0, w0)
    hist = hist
    achromatic_ui = achromatic_u(hist, stain_achromatic_T, h_t, s_t, v_t, h0, w0)
    achromatic_ui = ac(achromatic_ui)
    u = [4.6, 5.0, achromatic_ui]
    fuzzy_membership_vij = fuzzy_membership_v(u, c, m, hist)
    first_max_set1 = []
    for i in range(len(hist.T)):
        first_max_set1.append(0)
    for ij in range(100):
        lower_approximation, fuzzy_boundary, fuzzy_membership_vij1, fuzzy_max_set = sort(fuzzy_membership_vij, hist, la)
        centroid_set = centroid_u(c, hist, fuzzy_membership_vij1, lower_approximation, fuzzy_boundary, m)
        centroid_set = trans(centroid_set)
        first_max_set2 = fuzzy_max_set
        vij_difference = []
        for i in range(len(fuzzy_max_set)):
            vij_difference.append(abs(first_max_set2[i] - first_max_set1[i]))
        fuzzy_membership_vij = fuzzy_membership_v(centroid_set, c, m, hist)
        first_max_set1 = first_max_set2
        if np.max(vij_difference) < 2:
            break
    ss, vv = s_v(centroid_set, c, h_t, s_t, v_t, h0, w0)
    I_i_bgr = hsi_rgb(centroid_set, ss, vv, c)
    W1, H = t_optimal(I_i_bgr, b1, g1, r1, c, h0, w0)
    W, img00 = template()
    img_normal, HH, EE = normalization(W, H, h0, w0)
    cv2.imshow('1', img_normal)
    cv2.waitKey(0)
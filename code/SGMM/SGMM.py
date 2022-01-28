import time
import numpy as np
import copy
from collections import defaultdict
from sklearn.cluster import KMeans
import cv2
import scipy.stats as st
from numpy import *
from sklearn.cluster import MeanShift, estimate_bandwidth
import os
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

def achromatic_saturation_weighted(stain_achromatic_T, h, s, v, h0, w0):
    stain_h, stain_s, achromatic_h, achromatic_s = stain_achromatic_partition(stain_achromatic_T, h, s, v, h0, w0)
    hist1 = np.zeros([1, 361], np.float32)
    for i in range(len(stain_h)):
        theta = stain_h[i]
        hist1[0, theta] = hist1[0, theta] + stain_s[i]
    hist1 = hist1 / sum(sum(hist1))  
    n1 = len(stain_h)
    hist2 = np.zeros([1, 361], np.float32)
    for i in range(len(achromatic_h)):
        theta = achromatic_h[i]
        hist2[0, theta] = hist2[0, theta] + achromatic_s[i]
    hist2 = hist2 / sum(sum(hist2))
    n2 = len(achromatic_h)
    return hist1, n1, hist2, n2
def stain_achromatic_partition(stain_achromatic_T, h, s, I, h0, w0):
    stain_h = []
    stain_s = []
    achromatic_h = []
    achromatic_s = []
    for i in range(h0):
        for j in range(w0):
            if I[i, j] > stain_achromatic_T:
                achromatic_h.append(h[i, j])
                achromatic_s.append(s[i, j])
            else:
                stain_h.append(h[i, j])
                stain_s.append(s[i, j])
    return stain_h, stain_s, achromatic_h, achromatic_s
def hue(h, s, h0, w0):
    hist1 = np.zeros([1, 361], np.float32)
    for i in range(h0):
        for j in range(w0):
          thet = h[i, j]
          hist1[0, thet] = hist1[0, thet] + s[i, j]
    hist1 = hist1 / sum(sum(hist1))
    return hist1

class GEM:
    def __init__(self, maxstep=1000, epsilon=10 ** (-5)):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = None  

        self.alpha = None  
        self.mu = None  
        self.sigma = None  
        self.gamma_all_final = None  

        self.D = None  
        self.N = None  

    def inin_param(self, data_s,data_a, hist, su, aa):
        self.D = hist.shape[1]
        self.N = hist.shape[0]
        self.init_param_helper(data_s,data_a, su, aa)
        return

    def init_param_helper(self, data_s,data_a, su, aa):
        KMEANS = KMeans(n_clusters=2).fit(data_s)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_):
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        skew = []
        for inds in clusters.values():
            partial_data = data_s[inds]
            mu.append(partial_data.mean(axis=0))
            alpha.append(len(inds) / su)
            sigma.append(np.std(partial_data.T))
            counts = np.bincount([int(numbers) for numbers in partial_data.flatten()])
            skew.append((partial_data.mean(axis=0) - np.argmax(counts)) / partial_data.std(axis=0))


        if aa == 12:
            bandwidth = estimate_bandwidth(data_a, quantile=0.7, n_samples=500)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data_a)
            clusters = defaultdict(list)
            if len(ms.cluster_centers_) <= 2:
                for ind, label in enumerate(ms.labels_):
                    clusters[label].append(ind)
                for inds in clusters.values():
                    partial_data = data_a[inds]
                    mu.append(partial_data.mean(axis=0))
                    alpha.append(len(inds) / su)
                    sigma.append(np.std(partial_data.T))
                    counts = np.bincount([int(numbers) for numbers in partial_data.flatten()])
                    skew.append(((partial_data.mean(axis=0) - np.argmax(counts))) / (partial_data.std(axis=0)))
            else:
                KMEANS = KMeans(n_clusters=1).fit(data_a)
                clusters = defaultdict(list)
                for ind, label in enumerate(KMEANS.labels_):
                   clusters[label].append(ind)
                for inds in clusters.values():
                   partial_data = data_a[inds]
                   mu.append(partial_data.mean(axis=0))
                   alpha.append(len(inds) / su)
                   sigma.append(np.std(partial_data.T))
                   counts = np.bincount([int(numbers) for numbers in partial_data.flatten()])
                   skew.append(((partial_data.mean(axis=0) - np.argmax(counts))) / (partial_data.std(axis=0) ))
        else:
            bandwidth = estimate_bandwidth(data_a, quantile=0.7, n_samples=500)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(data_a)
            clusters = defaultdict(list)
            if len(ms.cluster_centers_) == aa:
                for ind, label in enumerate(ms.labels_):
                    clusters[label].append(ind)
                for inds in clusters.values():
                    partial_data = data_a[inds]
                    mu.append(partial_data.mean(axis=0))
                    alpha.append(len(inds) / su) 
                    sigma.append(np.std(partial_data.T))
                    counts = np.bincount([int(numbers) for numbers in partial_data.flatten()])
                    skew.append((partial_data.mean(axis=0) - np.argmax(counts)) / partial_data.std(axis=0))
            else:
                KMEANS = KMeans(n_clusters=aa).fit(data_a)
                clusters = defaultdict(list)
                for ind, label in enumerate(KMEANS.labels_):
                    clusters[label].append(ind)
                for inds in clusters.values():
                    partial_data = data_a[inds]
                    mu.append(partial_data.mean(axis=0))
                    alpha.append(len(inds) / su)  
                    sigma.append(np.std(partial_data.T))
                    counts = np.bincount([int(numbers) for numbers in partial_data.flatten()])
                    skew.append((partial_data.mean(axis=0) - np.argmax(counts)) / partial_data.std(axis=0))

        self.mu = np.array(mu)

        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        self.skew = np.array(skew)
        self.delta = np.array(skew)
        self.tau = np.array(skew)
        self.K = len(self.mu)
  
        return

    def _phi(self, y, mu, sigma, skew):
        
        a = st.norm.pdf(y, loc=mu, scale=np.sqrt(sigma))
        b = st.norm.cdf(skew * (y - mu) / math.sqrt(sigma), loc=0, scale=1)  
        return 2 * a * b  

    def fit(self, data_s,data_a, hist, su, aa):
       
        self.inin_param(data_s,data_a, hist, su, aa)
        step = 0
        L1 = 0
        while step < self.maxstep:

            step += 1
            # E步
            gamma_z = []
            gamma_t = []
            gamma_t2 = []
            L = 0
            for j in range(self.N):
                gamma_jz = []    
                gamma_jt = []
                gamma_jt2 = []

                l = 0

                for k in range(self.K):#聚类的个数
                    gamma_jz.append(self.alpha[k] * self._phi(j, self.mu[k], self.sigma[k], self.skew[k]))

                    a1 = ((self.skew[k] / np.sqrt(self.sigma[k])) * (j - self.mu[k])) / np.sqrt(1 + self.skew[k] ** 2)
                    a2 = 1.0 / np.sqrt(1 + self.skew[k] ** 2)
                    f1 = st.norm.pdf(a1 / a2, loc=0, scale=1)
                    f2 = st.norm.cdf(a1 / a2, loc=0, scale=1)
                    gamma_jt.append(a1 + (f1 / (f2 + 10 ** (-5))) * a2)
                    gamma_jt2.append(a1 ** 2 + a2 ** 2 + (f1 / (f2 + 10 ** (-5))) * a1 * a2)
                    l = l + self.alpha[k] * self._phi(j, self.mu[k], self.sigma[k], self.skew[k])

                ss = sum(gamma_jz)
                # print(ss)
                gamma_jz = [item / (ss + 10 ** (-5)) for item in gamma_jz]
                gamma_jt = [item / 1.0 for item in gamma_jt]
                gamma_jt2 = [item / 1.0 for item in gamma_jt2]

                gamma_z.append(gamma_jz)
                gamma_t.append(gamma_jt)
                gamma_t2.append(gamma_jt2)

                L = L + np.log(1 + l) * hist[j, 0]


            gamma_z_arr = np.array(gamma_z)
            gamma_t_arr = np.array(gamma_t)
            gamma_t2_arr = np.array(gamma_t2)



            
            for k in range(self.K):
                gamma_kz = gamma_z_arr[:, k]
                gamma_kt = gamma_t_arr[:, k]
                gamma_kt2 = gamma_t2_arr[:, k]
            





                SUM = 0
                for j in range(self.N):
                    SUM = SUM + gamma_kz[j, 0] * hist[j, 0]
               
                self.alpha[k] = SUM / sum(sum(hist))  

               

                dd = np.sqrt(self.sigma[k]) * self.skew[k] / np.sqrt(1 + self.skew[k] ** 2)

                q1 = 0
                for j in range(self.N):
                    q1 = q1 + gamma_kz[j, 0] * (j - dd * gamma_kt[j, 0]) * hist[j, 0]
                me = q1 / SUM
                #if me >= 0:
                self.mu[k] = abs(me)

                q1 = 0
                q2 = 0
                for j in range(self.N):
                    q1 = q1 + (gamma_kz[j, 0] * gamma_kt[j, 0] * (j - self.mu[k])) * hist[j, 0]
                    q2 = q2 + gamma_kz[j, 0] * gamma_kt2[j, 0] * hist[j, 0]
                self.delta[k] = q1 / q2

                q1 = 0
                for j in range(self.N):
                    q1 = q1 + gamma_kz[j, 0] * ((j - self.mu[k]) ** 2 - 2 * (j - self.mu[k]) * self.delta[k] * gamma_kt[j,0] + self.delta[k] ** 2 * gamma_kt2[j, 0]) * hist[j, 0]
                self.tau[k] = q1 / SUM

                self.sigma[k] = self.tau[k] + self.delta[k] ** 2

                self.skew[k] = (self.delta[k] / np.sqrt(self.sigma[k])) / np.sqrt(1 - (self.delta[k] ** 2) / self.sigma[k])

            if abs(L[0] - L1) < self.epsilon:
                break
            L1 = copy(L[0])

        return
def train(V, W, H, r, k, e):
    m, n = np.shape(V)
    for x in range(k):
        V_pre = np.dot(W, H)
        E = V - V_pre
        err = np.sum(E * E)
        if err < e:
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

    mi = np.min(H)
    mx = np.max(H)
    H = H + (-mi)
    H = H/mx
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
    W1, H1 = train(BGR_I, W, H, c,50, 1e-3)
    return W1, H1



def normalization(M, D, h0, w0):
    img_normal = np.multiply(255, np.exp(-M.dot(D)))
    img_normal[img_normal > 255] = 255
    img_normal = np.reshape(img_normal.T, (h0, w0, 3)).astype(np.uint8)
    return img_normal


def template(c):
    img00 = cv2.imread('.\.', 1)
    img00 = cv2.resize(img00, (256, 256), interpolation=cv2.INTER_CUBIC)
    h0 = 256
    w0 = 256
    b, g, r = cv2.split(img00)
    h, s, v = rgbtohsi(b, g, r)
    hist = hue(h, s, h0, w0)
    stain_achromatic_T = 220
    hist1, n1, hist2, n2 = achromatic_saturation_weighted(stain_achromatic_T, h, s, v, h0, w0)
    hist = hist.T
    hist1 = hist1.T
    hist2 = hist2.T
    data01 = []
    for i in range(361):
        if hist1[i, 0] != 0:
            for j in range(int(np.around(n1 * hist1[i, 0]))):
                data01.append(i)
    data_s = np.zeros([len(data01), 1], np.float32)
    for i in range(len(data01)):
        data_s[i, 0] = data01[i]
    data02 = []
    for i in range(361):
        if hist2[i, 0] != 0:
            for j in range(int(np.around(n2 * hist2[i, 0]))):
                data02.append(i)
    data_a = np.zeros([len(data02), 1], np.float32)
    for i in range(len(data02)):
        data_a[i, 0] = data02[i]
    su = sum(len(data01) + len(data02))
    gem = GEM()
    aa = c - 2
    gem.fit(data_s,data_a, hist, su, aa)
    c1 = len(gem.mu)
    ss, vv = s_v(gem.mu, c1, h, s, v, h0, w0)
    I_i_bgr = hsi_rgb(gem.mu, ss, vv,c)
    W, H = t_optimal(I_i_bgr, b, g, r, c, h0, w0)
    return W, img00

if __name__ == '__main__':
    file_name1 = ".\."
    img01 = cv2.imread(file_name1, 1)
    img01 = cv2.resize(img01, (256, 256), interpolation=cv2.INTER_CUBIC)
    h0, w0, alpha = img01.shape
    b, g, r = cv2.split(img01)
    b = (255 * (b / np.max(b))).astype(np.uint8)
    g = (255 * (g / np.max(g))).astype(np.uint8)
    r = (255 * (r / np.max(r))).astype(np.uint8)
    h, s, v = rgbtohsi(b, g, r)
    hist = hue(h, s, h0, w0)
    stain_achromatic_T = 220
    hist1, n1, hist2, n2 = achromatic_saturation_weighted(stain_achromatic_T, h, s, v, h0, w0)
    hist = hist.T
    hist1 = hist1.T
    hist2 = hist2.T
    data01 = []
    for i in range(361):
        if hist1[i, 0] != 0:
            for j in range(int(np.around(n1 * hist1[i, 0]))):
                data01.append(i)
    data_s = np.zeros([len(data01), 1], np.float32)
    for i in range(len(data01)):
        data_s[i, 0] = data01[i]
    data02 = []
    for i in range(361):
        if hist2[i, 0] != 0:
            for j in range(int(np.around(n2 * hist2[i, 0]))):
                data02.append(i)
    data_a = np.zeros([len(data02), 1], np.float32)
    for i in range(len(data02)):
        data_a[i, 0] = data02[i]
    su = sum(len(data01) + len(data02))
    gem = GEM()
    aa = 12
    gem.fit(data_s, data_a, hist, su, aa)
    c = len(gem.mu)
    ss, vv = s_v(gem.mu, c, h, s, v, h0, w0)
    I_i_bgr = hsi_rgb(gem.mu, ss, vv, c)
    W1, H = t_optimal(I_i_bgr, b, g, r, c, h0, w0)
    W, img00 = template(c)
    img_normal = normalization(W, H, h0, w0)
    cv2.imshow('1', img_normal)
    cv2.waitKey(0)
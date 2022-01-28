import argparse
import numpy as np
from PIL import Image
import cv2
import os
import time
def normalizeStaining(img, Io=255, alpha=1, beta=0.15):

    HERef = np.array([[3.4379053e-01, 3.4907752e-01],
                     [2.0779057e+00, 1.5549829e+00],
                     [1.6621019e+00, 3.1027577e-03]])

    maxCRef = np.array([1, 1])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = np.log(Io / (img.astype(np.float) + 1))

    ODhat = OD[~np.any(OD < beta, axis=1)]

    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two

    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(HE, Y, rcond=None)[0]


    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])



    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))

    Inorm[Inorm > 255] = 255
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 255
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 255
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    Inorm1 = cv2.cvtColor(Inorm, cv2.COLOR_BGR2RGB)
    H1 = cv2.cvtColor(H, cv2.COLOR_BGR2RGB)
    E1 = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
    cv2.imshow('i', Inorm1)
    cv2.imshow('h', H1)
    cv2.imshow('e', E1)
    cv2.waitKey(0)


    return Inorm, H, E


if __name__ == '__main__':
    file_name = '.\\.'
    img = cv2.imread(file_name, 1)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    re, re1, re2= normalizeStaining(img)




import random
import math
import numpy as np
import scipy
import time
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt

def getyhat(W1, W2, b1, b2, x):
    newx = inner(W1, x, b1)
    yhat = activate(outer(W2, newx, b2))
    return yhat

def buildMatrix(dim1, dim2):
    return np.random.rand(dim1, dim2)

def inner(W, x, b):
    W = np.asarray(W)
    inner1 = W.transpose() * x + b
    for i in range(0, inner1.shape[0]):
        inner1[i] = activate(inner1[i])
    return inner1

def outer(W, x, b):

    W = np.asarray(W)
    x = np.asarray(x)
    b = np.asarray(b)
    resp = np.dot(W.transpose(), x)
    return resp + b


def term1(W1, W2, x, y, b1, b2):

    interm = inner(W1, x, b1)
    outerm = outer(W2, interm, b2)
    outerm = activate(outerm)
    finterm = outerm - y
    return finterm

def term2(W1, W2, x, b1, b2):

    interm = inner(W1, x, b1)
    outerm = outer(W2, interm, b2)
    outerm = activate(outerm)
    finterm = outerm*(1 - outerm)

    return finterm

def calcWd2(W1, W2, x, y, b1, b2):
    W2term1 = term1(W1, W2, x, y, b1, b2)
    W2term2 = term2(W1, W2, x, b1, b2)
    W2term3 = W1.transpose() * (x) + b1
    finterm = W2term1 * W2term2 * W2term3
    return finterm

def calcbd2(W1, W2, x, y, b1, b2):
    W2term1 = term1(W1, W2, x, y, b1, b2)
    W2term2 = term2(W1, W2, x, b1, b2)
    return W2term1 * W2term2

def calcWd1(W1, W2, x, y, b1, b2):
    Wterm1 = term1(W1, W2, x, y, b1, b2)
    Wterm2 = term2(W1, W2, x, b1, b2)
    w1term1 = Wterm1 * Wterm2
    w1term2 = W2.transpose()
    wterm3 = W1term3(W1, x, b1)
    finterm = w1term1 * np.dot(np.asarray(w1term2), np.asarray(wterm3))
    return finterm

def calcbd1(W1, W2, x, y, b1, b2):
    Wterm1 = term1(W1, W2, x, y, b1, b2)
    Wterm2 = term2(W1, W2, x, b1, b2)
    w1term1 = Wterm1 * Wterm2
    w1term2 = W2.transpose()
    bterm3 = b1term3(W1, x, b1)
    finterm = (w1term1 * w1term2 * bterm3.transpose()).transpose()
    return finterm

def W1term3(W, x, b):

    W = np.asarray(W)
    t1 = np.zeros((b.shape[0], b.shape[0]))
    for i in range(0, b.shape[0]):
        a = W[0][i] * x
        a = a + b[i]
        t1[i][i] = activate(a) * (1 - activate(a))
    t2 = np.zeros((b.shape[0], b.shape[0]))
    for i in range(0, b.shape[0]):
        t2[i][i] = x
    finterm = np.dot(t1,t2)
    return finterm

def b1term3(W, x, b):
    t1 = np.zeros(b.shape)
    for i in range(0, t1.shape[0]):
        a = W[0][i] * x + b[i]
        t1[i] = activate(a)*(1 - activate(a))
    return t1

def activate(x):
    t1 = 1 + math.exp(-x)
    return 1/t1


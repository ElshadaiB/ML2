import random
import numpy as np
import scipy
import time
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt

import function_lib as f

def train(x_data, y_data, k, stp):
    W1 =f.buildMatrix(1, k)
    W2 = f.buildMatrix(k, 1)
    b1 = f.buildMatrix(k, 1)
    b2 = f.buildMatrix(1, 1)
    Wd1 = f.buildMatrix(1, k)
    Wd2 = f.buildMatrix(k, 1)
    bd1 = f.buildMatrix(k, 1)
    bd2 = f.buildMatrix(1, 1)

    oWd1 = f.buildMatrix(1, k)
    oWd2 = f.buildMatrix(k, 1)
    obd1 = f.buildMatrix(k, 1)
    obd2 = f.buildMatrix(1, 1)
    epoch = 0
    bsum = 0

    while 1:
        epoch = epoch + 1
        print("training epoch: ", epoch)
        Wd1.fill(0)
        Wd2.fill(0)
        bd1.fill(0)
        bd2.fill(0)
        n = len(x_data)

        sum = 0
        for i in range(n):
            yhat = f.getyhat(W1, W2, b1, b2, x_data[i])
            sum = sum + (yhat - y_data[i])*(yhat - y_data[i])

            W2term1 = f.term1(W1, W2, x_data[i], y_data[i], b1, b2)
            W2term2 = f.term2(W1, W2, x_data[i], b1, b2)
            W2term3 = W1.transpose()*(x_data[i]) + b1

            Wd2 = Wd2 + (W2term1 * W2term2 * W2term3)

            W1term1 = W2term1 * W2term2
            W1term2 = W2.transpose()
            W1term3 = f.W1term3(W1, x_data[i], b1)

            Wd1 = Wd1 + (W1term1 * np.dot(np.asarray(W1term2), np.asarray(W1term3)))

            bd2 = bd2 + W2term1 * W2term2
            b1term3 = f.b1term3(W1, x_data[i], b1)
            bd1 = bd1 + (W1term1 * W1term2 * b1term3.transpose()).transpose()

        sum = sum/(2*n)
        print("Loss: ", sum)
        print("Accuracy %", 100 * (1-sum))
        print("Loss diff: ", abs(bsum - sum))
        if abs(sum - bsum) < 0.0009 and epoch > 10:
            break
        bsum = sum
        Wd1  = Wd1/n
        Wd2 = Wd2/n
        bd1 = bd1/n
        bd2 = bd2/n

        W1 = W1 -  stp * Wd1
        W2 = W2 - stp * Wd2
        b1 = b1 - stp * bd1
        b2 = b2 - stp * bd2

    params = {}
    params["W1"] = W1
    params["W2"] = W2
    params["b1"] = b1
    params["b2"] = b2

    return params


if __name__ == '__main__':
    orig_stdout = sys.stdout
    k = sys.argv[1]
    stp = sys.argv[2]
    mat = loadmat("hw2data_2.mat")
    x = mat['X']
    y = mat['Y']

    params = train(x, y, int(k), float(stp))
    print("Trained!")
    yhats = []

    for i in range(len(x)):
        temp = f.getyhat(params['W1'], params['W2'], params['b1'], params['b2'], x[i])
        yhats.append(temp)

    plt.scatter(x, yhats, color="r")
    plt.scatter(x, y, color="b")
    plt.show()



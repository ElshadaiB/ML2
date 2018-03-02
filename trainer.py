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

    epoch = 0
    bsum = 0

    while 1:
        epoch = epoch + 1
        print "training epoch: ", epoch
        print " "
        Wd1.fill(0)
        Wd2.fill(0)
        bd1.fill(0)
        bd2.fill(0)
        n = len(x_data)

        sum = 0
        acc = 0
        for i in range(n):
            yhat = f.getyhat(W1, W2, b1, b2, x_data[i])
            sum = sum + (yhat - y_data[i])*(yhat - y_data[i])
            acc = acc + abs(yhat - y_data[i])

            Wd2 = Wd2 + f.calcWd2(W1, W2, x_data[i], y_data[i], b1, b2)

            Wd1 = Wd1 + f.calcWd1(W1, W2, x_data[i], y_data[i], b1, b2)

            bd2 = bd2 + f.calcbd2(W1, W2, x_data[i], y_data[i], b1, b2)

            bd1 = bd1 + f.calcbd1(W1,W2, x_data[i], y_data[i], b1, b2)

        sum = sum/(2*n)
        acc = acc/n
        print "Loss: ", sum[0]
        print "Accuracy %", (100 * (1-sum))[0]
        print "Avg difference: ", acc[0]
        print "Loss diff: ", (abs(bsum - sum))[0]
        print " "

        if abs(sum - bsum) < 0.0005 and epoch > 15:
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



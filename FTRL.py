# Date: 2018-08-17 9:47
# Author: Enneng Yang
# Abstract：ftrl ref:http://www.cnblogs.com/zhangchaoyang/articles/6854175.html

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import tensorflow as tf

# logistic regression
class LR(object):

    @staticmethod
    def fn(w, x):
        ''' sigmod function '''
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        '''cross-entropy loss function'''
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1-y)*np.log(1-y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        '''gradient function'''
        return (y_hat - y) * x

class FTRL(object):

    def __init__(self, dim, l1, l2, alpha, beta, decisionFunc=LR):
        self.dim = dim
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.decisionFunc = decisionFunc

        self.z = np.zeros(dim)
        self.q = np.zeros(dim)
        self.w = np.zeros(dim)

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1
                             else (np.sign(self.z[i] * self.l1) * self.l1 - self.z[i]) / (self.l2 + (self.beta + np.sqrt(self.q[i]))/self.alpha)
                             for i in range(self.dim)])


        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.q + g*g) - np.sqrt(self.q)) / self.alpha
        self.z += g - sigma * self.w
        self.q += g*g
        return self.decisionFunc.loss(y,y_hat)

    def training(self, trainSet, max_itr=100000):
        n = 0

        all_loss = []
        all_step = []
        while True:
            for var in trainSet:
                x= var[:4]
                y= var[4:5]
                loss = self.update(x, y)

                all_loss.append(loss)
                all_step.append(n)

                print("itr=" + str(n) + "\tloss=" + str(loss))

                n += 1
                if n > max_itr:
                    print("reach max iteration", max_itr)
                    return all_loss, all_step

if __name__ ==  '__main__':

    d = 4
    trainSet = np.loadtxt('Data/FTRLtrain.txt')
    train_x = trainSet[:, 3]
    train_y = trainSet[:, 4]
    ftrl = FTRL(dim=d, l1=0.1, l2=0.1, alpha=0.1, beta=1.0)
    all_loss, all_step = ftrl.training(trainSet,  max_itr=100000)
    w = ftrl.w
    print(w)

    testSet = np.loadtxt('Data/FTRLtest.txt')
    correct = 0
    wrong = 0
    for var in testSet:
        x = var[:4]
        y = var[4:5]
        y_hat = 1.0 if ftrl.predict(x) > 0.5 else 0.0
        if y == y_hat:
            correct += 1
        else:
            wrong += 1
    print("correct ratio:", 1.0 * correct / (correct + wrong), "\t correct:", correct, "\t wrong:", wrong)

    plt.title('FTRL')
    plt.xlabel('training_epochs')
    plt.ylabel('loss')
    plt.plot(all_step, all_loss)
    plt.show()


# Date: 2018-08-17 8:47
# Author: Enneng Yang
# Abstract：OGD

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

class OGD(object):

    def __init__(self,alpha,decisionFunc=LR):
        self.alpha = alpha
        self.w = np.zeros(4)
        self.decisionFunc = decisionFunc

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y,step):
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        learning_rate = self.alpha / np.sqrt(step + 1)  # damping step size
        # SGD Update rule theta = theta - learning_rate * gradient
        self.w = self.w - learning_rate * g
        return self.decisionFunc.loss(y,y_hat)

    def training(self, trainSet, max_itr=100000):
        n = 0

        all_loss = []
        all_step = []
        while True:
            for var in trainSet:
                x= var[:4]
                y= var[4:5]
                loss = self.update(x, y,n)

                all_loss.append(loss)
                all_step.append(n)

                print("itr=" + str(n) + "\tloss=" + str(loss))

                n += 1
                if n > max_itr:
                    print("reach max iteration", max_itr)
                    return all_loss, all_step

if __name__ ==  '__main__':

    trainSet = np.loadtxt('Data/FTRLtrain.txt')
    OGD = OGD(alpha=0.01)
    all_loss, all_step = OGD.training(trainSet,  max_itr=100000)
    w = OGD.w
    print(w)

    testSet = np.loadtxt('Data/FTRLtest.txt')
    correct = 0
    wrong = 0
    for var in testSet:
        x = var[:4]
        y = var[4:5]
        y_hat = 1.0 if OGD.predict(x) > 0.5 else 0.0
        if y == y_hat:
            correct += 1
        else:
            wrong += 1
    print("correct ratio:", 1.0 * correct / (correct + wrong), "\t correct:", correct, "\t wrong:", wrong)

    plt.title('OGD')
    plt.xlabel('training_epochs')
    plt.ylabel('loss')
    plt.plot(all_step, all_loss)
    plt.show()


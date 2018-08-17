# Date: 2018-08-17 8:47
# Author: Enneng Yang
# Abstract：Truncated Gradient

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

class TG(object):

    def __init__(self,K,alpha,theta,lambda_,decisionFunc=LR):
        self.K = K #to zero after every K online steps
        self.alpha = alpha # learning rate
        self.theta = theta # threshold value
        self.lambda_ = lambda_ #
        self.w = np.zeros(4) # param
        self.decisionFunc = decisionFunc #decision Function

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y, step):
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)

        if step % self.K == 0:

           learning_rate = self.alpha / np.sqrt(step+1) # damping step size

           temp_lambda = self.K * self.lambda_

           for i in range(4):
              w_e_g = self.w[i] -learning_rate * g[i]
              if (0< w_e_g <self.theta) :
                  self.w[i] = max(0, w_e_g - learning_rate * temp_lambda)
              elif (-self.theta< w_e_g <0) :
                  self.w[i] = max(0, w_e_g + learning_rate * temp_lambda)
              else:
                  self.w[i] = w_e_g
        else:
            # SGD Update rule theta = theta - learning_rate * gradient
            self.w = self.w - self.alpha * g




        return self.decisionFunc.loss(y,y_hat)

    def training(self, trainSet, max_itr=100000):
        n = 0

        all_loss = []
        all_step = []
        while True:
            for var in trainSet:
                x= var[:4]
                y= var[4:5]
                loss = self.update(x, y, n)

                all_loss.append(loss)
                all_step.append(n)

                print("itr=" + str(n) + "\tloss=" + str(loss))

                n += 1
                if n > max_itr:
                    print("reach max iteration", max_itr)
                    return all_loss, all_step

if __name__ ==  '__main__':

    trainSet = np.loadtxt('Data/FTRLtrain.txt')
    TG = TG(K=5,alpha=0.01,theta=0.001,lambda_=1.)
    all_loss, all_step = TG.training(trainSet,  max_itr=100000)
    w = TG.w
    print(w)

    testSet = np.loadtxt('Data/FTRLtest.txt')
    correct = 0
    wrong = 0
    for var in testSet:
        x = var[:4]
        y = var[4:5]
        y_hat = 1.0 if TG.predict(x) > 0.5 else 0.0
        if y == y_hat:
            correct += 1
        else:
            wrong += 1
    print("correct ratio:", 1.0 * correct / (correct + wrong), "\t correct:", correct, "\t wrong:", wrong)

    plt.title('Truncated Gradient')
    plt.xlabel('training_epochs')
    plt.ylabel('loss')
    plt.plot(all_step, all_loss)
    plt.show()




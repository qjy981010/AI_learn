#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import *
from evaluate import accuracy
from sklearn import svm


class SVM:

    def __init__(self, C=1, max_iter=1000, sigma=1.3, tol=0.0001):
        self.b = 0
        self.tol = tol  # fault tolerance
        self.C = C
        self.sigma = sigma
        self.max_iter = max_iter

    def calc_error(self, i):
        prediction = np.dot(self.alphas * self.labels,
                            self.kernel[:, i]) + self.b
        return prediction - self.labels[i]

    def random_j(self, i):
        j = np.random.randint(self.m)
        while j == i:
            j = np.random.randint(self.m)
        return j

    def get_alpha2(self, origin_error, i):
        errors = []
        self.ecache[i] = 1
        validlist = np.nonzero(self.ecache)[0]
        j = error = 0
        if len(validlist) > 1:
            for x in validlist:  # range(self.m):
                if (x != i):
                    errors.append((x, self.calc_error(x)))
            j, error = max(errors, key=lambda x: np.abs(x[1]-origin_error))
        else:
            j = self.random_j(i)
            error = self.calc_error(j)
        # for x in range(self.m):
        #     if (x != i):
        #         errors.append((x, self.calc_error(x)))
        # j, error = max(errors, key=lambda x: np.abs(x[1]-origin_error))
        return j, error

    def updateEk(self, i):
        self.ecache[i] = 1

    def get_L_H(self, i, j):
        L = H = 0
        if self.labels[i] != self.labels[j]:
            alpha_diff = self.alphas[j] - self.alphas[i]
            L = max(0, alpha_diff)
            H = min(self.C, self.C + alpha_diff)
        else:
            alpha_sum = self.alphas[i] + self.alphas[j]
            L = max(0, alpha_sum - self.C)
            H = min(self.C, alpha_sum)
        return L, H

    def get_b(self, i, j, error, diff1, diff2, is1):
        if is1:
            b = (self.b - self.labels[i]*self.kernel[i, i]*diff1 -
                 self.labels[j]*self.kernel[j, i]*diff2 - error)
        else:
            b = (self.b - self.labels[j]*self.kernel[j, j]*diff2 -
                 self.labels[i]*self.kernel[i, j]*diff1 - error)
        return b

    # Gaussian kernel
    def rbf_kernel(self):
        for i in range(self.m):
            for j in range(self.m):
                if j < i:
                    # for it's a diagonal matrix
                    self.kernel[i, j] = self.kernel[j, i]
                else:
                    self.kernel[i, j] = np.sum((self.data[i]-self.data[j])**2)
        self.kernel = np.exp(-self.kernel / 2 / self.sigma**2)

    def kernel_trans(self, SVs, X):
        m = SVs.shape[0]
        K = np.zeros(m)
        for i in range(m):
            K[i] = np.sum((SVs[i] - X)**2)
        K = np.exp(-K / 2 / self.sigma**2)
        return K

    def update(self, i):
        error = self.calc_error(i)
        if (self.labels[i] * error < -self.tol and self.alphas[i] < self.C or
                self.labels[i] * error > self.tol and self.alphas[i] > 0):
            j, error2 = self.get_alpha2(error, i)
            L, H = self.get_L_H(i, j)
            if L == H:
                return False
            eta = self.kernel[i, i] + self.kernel[j, j] - 2*self.kernel[i, j]
            if eta <= 0:
                return False
            newalpha2 = self.alphas[j] + self.labels[j] * (error-error2) / eta
            newalpha2 = np.clip(newalpha2, L, H)
            self.updateEk(j)
            alpha2_diff = newalpha2 - self.alphas[j]  # diff = new - old
            if (np.abs(alpha2_diff) < self.tol / 10):
                return False
            else:
                self.alphas[j] = newalpha2
            alpha1_diff = self.labels[i] * self.labels[j] * (-1 * alpha2_diff)
            self.alphas[i] += alpha1_diff
            self.updateEk(i)
            if 0 < self.alphas[i] < self.C:
                self.b = self.get_b(
                    i, j, error, alpha1_diff, alpha2_diff, True)
            elif 0 < self.alphas[j] < self.C:
                self.b = self.get_b(
                    i, j, error2, alpha1_diff, alpha2_diff, False)
            else:
                b1 = self.get_b(i, j, error, alpha1_diff, alpha2_diff, True)
                b2 = self.get_b(i, j, error2, alpha1_diff, alpha2_diff, False)
                self.b = (b1 + b2) / 2.0
            return True
        else:
            return False

    # SMO algorithm
    def smo(self):
        iternum = 0
        while iternum < self.max_iter:
            changed = False

            # get support vectors and others
            is_SVs = {True: [], False: []}
            for i in range(self.m):
                is_SVs[0 < self.alphas[i] < self.C].append(i)

            # update support vectors
            for i in is_SVs[True]:
                changed = self.update(i) or changed
            if changed:
                iternum += 1
                continue

            # update others if not changed
            for i in is_SVs[False]:
                changed = self.update(i) or changed
            iternum += 1

    def train(self, data, label):
        self.data = np.array(data)
        self.labels = np.array(label)
        self.m, self.n = self.data.shape
        self.kernel = np.zeros((self.m, self.m))
        self.ecache = np.zeros(self.m)
        self.alphas = np.zeros(self.m)
        self.rbf_kernel()
        self.smo()

    def evaluate(self, data, label):
        SV_indexes = self.alphas > 0
        SVs = self.data[SV_indexes]
        SV_labels = self.labels[SV_indexes]
        SV_alphas = self.alphas[SV_indexes]
        m = data.shape[0]
        error_count = 0
        for i in range(m):
            k = self.kernel_trans(SVs, data[i])
            prediction = np.dot(k, SV_alphas * SV_labels) + self.b
            if np.sign(prediction) != np.sign(label[i]):
                error_count += 1
        print('test accuracy: ', 1 - error_count / len(data))

    def draw(self):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        for i in range(self.m):
            if self.alphas[i] != 0:
                x3.append(self.data[i][0])
                y3.append(self.data[i][1])
            elif self.labels[i] == 1:
                x1.append(self.data[i][0])
                y1.append(self.data[i][1])
            else:
                x2.append(self.data[i][0])
                y2.append(self.data[i][1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x1, y1, s=10, c='red', marker='s')
        ax.scatter(x2, y2, s=10, c='green')
        ax.scatter(x3, y3, s=60, c='blue', marker=r'$\bigodot$')
        plt.show()


def linear_test():
    data = load_data(
        'data/binary_classification_for_svm_linear.csv', header=None, shuffle=False)
    data, label = data[:, :-1], data[:, -1]
    svc = SVM(C=0.3, max_iter=100, sigma=6, tol=0.001)
    svc.train(data, label)

    clf = svm.SVC(C=0.3, max_iter=100, tol=0.001)
    clf.fit(data, label)

    svc.draw()


def test():
    data = load_data('data/binary_classification_for_svm.csv',
                     header=None, shuffle=False)
    data, label = data[:, :-1], data[:, -1]
    svc = SVM(C=200, max_iter=1000, sigma=1.3)
    svc.train(data, label)

    clf = svm.SVC(C=200, max_iter=1000, tol=0.0001)
    clf.fit(data, label)

    test_data = load_data(
        'data/binary_classification_for_svm_test.csv', header=None, shuffle=False)
    test_data, test_label = test_data[:, :-1], test_data[:, -1]
    svc.evaluate(test_data, test_label)
    print('sklearn accuracy: ', clf.score(test_data, test_label))

    svc.draw()


if __name__ == '__main__':
    # linear_test()
    test()

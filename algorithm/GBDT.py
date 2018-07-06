#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from data_processing import load_data, train_test_split


class CartTree(object):
    """
    """
    def __init__(self, max_depth=3, file=None):
        super(CartTree, self).__init__()
        self.max_depth = max_depth
        self.data = None
        if file:
            self.tree = pickle.load(open(file, 'rb'))
        else:
            self.tree = {}

    def fit(self, data):
        self.data = data
        self.m = data.shape[0]
        self.n = data.shape[1] - 1
        self.tree = self._get_branch(data, 1)

    def _get_branch(self, data, depth):
        if depth >= self.max_depth:
            return np.mean(data[:, -1])
        if self._all_same(data[:, -1]):
            return data[0, -1]
        feat, val = self._get_split_point(data)
        branch = {}
        branch['feat'] = feat
        branch['val'] = val
        bool_list = data[:, feat] < val
        branch['left'] = self._get_branch(data[bool_list], depth+1)
        branch['right'] = self._get_branch(data[~bool_list], depth+1)
        return branch

    def _all_same(self, Xs):
        X0 = Xs[0]
        for Xi in Xs[1:]:
            if Xi != X0:
                return False
        return True

    def _get_split_point(self, data):
        best_var = np.inf
        best_feat_idx = best_value = 0
        for feat_idx in range(self.n):
            for value in set(data[:, feat_idx]):
                bool_list = data[:, feat_idx] < value
                var = self._sum_var(data[bool_list, -1])
                var += self._sum_var(data[~bool_list, -1])
                if best_var > var:
                    best_var = var
                    best_feat_idx = feat_idx
                    best_value = value
        return best_feat_idx, best_value

    def _sum_var(self, Xs):
        if len(Xs) < 2:
            return 0
        return np.var(Xs) * len(Xs)

    def predict(self, data_set):
        result = []
        for data in data_set:
            branch = self.tree
            while True:
                if isinstance(branch, dict):
                    lorr = ('left' if data[branch['feat']] < branch['val']
                            else 'right')
                    branch = branch[lorr]
                else:
                    result.append(branch)
                    break
        return np.array(result)


class GBDT(object):
    """
    """
    def __init__(self, max_depth=3, lr=0.1, size=100):
        super(GBDT, self).__init__()
        self.max_depth = max_depth
        self.lr = lr
        self.size = size
        self.trees = []

    def fit(self, data):
        data = data.copy()
        for tree_idx in range(self.size):
            self.trees.append(CartTree(max_depth=self.max_depth))
            self.trees[-1].fit(data)
            if tree_idx < self.size - 1:
                lr = self.lr # if tree_idx < self.size - 2 else 1
                result = self.trees[-1].predict(data)
                for i in range(data.shape[0]):
                    data[i, -1] = data[i, -1] - result[i] * lr

    def predict(self, data_set):
        results = []
        for tree in self.trees:
            results.append(tree.predict(data_set) * self.lr)
        return [sum(result) for result in zip(*results)]


if __name__ == '__main__':
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data = load_data('data/iris.csv', labelmap)
    train_data, test_data = train_test_split(data)

    gbdt = GBDT()
    gbdt.fit(train_data)
    print(list(zip(gbdt.predict(test_data), test_data[:, -1])))

#!/usr/bin/python3
# coding=utf-8

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file, labelmap=None):
    data = pd.read_csv(file, header=None)
    values = data.values
    data = values[:, :-1]
    label = values[:, -1]
    if labelmap:
        label = np.array([labelmap[x] for x in label])
    return data, label


def train_test_split(data, label, test_size=0.2):
    n = data.shape[0]
    test_num = round(test_size * n)
    train_index = list(range(n))
    test_index = []
    for i in range(test_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        test_index.append(random_index)
        del train_index[random_index]
    train = np.array([data[x] for x in train_index])
    train_label = np.array([label[x] for x in train_index])
    test = np.array([data[x] for x in test_index])
    test_label = np.array([label[x] for x in test_index])
    return train, test, train_label, test_label


class DicisionTree(object):

    def __init__(self, file=None):
        if file:
            self.load(file)
        else:
            self.tree = None
        self.data = None
        self.label = None
        self.n = None
        self.m = None
        self.avg = None

    def calcEnt(self, labels, weights, weight_sum):
        count = {}
        for i in range(len(labels)):
            if labels[i] in count:
                count[labels[i]] += weights[i]
            else:
                count[labels[i]] = weights[i]
        Ent = 0
        for weight in count.values():
            prob = weight / weight_sum
            Ent -= prob * np.log2(prob)
        return Ent

    def calcInfoGain(self, values, label, bound, Ent, weight, weight_sum):
        bool_less = []
        for value in values:
            bool_less.append(value < bound)
        bool_more = [not x for x in bool_less]

        less_label = label[bool_less]
        less_weight = weight[bool_less]
        more_label = label[bool_more]
        more_weight = weight[bool_more]

        less_weight_sum = sum(less_weight)
        more_weight_sum = weight_sum - less_weight_sum
        less_ratio = less_weight_sum / weight_sum
        more_ratio = 1 - less_ratio

        less_ent = self.calcEnt(less_label, less_weight, less_weight_sum)
        more_ent = self.calcEnt(more_label, more_weight, more_weight_sum)

        result = Ent - less_ent * less_ratio
        result -= more_ent * more_ratio
        return result, less_ratio

    def calcSplitInfo(self, data, weight, weight_sum):
        count = {}
        result = 0
        for i in range(len(data)):
            if data[i] in count:
                count[data[i]] += weight[i]
            else:
                count[data[i]] = weight[i]
        for key in count:
            p_key = count[key] / weight_sum
            result -= p_key * np.log2(p_key)
        return result

    def delMissing(self, data_set, label, weight):
        weight_sum = sum(weight)
        is_valid = []
        for data in data_set:
            is_valid.append(data != np.NaN)
        valid_data = data_set[is_valid]
        valid_label = label[is_valid]
        valid_weight = weight[is_valid]
        return (valid_data, valid_label, valid_weight,
                sum(weight[is_valid]), weight_sum)

    def chooseBound(self, data, label, weight):
        max_gain_ratio = 0
        best_feature = None
        best_bound = None
        best_less_ratio = None
        for feature in range(self.m):
            valid_info = self.delMissing(data[:, feature], label, weight)
            (valid_data, valid_label, valid_weight, valid_weight_sum,
                weight_sum) = valid_info
            Ent = self.calcEnt(valid_label, valid_weight, valid_weight_sum)
            values = sorted(valid_data)
            last_value = values[0]
            split_info = self.calcSplitInfo(values, valid_weight,
                                            valid_weight_sum)
            for value in values:
                if value == last_value:
                    continue
                bound = (value + last_value) / 2
                last_value = value
                valid_ratio = valid_weight_sum / weight_sum
                info_gain, less_ratio = self.calcInfoGain(values, valid_label,
                                              bound, Ent, valid_weight,
                                              valid_weight_sum)
                gain_ratio = valid_ratio * info_gain / split_info
                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_feature = feature
                    best_bound = bound
                    best_less_ratio = less_ratio
        return best_feature, best_bound, best_less_ratio

    def getMost(self, labels):
        count = {}
        for label in labels:
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
        most_label = 0
        max_num = 0
        for label, num in count.items():
            if num > max_num:
                max_num = num
                most_label = label
        return most_label

    def extendTree(self, data, label, weight):
        if len(set(label.tolist())) == 1:
            return label[0]
        feature, bound, less_ratio = self.chooseBound(data, label, weight)
        if feature is None:
            return self.getMost(label)

        less_data = []
        less_label = []
        less_weight = []
        more_data = []
        more_label = []
        more_weight = []
        values = data[:, feature]
        for i in range(len(values)):
            if values[i] == np.NaN:
                less_data.append(data[i])
                less_label.append(label[i])
                less_weight.append(less_ratio)
                more_data.append(data[i])
                more_label.append(label[i])
                more_weight.append(1 - less_ratio)
            elif values[i] < bound:
                less_data.append(data[i])
                less_label.append(label[i])
                less_weight.append(weight[i])
            else:
                more_data.append(data[i])
                more_label.append(label[i])
                more_weight.append(weight[i])

        less_data = np.array(less_data)
        less_label = np.array(less_label)
        less_weight = np.array(less_weight)
        more_data = np.array(more_data)
        more_label = np.array(more_label)
        more_weight = np.array(more_weight)

        tree = [feature, bound]
        tree.append(self.getMost(label))
        tree.append(self.extendTree(less_data, less_label, less_weight))
        tree.append(self.extendTree(more_data, more_label, more_weight))
        return tree

    def preprocess(self, data_set, missing_values=np.NaN, avg=None):
        if avg:
            self.avg = avg
        if self.avg:
            for i in range(len(dataset)):
                for j in range(len(data_set[0])):
                    if data_set[i][j] == missing_values:
                        data_set[i][j] = self.avg[j]
        else:
            avg = []
            is_missing = []
            for i in range(data_set[0]):
                col_sum = 0
                length = len(data_set)
                for j in range(length):
                    if data_set[i][j] == missing_values:
                        is_missing.append((j, i))
                        length -= 1
                    else:
                        col_sum += data_set[j][i]
                avg.append(col_sum / length)
            self.avg = avg
        for missing in is_missing:
            data_set[missing[0]][missing[1]] = self.avg[missing[1]]

    def train_prune_split(self, data, label, prune_size=0.2):
        n = data.shape[0]
        prune_num = round(prune_size * n)
        train_index = list(range(n))
        prune_index = []
        for i in range(prune_num):
            random_index = int(np.random.uniform(0, len(train_index)))
            prune_index.append(random_index)
            del train_index[random_index]
        train = np.array([data[x] for x in train_index])
        train_label = np.array([label[x] for x in train_index])
        prune = np.array([data[x] for x in prune_index])
        prune_label = np.array([label[x] for x in prune_index])
        return train, prune, train_label, prune_label

    def evaluate(self, data, label):
        h = self.predict(data)
        length = len(h)
        rightnum = 0
        for i in range(length):
            rightnum += (h[i] == label[i])
        return rightnum / length

    def prune(self, sub_tree, data, label, parent,
              parent_index, best_result, tree=None):
        if not tree:
            tree = self.tree
        if isinstance(sub_tree[3], list):
            # print('hh')
            best_result = max(best_result, self.prune(
                sub_tree[3], data, label, sub_tree, 3, best_result))
        if isinstance(sub_tree[4], list):
            best_result = max(best_result, self.prune(
                sub_tree[4], data, label, sub_tree, 4, best_result))
        parent[parent_index] = sub_tree[2]
        # self.createPlot()
        temp_result = self.evaluate(data, label)
        if temp_result <= best_result:
            parent[parent_index] = sub_tree
            return best_result
        else:
            return temp_result

    def train(self, data, label):
        self.data = data
        self.label = label
        self.n, self.m = self.data.shape
        split_result = self.train_prune_split(self.data, self.label)
        train_data, prune_data, train_label, prune_label = split_result
        weight = np.ones(len(train_data))
        self.tree = self.extendTree(train_data, train_label, weight)
        root_parent = [self.tree]
        result = self.evaluate(prune_data, prune_label)
        print(result)
        self.createPlot()
        result = self.prune(self.tree, prune_data,
                            prune_label, root_parent, 0, result)
        print(result)
        self.createPlot()

    def save(self, file):
        pickle.dump((self.tree, self.avg), open(file, 'wb'))

    def load(self, file):
        self.tree, self.avg = pickle.load(open(file, 'rb'))

    def predict(self, data_set):
        result = []
        for data in data_set:
            tree = self.tree
            while isinstance(tree, list):
                is_more = data[tree[0]] > tree[1]
                tree = tree[is_more + 3]
            result.append(tree)
        return np.array(result)

    def getLeafNum(self, tree=None):
        if not tree:
            tree = self.tree
        leafnum = 0
        if isinstance(tree[3], list):
            leafnum += self.getLeafNum(tree[3])
        else:
            leafnum += 1
        if isinstance(tree[4], list):
            leafnum += self.getLeafNum(tree[4])
        else:
            leafnum += 1
        return leafnum

    def getDepth(self, tree=None):
        if not tree:
            tree = self.tree
        max_depth = 1
        if isinstance(tree[3], list):
            max_depth = max(max_depth, self.getLeafNum(tree[3]))
        if isinstance(tree[4], list):
            max_depth = max(max_depth, self.getLeafNum(tree[4]))
        return max_depth + 1

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                          xytext=centerPt, textcoords='axes fraction',
                          va="center", ha="center", bbox=nodeType,
                          arrowprops=self.arrow_args)

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString, va="center",
                      ha="center", rotation=30)

    # if the first key tells you what feat was split on
    def plotTree(self, myTree, parentPt, nodeTxt):
        # this determines the x width of this tree
        numLeafs = self.getLeafNum(myTree)
        depth = self.getDepth(myTree)
        # the text label for this node should be this
        firstStr = str(myTree[:2])
        cntrPt = (self.xOff + (1.0 + float(numLeafs)) /
                  2.0/self.totalW, self.yOff)
        # self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, self.decisionNode)
        self.yOff = self.yOff - 1.0 / self.totalD
        for i in (3, 4):
            if isinstance(myTree[i], list):
                self.plotTree(myTree[i], cntrPt, str(i-3))
            else:
                self.xOff = self.xOff + 1.0 / self.totalW
                self.plotNode(
                    myTree[i], (self.xOff, self.yOff), cntrPt, self.leafNode)
                # self.plotMidText((self.xOff, self.yOff), cntrPt, str(i-3))
        self.yOff = self.yOff + 1.0/self.totalD

    def createPlot(self):
        self.decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
        # self.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
        self.totalW = float(self.getLeafNum(self.tree))
        self.totalD = float(self.getDepth(self.tree))
        self.xOff = -0.5/self.totalW
        self.yOff = 1.0
        self.plotTree(self.tree, (0.5, 1.0), '')
        plt.show()


if __name__ == '__main__':
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data, label = load_data('iris.csv', labelmap)
    train, test, train_label, test_label = train_test_split(data, label)
    dt = DicisionTree()
    dt.train(train, train_label)
    print(dt.predict(test))
    print(test_label)

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pickle
import math
import sys
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np


cfg = {
    11: (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    13: (64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M'),
    16: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
         512, 512, 512, 'M', 512, 512, 512, 'M'),
    19: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'),
}


class VGG(nn.Module):

    def __init__(self, features, out_channels):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, out_channels),
        )
        # self.classifier = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(in_channels, cfg, batch_norm=False):
    layers = []
    for i in cfg:
        if i == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(i))
            layers.append(nn.ReLU(inplace=True))
            in_channels = i
    return nn.Sequential(*layers)


class MnistEx(Dataset):

    def __init__(self, file, training=True):
        super(MnistEx, self).__init__()
        if training:
            data = pickle.load(open(file, 'rb'))[:4000]
        else:
            data = pickle.load(open(file, 'rb'))[5000:]
        transformer = transforms.ToTensor()
        self.img, self.label = zip(*[(transformer(x[0].reshape((80, 80))[:, :, np.newaxis]), x[1][0][0]*10 + x[1][1][0]) for x in data])

    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


def get_data(training=True):
    if training:
        dataset = MnistEx('data/16-Dataset/two_numbers_dataset_6k.pkl', True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    else:
        dataset = MnistEx('data/16-Dataset/two_numbers_dataset_6k.pkl', False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader


def train(vgg_num, net=None, start_epoch=0, epoch_num=2):
    trainloader = get_data(training=True)
    if not net:
        net = VGG(make_layers(1, cfg[vgg_num], True), 100)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.000001)
    print('====   Training..   ====')
    net.train()

    for epoch in range(start_epoch, start_epoch+epoch_num):
        print('----    epoch: %d    ----' % (epoch, ))
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
        print('loss: %.3f' % (running_loss, ))
    del trainloader
    print('Finished Training')
    return net


def test(net):
    testloader = get_data(training=False)
    correct = 0
    total = 0
    net.eval()
    for images, labels in testloader:
        outputs = net(Variable(images.cuda()))
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        result = (predicted == labels.cuda())
        correct += result.sum()
    del testloader
    print('Accuracy of the network on the test images: %f %%' %
          (100 * correct / total))


if __name__ == '__main__':
    vgg_num = 16
    start_epoch = 0
    net = None
    epoch_num = 10
    train_num_each_test = 2
    file_name = './model/vgg%d_net.pkl' % (vgg_num, )
    if os.path.exists(file_name):
        print('loading model...')
        (start_epoch, net) = pickle.load(open(file_name, 'rb'))
    for i in range(epoch_num // train_num_each_test):
        net = train(vgg_num, net, start_epoch, train_num_each_test)
        print('----  Saving.. ----')
        start_epoch += train_num_each_test
        print('---- Testing.. ----')
        test(net)
    pickle.dump((start_epoch, net), open(file_name, 'wb'), True)

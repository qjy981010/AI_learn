#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np
from collections import Iterable
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from warpctc_pytorch import CTCLoss
from PIL import Image


class FixHeightResize(object):

    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), 100)
        return img.resize((width, self.height), Image.ANTIALIAS)


class LabelTransformer(object):

    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def __call__(self, text, encode=True, length=None):
        if encode:
            if isinstance(text, str):
                length = [len(text)]
                result = [self.encode_map[letter] for letter in text]
            else:
                length = []
                result = []
                for word in text:
                    length.append(len(word))
                    result.extend([self.encode_map[letter] for letter in word])
            return torch.IntTensor(result), torch.IntTensor(length)
        else:
            if not length or len(length) == 1:
                result = []
                for i in range(len(text)):
                    if text[i] != 0 and (not(i > 0 and text[i] == text[i-1])):
                        result.append(decode_map[text[i]])
            else:
                result = []
                count = 0
                for i in range(len(length)):
                    word = []
                    for j in range(length[i]):
                        count += 1
                        if text[count] != 0 and (not(j > 0 and
                                                text[count] == text[count-1])):
                            word.append(decode_map[text[i]])
                    result.append(''.join(word))
            return ''.join(result)


class IIIT5k(Dataset):

    def __init__(self, root, train=True):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if train else 'testdata'
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in sio.loadmat(
            root+data_str+'.mat')[data_str][0]])
        transform = transforms.Compose([
            FixHeightResize(32),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def load_data():
    print('==== Loading data.. ====')
    trainset = IIIT5k('data/IIIT5K/', train=True)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True,
                             num_workers=4, collate_fn=my_collate)
    return trainloader


class DeepLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # ????

    def forward(self, x):
        x = self.lstm(x)[0]
        w, b, c = x.size()
        x = x.view(w*b, c)
        out = self.fc(x)
        out = out.view(w, b, -1)
        return out


class CRNN(nn.Module):

    def __init__(self, in_channels, out_channels, ):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 256
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        self.batchnorm = (False, False, False, True, False)
        self.cnn = self._get_cnn_layers()
        self.rnn = nn.Sequential(
            DeepLSTM(512, hidden_size, hidden_size),
            DeepLSTM(hidden_size, hidden_size, out_channels)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        return x

    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def train(start_epoch, epoch_num, letters, net=None, lr=0.1):
    trainloader = load_data()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU! start up!')
    if not net:
        net = CRNN(1, len(letters) + 1)
    criterion = CTCLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('<---- epoch: %d ---->' % (epoch, ))
        loss_sum = 0
        for i, (images, labels) in enumerate(trainloader):
            for img, label in zip(images, labels):
                label, label_length = labeltransformer(label, encode=True)
                img.unsqueeze_(0)
                if use_cuda:
                    img = img.cuda()
                    # label = label.cuda()
                    # label_length = label_length.cuda()
                img, label = Variable(img), Variable(label)
                label_length = Variable(label_length)
                optimizer.zero_grad()

                outputs = net(img)
                output_length = Variable(torch.IntTensor([outputs.size(0)]))
                loss = criterion(outputs, label, output_length, label_length)
                loss.backward()
                optimizer.step()
                loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return net


if __name__ == '__main__':
    file_name = 'rcnn.pkl'
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    net = None
    start_epoch = 0
    epoch_num = 10
    lr = 0.1
    if os.path.exists(file_name):
        start_epoch, net = pickle.load(open(file_name, 'rb'))
    net = train(start_epoch, epoch_num, letters, net=net, lr=lr)
    pickle.dump((start_epoch+epoch_num, net), open(file_name, 'wb'))

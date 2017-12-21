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


class LabelTransformer(object):

    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def encode(self, text):
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

    def decode(self, text, length=None):
        if length is None or len(length) == 1:
            result = []
            for i in range(len(text)):
                if text[i] != 0 and (not(i > 0 and text[i] == text[i-1])):
                    result.append(self.decode_map[text[i]])
            return ''.join(result)
        else:
            result = []
            count = 0
            for i in range(len(length)):
                word = []
                for j in range(length[i]):
                    count += 1
                    if text[count] != 0 and (not(j > 0 and
                                            text[count] == text[count-1])):
                        word.append(self.decode_map[text[i]])
                result.append(''.join(word))
            return result


class FixHeightResize(object):

    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), 100)
        return img.resize((width, self.height), Image.ANTIALIAS)


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


def load_data(train=True):
    print('==== Loading data.. ====')
    if train:
        if os.path.exists('data/IIIT5K/train.pkl'):
            dataset = pickle.load(open('data/IIIT5K/train.pkl', 'rb'))
        else:
            dataset = IIIT5k('data/IIIT5K/', train=True)
            pickle.dump(dataset, open('data/IIIT5K/train.pkl', 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                                 num_workers=4, collate_fn=my_collate)
    else:
        if os.path.exists('data/IIIT5K/test.pkl'):
            dataset = pickle.load(open('data/IIIT5K/test.pkl', 'rb'))
        else:
            dataset = IIIT5k('data/IIIT5K/', train=False)
            pickle.dump(dataset, open('data/IIIT5K/test.pkl', 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                                collate_fn=my_collate)
    return dataloader


class DeepLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # multiply 2 because it's bidirectional

    def forward(self, x):
        x = self.lstm(x)[0]         # length, batch, hidden_size*num_directions
        l, b, h = x.size()
        x = x.view(l*b, h)          # length*batch, hidden_size*num_directions
        out = self.fc(x)            # length*batch, output_size
        out = out.view(l, b, -1)    # length, batch, output_size
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

    def forward(self, x):           # input: height = 32
        x = self.cnn(x)             # batch, channel, height=1, width
        x = x.squeeze(2)            # batch, channel, width
        x = x.permute(2, 0, 1)      # width, batch, channel
        x = self.rnn(x)             # width/length, batch, channel
        return x                    # length, batch, out_channels

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
    trainloader = load_data(train=True)
    use_cuda = torch.cuda.is_available()
    if not net:
        net = CRNN(1, len(letters) + 1)
    criterion = CTCLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        print('GPU! start up!')
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
                label, label_length = labeltransformer.encode(label)
                img.unsqueeze_(0)
                if use_cuda:
                    img = img.cuda()
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


def test(net, letters):
    testloader = load_data(train=False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('GPU! start up!')
        net = net.cuda()
    labeltransformer = LabelTransformer(letters)

    print('====   Testing..   ====')
    net.eval()
    correct = 0
    for i, (images, labels) in enumerate(testloader):
        for img, origin_label in zip(images, labels):
            label, label_length = labeltransformer.encode(origin_label)
            img.unsqueeze_(0) # (channel, w, h) to (batch, channel, w, h)
            if use_cuda:
                img = img.cuda()
            img, label = Variable(img), Variable(label)
            label_length = Variable(label_length)

            outputs = net(img)                      # length, batch, num_letters
            length = [outputs.size(0)]
            outputs = outputs.max(2)[1].squeeze(1)  # length
            outputs = labeltransformer.decode(outputs.data, length=length)
            if outputs == origin_label:
                correct += 1
    print('test accuracy: ', correct / 30, '%')


def main(istrain=True):
    file_name = 'rcnn.pkl'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    if train:
        net = None
        start_epoch = 0
        epoch_num = 2
        lr = 0.1
        if os.path.exists(file_name):
            start_epoch, net = pickle.load(open(file_name, 'rb'))
        for i in range(10):
            net = train(start_epoch, epoch_num, letters, net=net, lr=lr)
            start_epoch += epoch_num
            test(net, letters)
        pickle.dump((start_epoch, net), open(file_name, 'wb'), True)
    else:
        start_epoch, net = pickle.load(open(file_name, 'rb'))
        test(net, letters)


if __name__ == '__main__':
    main(istrain=True)

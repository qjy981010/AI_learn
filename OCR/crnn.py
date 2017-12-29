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
from torchvision import transforms
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

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
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

    def __init__(self, root, training=True, fix_width=True):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in sio.loadmat(
            root+data_str+'.mat')[data_str][0]])
        transform = [transforms.Resize((32, 100), Image.ANTIALIAS)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


def load_data(root, training=True, fix_width=True):
    if training:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'train'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=True, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=4)
    else:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'test'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=False, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


class CRNN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CRNN, self).__init__()
        self.in_channels = in_channels
        hidden_size = 256
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        self.batchnorm = (False, False, False, True, False)
        self.cnn = self._get_cnn_layers()
        self.rnn1 = nn.LSTM(self.cnn_struct[-1][-1], hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, out_channels)
        self._initialize_weights()

    def forward(self, x):           # input: height=32, width>=100
        x = self.cnn(x)             # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)            # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)      # width>=24, batch, channel=512
        x = self.rnn1(x)[0]         # length=width>=24, batch, channel=256*2
        x = self.rnn2(x)[0]         # length=width>=24, batch, channel=256*2
        l, b, h = x.size()
        x = x.view(l*b, h)          # length*batch, hidden_size*2
        x = self.fc(x)              # length*batch, output_size
        x = x.view(l, b, -1)        # length>=24, batch, output_size
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


def train(root, start_epoch, epoch_num, letters, net=None, lr=0.1, fix_width=True):
    trainloader = load_data(root, training=True, fix_width=fix_width)
    use_cuda = torch.cuda.is_available()
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
        print('----    epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        for i, (img, label) in enumerate(trainloader):
            label, label_length = labeltransformer.encode(label)
            if use_cuda:
                img = img.cuda()
            img, label = Variable(img), Variable(label)
            label_length = Variable(label_length)
            optimizer.zero_grad()

            outputs = net(img)
            output_length = Variable(torch.IntTensor([outputs.size(0)]*outputs.size(1)))
            loss = criterion(outputs, label, output_length, label_length)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return net


def test(root, net, letters, fix_width=True):
    testloader = load_data(root, training=False, fix_width=fix_width)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')
    net.eval()
    correct = 0
    for i, (img, origin_label) in enumerate(testloader):
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        # length, batch, num_letters
        outputs = net(img)
        outputs = outputs.max(2)[1].transpose(0, 1)  # batch, length
        outputs = labeltransformer.decode(outputs.data)
        correct += sum([out == real for out, real in zip(outputs, origin_label)])
    print('test accuracy: ', correct / 30, '%')


def main(training=True, fix_width=True):
    file_name = ('fix_width_' if fix_width else '') + 'crnn_.pkl'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    root = 'data/IIIT5K/'
    if training:
        net = None
        start_epoch = 0
        epoch_num = 2
        lr = 0.05
        if os.path.exists(file_name):
            print('Pre-trained model detected.\nLoading model...')
            start_epoch, net = pickle.load(open(file_name, 'rb'))
        if torch.cuda.is_available():
            print('GPU detected.')
        for i in range(5):
            net = train(root, start_epoch, epoch_num, letters, net=net, lr=lr, fix_width=fix_width)
            start_epoch += epoch_num
            test(root, net, letters, fix_width=fix_width)
        pickle.dump((start_epoch, net), open(file_name, 'wb'), True)
    else:
        start_epoch, net = pickle.load(open(file_name, 'rb'))
        test(root, net, letters, fix_width=fix_width)


if __name__ == '__main__':
    main(training=True, fix_width=True)

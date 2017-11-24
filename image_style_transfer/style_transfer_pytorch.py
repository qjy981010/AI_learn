#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import pickle
import torch
import torchvision

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models

from load_image import *

CONTENT_IMG_PATH = 'image/resized_content1.jpg'
STYLE_IMG_PATH = 'image/resized_style3.jpg'


class FakeVGG(nn.Module):

    def __init__(self):
        super(FakeVGG, self).__init__()
        original_model = models.vgg19_bn(pretrained=True)
        # for layer in enumerate(original_model.features.children()):
        #     print(layer)
        self.features1 = nn.Sequential(
            *list(original_model.features.children())[:6])
        self.features2 = nn.Sequential(
            *list(original_model.features.children())[6:13])
        self.features3 = nn.Sequential(
            *list(original_model.features.children())[13:26])
        self.features4 = nn.Sequential(
            *list(original_model.features.children())[26:39])
        self.features5 = nn.Sequential(
            *list(original_model.features.children())[39:52])
        self.pool = list(original_model.features.children())[-1]

    def forward(self, x):
        results = []
        results.append(self.features1(x))
        results.append(self.features2(results[-1]))
        results.append(self.features3(results[-1]))
        results.append(self.features4(results[-1]))
        results.append(self.features5(results[-1]))
        return results


def gram_matrix(x):
    n, c, h, w = x.size()
    view = x.view(n*c, h*w)
    G = torch.mm(view, view.t())
    return G


def show_image(img):
    unloader = transforms.ToPILImage()
    image = img.clone().cpu()
    image = image.view(3, 500, -1)
    image = unloader(image.data)
    plt.imshow(image)
    plt.show()


class TotalLoss(nn.Module):
    def __init__(self, As, P, alpha, beta):
        super(TotalLoss, self).__init__()
        self.As = As
        self.NxMs = [A.size()[0] * A.size()[1] for A in As]
        self.P = P
        self.alpha = alpha
        self.beta = beta

    def forward(self, Gs, F):
        content_loss = torch.sum((F - self.P)**2).div(2)
        Gs = list(map(gram_matrix, Gs))
        style_loss = Variable(torch.Tensor([0])).cuda()
        for A, G, NxM in zip(self.As, Gs, self.NxMs):
            style_loss += torch.sum((A - G)**2).div(4 * NxM**2)
        total_loss = self.alpha * content_loss + self.beta * style_loss
        return total_loss


def train(alpha, beta, epoch_num=100):
    model = FakeVGG()
    model.cuda()
    content_img = Variable(torch.Tensor(load_image(CONTENT_IMG_PATH)).cuda())
    style_img = Variable(torch.Tensor(load_image(STYLE_IMG_PATH)).cuda())
    # print(content_img.size(), style_img.size())
    P = model.forward(content_img)[3]
    As = list(map(gram_matrix, model.forward(style_img)))
    input_img = Variable(torch.zeros(content_img.size()).cuda(), requires_grad=True)
    input_img = content_img
    input_img = nn.Parameter(input_img.data)
    criterion = TotalLoss(As, P, alpha, beta).cuda()
    optimizer = optim.LBFGS([input_img])
    retain_graph = True

    def closure():
        optimizer.zero_grad()
        result = model.forward(input_img)
        loss = criterion(result, result[3])
        print('loss: ', float(loss.data[0]))
        loss.backward(retain_graph=True)
        return loss

    for epoch in range(epoch_num):
        print('---- epoch: %d ----' % epoch)
        optimizer.step(closure)
    show_image(input_img)


if __name__ == '__main__':
    train(1, 1000)

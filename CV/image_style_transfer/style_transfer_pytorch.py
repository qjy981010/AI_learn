#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import pickle
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models

CONTENT_IMG_PATH = 'image/resized_content1.jpg'
STYLE_IMG_PATH = 'image/resized_stylehh.jpg'


def image_loader(image_name):
    loader = transforms.ToTensor()
    image = Image.open(image_name)
    image = Variable(loader(image).unsqueeze(0).cuda(), requires_grad=True)
    return image

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])  # 定义转换，将图片归一化。
    # 加载两张图片
    content_image = Image.open(CONTENT_IMG_PATH)
    style_image = Image.open(STYLE_IMG_PATH)
    # 因为图片是 3*32*32的，但是pytorch接受的输入是4维的，所以要添加一个1 的维度，相当于变成了1*3*32*32 
    content_image = transform(content_image).unsqueeze(0)
    style_image = transform(style_image).unsqueeze(0)
    # 类型转换
    content_image = content_image.type(torch.FloatTensor).cuda()
    style_image = style_image.type(torch.FloatTensor).cuda()
    return content_image,style_image


class FakeVGG(nn.Module):

    def __init__(self):
        super(FakeVGG, self).__init__()
        original_model = models.vgg19(pretrained=True)
        # for layer in enumerate(original_model.features.children()):
        #     print(layer)
        self.features1 = nn.Sequential(
            *list(original_model.features.children())[:4])
        self.features2 = nn.Sequential(
            *list(original_model.features.children())[5:9])
        self.features3 = nn.Sequential(
            *list(original_model.features.children())[10:18])
        self.features4 = nn.Sequential(
            *list(original_model.features.children())[19:27])
        self.features5 = nn.Sequential(
            *list(original_model.features.children())[28:36])
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        results = []
        results.append(self.features1(x))
        results.append(self.features2(self.pool(results[-1])))
        results.append(self.features3(self.pool(results[-1])))
        results.append(self.features4(self.pool(results[-1])))
        results.append(self.features5(self.pool(results[-1])))
        return results


def gram_matrix(x):
    n, c, h, w = x.size()
    view = x.view(n*c, h*w)
    G = torch.mm(view, view.t())
    return G


def show_image(img):
    unloader = transforms.ToPILImage()
    image = img.data.clone().cpu()
    image = image.view(3, 400, 600)
    image = unloader(image)
    image.save('result.jpg', 'JPEG')
    # plt.imshow(image)
    # plt.show()


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
        style_loss = 0
        for A, G, NxM in zip(self.As, Gs, self.NxMs):
            style_loss += torch.sum((A - G)**2).div(4 * NxM**2)
        total_loss = self.alpha * content_loss + self.beta * style_loss
        print('content_loss: ', content_loss.data[0], ' style loss: ', style_loss.data[0], ' total loss: ', total_loss.data[0])
        return total_loss


def train(alpha, beta, epoch_num=100):
    model = FakeVGG()
    model.cuda()
    content_img = image_loader(CONTENT_IMG_PATH)
    style_img = image_loader(STYLE_IMG_PATH)
    # content_img, style_img = load_data()
    P = model(content_img)[3]
    As = list(map(gram_matrix, model(style_img)))
    input_img = Variable(image_loader('result.jpg').data, requires_grad=True)
    # input_img = Variable(content_img.data.clone(), requires_grad=True)
    # input_img = nn.Parameter(input_img.data)
    criterion = TotalLoss(As, P, alpha, beta).cuda()
    optimizer = optim.Adam([input_img], lr=0.05)
    # optimizer = optim.LBFGS([input_img])

    for i in range(5):
        for epoch in range(epoch_num):
            print('---- epoch: %d ----' % epoch)
            optimizer.zero_grad()
            result = model(input_img)
            loss = criterion(result, result[3])
            loss.backward(retain_graph=True)
            optimizer.step()
        show_image(input_img)
        # show_image(content_img)


if __name__ == '__main__':
    train(1, 1000, 200)

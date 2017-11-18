#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.autograd import Variable

def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool  = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1   = torch.nn.Linear(16*5*5, 120)
        self.fc2   = torch.nn.Linear(120, 84)
        self.fc3   = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        print(x.size())
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        print(x.size())
        x = x.view(-1, 16*5*5)
        print(x.size())
        x = torch.nn.functional.relu(self.fc1(x))
        print(x.size())
        x = torch.nn.functional.relu(self.fc2(x))
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        print()
        return x


def train(trainloader):
    net = Net().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                            (epoch+1, i+1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')
    return net


def test(net, testloader, classes):
    correct = 0
    total = 0
    class_correct = [0.0] * 10
    class_total = [0.0] * 10
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        result = (predicted == labels.cuda())
        correct += result.sum()
        c = result.squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % 
                (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
                    (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    trainloader, testloader, classes = get_data()
    net = train(trainloader)
    pickle.dump(net, open('./net/alexnet.pkl', 'wb'))
    net = pickle.load(open('./net/alexnet.pkl', 'rb'))
    test(net, testloader, classes)
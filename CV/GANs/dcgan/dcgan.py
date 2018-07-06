import torch.nn as nn
import torch.nn.init as init


class Generator(nn.Module):
    """
    """

    def __init__(self, z_channels, c_channels):
        super(Generator, self).__init__()
        # 1 x 1
        self.cnn1 = nn.Sequential(
            nn.ConvTranspose2d(z_channels, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.cnn2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.cnn3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.cnn4 = nn.Sequential(
            nn.ConvTranspose2d(128, c_channels, 4, 2, 1),
            nn.Tanh())
        # 28 x 28
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Discriminator(nn.Module):
    """
    """

    def __init__(self, c_channels):
        super(Discriminator, self).__init__()
        # 28 x 28
        self.cnn1 = nn.Sequential(
            nn.Conv2d(c_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid())
        # 1 x 1
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x).squeeze()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
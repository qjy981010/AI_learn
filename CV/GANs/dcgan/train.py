import torch
import os
import time
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import load_mnist
from dcgan import Discriminator, Generator
from torchvision.utils import save_image


def train(z_channels, c_channels, epoch_num, batch_size, lr=0.0002, beta1=0.5,
          model_path='models/dcgan_checkpoint.pth'):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        cudnn.benchmark = True
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    loader = load_mnist(batch_size)

    generator = Generator(z_channels, c_channels).to(device)
    discriminator = Discriminator(c_channels).to(device)
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    start_epoch = 0
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        generator.load_state_dict(checkpoint['g'])
        discriminator.load_state_dict(checkpoint['d'])
        g_optimizer.load_state_dict(checkpoint['g_optim'])
        d_optimizer.load_state_dict(checkpoint['d_optim'])
        start_epoch = checkpoint['epoch'] + 1
    criterion = nn.BCELoss().to(device)

    generator.train()
    discriminator.train()
    std = 0.1
    for epoch in range(start_epoch, start_epoch+epoch_num):
        d_loss_sum, g_loss_sum = 0, 0
        print('----    epoch: %d    ----' % (epoch, ))
        for i, (real_image, number) in enumerate(loader):
            real_image = real_image.to(device)
            image_noise = torch.randn(real_image.size(), device=device).normal_(0, std)

            d_optimizer.zero_grad()
            real_label = torch.randn(number.size(), device=device).normal_(0.9, 0.1)
            real_image.add_(image_noise)
            out = discriminator(real_image)
            d_real_loss = criterion(out, real_label)
            d_real_loss.backward()

            noise_z = torch.randn((number.size(0), z_channels, 1, 1),
                                  device=device)
            fake_image = generator(noise_z)
            fake_label = torch.zeros(number.size(), device=device)
            fake_image = fake_image.add(image_noise)
            out = discriminator(fake_image.detach())
            d_fake_loss = criterion(out, fake_label)
            d_fake_loss.backward()

            d_optimizer.step()

            g_optimizer.zero_grad()
            out = discriminator(fake_image)
            g_loss =criterion(out, real_label)
            g_loss.backward()
            g_optimizer.step()

            d_loss_sum += d_real_loss.item() + d_fake_loss.item()
            g_loss_sum += g_loss.item()
            # if i % 10 == 0:
            #     print(d_loss, g_loss)
        print('d_loss: %f \t\t g_loss: %f' % (d_loss_sum/(i+1),
                                              g_loss_sum/(i+1)))
        std *= 0.9
        if epoch % 1 == 0:
            checkpoint = {
                'g':generator.state_dict(),
                'd':discriminator.state_dict(),
                'g_optim':g_optimizer.state_dict(),
                'd_optim':d_optimizer.state_dict(),
                'epoch':epoch,
            }
            save_image(fake_image, 'out/fake_samples_epoch_%03d.png' % (epoch,),
                       normalize=False)
            torch.save(checkpoint, model_path)
            os.system('cp ' + model_path + ' models/model%d' % (epoch,))
            print('saved!')


if __name__ == '__main__':
    train(100, 1, 50, 128)

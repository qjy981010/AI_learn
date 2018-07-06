import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def load_mnist(batch_size=128, train=True, workers=4):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST('../../datasets/', train=train,
                    transform=trans, download=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=workers,
                                         batch_size=batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    loader = load_mnist(1, True, 0)
    for img, label in loader:
        print(img)
        exit(0)
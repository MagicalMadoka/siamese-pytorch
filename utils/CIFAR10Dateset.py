import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import random


class CIFAR10SiameseDataset(Dataset):
    def __init__(self, train=True):
        self.cifar10 = datasets.CIFAR10(root='./data', train=train, download=True, transform=transforms.ToTensor())
        self.train = train

    def __getitem__(self, index):
        img1, label1 = self.cifar10[index]

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                index2 = random.choice(range(len(self.cifar10)))
                img2, label2 = self.cifar10[index2]
                if label1 == label2:
                    break
        else:
            while True:
                index2 = random.choice(range(len(self.cifar10)))
                img2, label2 = self.cifar10[index2]
                if label1 != label2:
                    break

        tag = 1 if label1 == label2 else 0
        return img1, img2, torch.tensor(tag).float()

    def __len__(self):
        return len(self.cifar10)

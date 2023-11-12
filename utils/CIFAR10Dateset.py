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

        # 为了创建成对的图像，我们选取另一张同类别或不同类别的图像
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # 同类别图像
                index2 = random.choice(range(len(self.cifar10)))
                img2, label2 = self.cifar10[index2]
                if label1 == label2:
                    break
        else:
            while True:
                # 不同类别图像
                index2 = random.choice(range(len(self.cifar10)))
                img2, label2 = self.cifar10[index2]
                if label1 != label2:
                    break

        tag = 1 if label1 == label2 else 0
        return img1, img2, torch.tensor(tag).float()

    def __len__(self):
        return len(self.cifar10)

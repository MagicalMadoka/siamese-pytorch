import torch
from torch import nn
from torchvision import models


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

        vgg_original = models.vgg16(pretrained=True)

        self.features = vgg_original.features

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    def forward(self, left, right):
        left_output = self.forward_one(left)
        right_output = self.forward_one(right)

        # L1距离
        l1_distance = torch.abs(left_output - right_output)

        output = self.fc1(l1_distance)
        output = self.relu(output)

        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

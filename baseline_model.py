import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()

        # Feature Extractor
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2)  # input channels=1, output channels=128, kernel_size=5
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1)  # input channels=128, output channels=128, kernel_size=2
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1)  # input channels=128, output channels=256, kernel_size=2
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)  # input channels=256, output channels=256, kernel_size=2
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        
        self.bn5 = nn.BatchNorm1d(256*8*8)
        self.fc1 = nn.Linear(256 * 8 * 8, 1000)  # input size = 256 * 7 * 7, output size = 1000
        
        self.bn6 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 10)  # input size = 1000, output size = 10

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.pool2(x)

        x = self.flatten(x)
        
        x = self.bn5(x)
        x = self.fc1(x)

        x = self.bn6(x)
        x = self.fc2(x)
        
        x = self.relu(x)
        x = self.softmax(x)
        
        return x

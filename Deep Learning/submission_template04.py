import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Первый сверточный слой: 3 входных канала (RGB), 3 фильтра (5x5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Второй сверточный слой: 3 входных канала после conv1, 5 фильтров (3x3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))  
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Полносвязные слои
        self.fc1 = nn.Linear(180, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # [batch, 3, 14, 14]
        x = self.pool2(F.relu(self.conv2(x)))  # [batch, 5, 6, 6]
        x = x.view(x.size(0), -1)  # Разворачиваем [batch, 180]
        x = F.relu(self.fc1(x))  # [batch, 100]
        x = self.fc2(x)  # [batch, 10]
        return x

import torch
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes):
        super(YourCustomModel, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 128 * 128, num_classes)

    def forward(self, x):
        # Implement the forward pass of your model
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

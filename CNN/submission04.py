from torch import nn
from torch.nn import functional as F


class create_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5)) 
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2)) 
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3,3)) 
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2)) 
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6*6*5, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
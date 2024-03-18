import torch
import torch.nn as nn
class predictor(nn.Module):
    """
    This class uses survival function to predict burst and spike times for a cascade
    """

    def __init__(self, input_length, output_dimention = 20):
        """
        observation_length:
        """
        super(predictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
        self.pooling_max = nn.MaxPool1d(kernel_size=5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((((input_length // 5) // 5) // 5) * 64, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, output_dimention)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling_max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling_max(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling_max(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x




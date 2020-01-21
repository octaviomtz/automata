import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork2(nn.Module):
    '''WARNING: the adaptiveMaxPool was added but its performance was not evaluated'''
    def __init__(self, action_dim, max_action, seed):
        super(QNetwork2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=action_dim, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(action_dim)
        self.adaptiveMaxPool = nn.AdaptiveMaxPool3d(1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_action * torch.tanh(self.adaptiveMaxPool(x))
        x = torch.squeeze(x)
        # print(x.shape)
        return x

class Actor(nn.Module):
    '''WARNING: the adaptiveMaxPool was added but its performance was not evaluated'''
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=action_dim, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(action_dim)
        self.adaptiveMaxPool = nn.AdaptiveMaxPool3d(1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_action * torch.tanh(self.adaptiveMaxPool(x))
        x = torch.squeeze(x)
        # print(x.shape)
        return x

class Critic(nn.Module):
    '''WARNING: the adaptiveMaxPool was added but its performance was not evaluated'''
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=action_dim, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(action_dim)
        self.adaptiveMaxPool = nn.AdaptiveMaxPool3d(1)
        self.max_action = max_action
        ###
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1a = nn.BatchNorm3d(16)
        self.conv2a = nn.Conv3d(in_channels=16, out_channels=action_dim, kernel_size=3)
        self.bn2a = nn.BatchNorm3d(action_dim)
        self.adaptiveMaxPoola = nn.AdaptiveMaxPool3d(1)
        
    def forward(self, x):
        xu = torch.cat([x,u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.bn1(self.conv1(xu)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = torch.tanh(self.adaptiveMaxPool(x1))
        x1 = torch.squeeze(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.bn1a(self.conv1a(xu)))
        x2 = F.relu(self.bn2a(self.conv2a(x2)))
        x2 = torch.tanh(self.adaptiveMaxPoola(x2))
        x2 = torch.squeeze(x2)
        return x1, x2
    
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

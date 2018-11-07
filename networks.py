import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        super(Actor, self).__init__() 
        self.seed = torch.manual_seed(seed)
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Batch normalization operations between layers
        self.batch_norm1 = nn.BatchNorm1d(state_size)
        self.batch_norm2 = nn.BatchNorm1d(fc1_units)
        self.batch_norm3 = nn.BatchNorm1d(fc2_units)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Layer weight and bias initialization
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(.1)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(.1)
        
    def forward(self, state):
        """ Build an actor (policy) network that maps states -> actions. """
        x = self.batch_norm1(state)
        x = self.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):      
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Batch normalization operation
        self.batch_norm = nn.BatchNorm1d(fc1_units)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Layer weight and bias initialization
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(.1)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(.1)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.relu(self.fc1(state))
        x = self.batch_norm(x)
        x = torch.cat([x, action], dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

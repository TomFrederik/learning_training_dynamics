import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    
    def forward(self, t, x):
        x = torch.cat([torch.tensor([t], device=self.device), x], dim=-1)
        return self.net(x)
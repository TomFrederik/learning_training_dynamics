import torchdiffeq as teq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

import ode_models
import datasets
import utils

# some hyperparams
hidden_dim = 100
train_steps = 1000
lr = 1e-3

# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# set up data
data_dir = './data/mini_mnist'
train_dataset = datasets.MiniMNISTParams(data_dir)
test_dataset = datasets.MiniMNISTParams(data_dir, train=False)

train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

# 
input_dim = train_dataset[0][1].shape[1]
time_stamps = torch.arange(0, train_dataset[0][1].shape[0], 1, dtype=float, device=device)


# set up model
model_kwargs = {'input_dim': input_dim+1, # +1 for time  
                'hidden_dim': hidden_dim,
                'output_dim':input_dim
                }
model = ode_models.MLP(**model_kwargs).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fct = nn.MSELoss()

# training
step = 0
while step < train_steps:

    model.train()
    for y_0, y in iter(train_loader):
        step += 1
        
        y_0 = y_0.squeeze().to(device)
        y = y.squeeze().to(device)
        
        # train step
        optimizer.zero_grad()
        pred = teq.odeint_adjoint(model, y_0, time_stamps, adjoint_options=dict(norm=utils.make_norm(y_0)))
        loss = loss_fct(pred, y.squeeze())
        print(f'Loss in step {step} is {loss.item():1.3f}')
        loss.backward()
        optimizer.step()

    # eval step
    #model.eval()
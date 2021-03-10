import torch
from torch.utils.data import DataLoader

import numpy as np
import os

import datasets
import base_models
import utils


def train_model(model_class, model_kwargs, opt_class, opt_kwargs, loss_fct, train_loader, test_loader, train_steps):

    # init model
    model = model_class(**model_kwargs)

    # init param trajectory
    params = []
    params.append(utils.get_params(model))
    
    # set up optimizer
    optimizer = opt_class(model.parameters(), **opt_kwargs)

    # initial random logits
    #train_logits = []
    #data, _ = next(iter(train_loader))
    #train_logits.append(model(data).squeeze().tolist())

    for step in range(train_steps):
        
        # one training step
        model.train()
        data, labels = next(iter(train_loader))
        logits = model(data).squeeze()
        loss = loss_fct(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store new params
        params.append(utils.get_params(model))

        # store output of network at this step
        #train_logits.append(logits.tolist())

        # eval on test set
        model.eval()
        test_data, test_labels = next(iter(test_loader))
        test_logits = model(test_data).squeeze()
        test_loss = loss_fct(test_logits, test_labels)
        test_acc = torch.sum(torch.round(torch.sigmoid(test_logits)) == test_labels) / len(test_labels)

        print(f'Step {step+1}: train loss = {loss.item():1.5f}, test loss = {test_loss.item():1.5f}, test acc = {test_acc.item():1.4f}')                

    return params


# hparams
num_trajectories = 100
lr = 1e-4
train_steps = 25

# set up data processing
data_dir = './data/mini_mnist'

train_dataset = datasets.MiniMNIST(data_dir, flatten=True)
test_dataset = datasets.MiniMNIST(data_dir, flatten=True, train=False)

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=len(train_dataset)) # full batch GD
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))

# set up model
model_class = base_models.MLP
model_kwargs = {'input_dim':28*28, 
                'hidden_dim':[20,20],
                'output_dim':1
                }

# optimizer
opt_class = torch.optim.Adam
opt_kwargs = {'lr':lr}

# loss function
loss_fct = torch.nn.BCEWithLogitsLoss()

train_kwargs = {'model_class':model_class,
                'model_kwargs':model_kwargs,
                'opt_class':opt_class,
                'opt_kwargs':opt_kwargs, 
                'loss_fct':loss_fct, 
                'train_loader':train_loader, 
                'test_loader':test_loader, 
                'train_steps':train_steps
                }


params = []
for i in range(num_trajectories):
    print(f'\n\nTraining model number {i+1}:\n')
    params.append(train_model(**train_kwargs))

# save
#print(params)
params = torch.tensor(params)
train_params, test_params = torch.split(params, [80,20])
torch.save(train_params, './data/mini_mnist/train_params.pt')
torch.save(test_params, './data/mini_mnist/test_params.pt')
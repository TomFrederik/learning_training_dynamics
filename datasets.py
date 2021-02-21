import torch
from torch.utils.data import Dataset
import os

class MiniMNIST(Dataset):
    
    def __init__(self, data_dir, train=True, flatten=False):

        super().__init__()

        # set prefix for train or test
        prefix = 'train' if train else 'test'
        
        # load data
        self.data = torch.load(os.path.join(data_dir, f'{prefix}_data.pt')).float()
        self.labels = torch.load(os.path.join(data_dir, f'{prefix}_labels.pt')).float()

        # flatten
        if flatten:
            self.data = torch.flatten(self.data, start_dim=1)

    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        return image, label
    
    def __len__(self):
        return len(self.labels)

class MiniMNISTLogits(Dataset):
    '''
    Dataset of the logits of models trained on MiniMNIST
    '''
    def __init__(self, data_dir, train=True, flatten=False):

        super().__init__()

        # set prefix for train or test
        prefix = 'train' if train else 'test'
        
        # load data
        self.data = torch.load(os.path.join(data_dir, f'{prefix}_training_logits.pt')).float()

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class MiniMNISTParams(Dataset):
    '''
    Dataset of the parameters of models trained on MiniMNIST
    '''
    def __init__(self, data_dir, train=True, flatten=False):

        super().__init__()

        # set prefix for train or test
        prefix = 'train' if train else 'test'
        
        # load data
        self.data = torch.load(os.path.join(data_dir, f'{prefix}_params.pt')).float()

    def __getitem__(self, index):
        item = self.data[index]
        y_0, y = item[0], item
        return y_0, y
    
    def __len__(self):
        return len(self.data)
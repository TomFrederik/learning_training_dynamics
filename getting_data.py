import torch
import torchvision as tv

# get training set
DOWNLOAD = False
mnist = tv.datasets.MNIST(root='./data/', download=DOWNLOAD)

# get targets and data for zero and 1s
train_labels = mnist.train_labels
train_data = mnist.train_data[(train_labels == 0) | (train_labels == 1)]
train_labels = train_labels[(train_labels == 0) | (train_labels == 1)]

# sort them and take 50 of each class
sort_idcs = torch.argsort(train_labels)
train_labels = train_labels[sort_idcs]
train_labels = torch.cat([train_labels[:50], train_labels[-50:]])
train_data = train_data[sort_idcs]
train_data = torch.cat([train_data[:50], train_data[-50:]])


# get targets and data for zero and 1s
test_labels = mnist.test_labels
test_data = mnist.test_data[(test_labels == 0) | (test_labels == 1)]
test_labels = test_labels[(test_labels == 0) | (test_labels == 1)]

# sort them and take 20 of each class
sort_idcs = torch.argsort(test_labels)
test_labels = test_labels[sort_idcs]
test_labels = torch.cat([test_labels[:20], test_labels[-20:]])
test_data = test_data[sort_idcs]
test_data = torch.cat([test_data[:20], test_data[-20:]])


# save the small dataset
torch.save(train_data, './data/mini_mnist/train_data.pt')
torch.save(train_labels, './data/mini_mnist/train_labels.pt')

torch.save(test_data, './data/mini_mnist/test_data.pt')
torch.save(test_labels, './data/mini_mnist/test_labels.pt')

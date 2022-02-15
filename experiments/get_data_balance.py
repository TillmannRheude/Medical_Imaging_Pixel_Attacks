
import os
import torch
from medmnist import INFO
from models import create_resnet
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle

import torch.utils.data as data

import torchvision.transforms as transforms

from utils import load_mnist, get_mnist_dataset

from medmnist import INFO
dataflags = ['octmnist', 'tissuemnist', 'retinamnist', 'breastmnist']

attack_transform = transforms.Compose([
transforms.Resize(64),
transforms.ToTensor(),
transforms.Normalize(mean=[0.5], std=[0.5])])
BATCH_SIZE = 1
num_workers = 1
for data_flag in dataflags:

    test_dataset = get_mnist_dataset(data_flag, test=False, download=True, data_transform=attack_transform,
                                 data_aug=False)

    info = INFO[data_flag]
    n_classes = len(info['label'])
    counters = np.array(np.zeros((n_classes,)))

    attack_loader = data.DataLoader(dataset=test_dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=num_workers)
    for batch, label in attack_loader:
        counters[label.item()] += 1
    print()





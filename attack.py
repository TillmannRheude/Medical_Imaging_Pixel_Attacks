import os
from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import random 

from medmnist import INFO
from models import create_resnet
import torchvision.transforms as transforms

from evaluate import evaluate
from utils import load_mnist, complement, select_random_pixels

"""For GUI"""
def complementary_attack(image, is_rgb, y, x):
    image[y, x] = complement(image[y, x]) if is_rgb else 1 - image[y, x]

def zero_one_attack(image, is_rgb, y, x, probability_1 = 0.5):
    rnd_number = np.random.uniform(0, 1)
    if is_rgb:
        image[y, x] = (1, 1, 1) if rnd_number <= probability_1 else (0, 0, 0)
    else:
        image[y, x] = 1 if rnd_number <= probability_1 else 0
    return image

def additive_noise_attack(image, is_rgb, y, x, mean=0, std=1):
    noise = np.random.normal(mean, std)
    if is_rgb:
        # TODO check if RGB is normalized
        image[y, x] = np.clip(image[y, x] + noise, 0, 1)
    else:
        image[y, x] = np.clip(image[y, x] + noise, 0, 1)
    return image

def attack_single_image(image, attack, k=1, seed=None):
    # 28,28 / 28,28,3

    image = image.copy()
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1


    if attack == 'complementary':
        foo = complementary_attack
    elif attack == 'zero_one':
        foo = zero_one_attack
    elif attack == 'additive_noise':
        foo = additive_noise_attack
    else:
        exit(1, 'illegal function')

    indeces = select_random_pixels(height, width, seed)

    is_rgb = channels == 3
    for y, x in indeces[:k]:
        foo(image, is_rgb, y, x)

    return image

"""For Evaluation"""
def complementary_tensor_attack(image, is_rgb, y, x):
    comp = complement(image[:, y, x]) if is_rgb else 1 - image[:, y, x]
    image[0, y, x] = comp[0]
    image[1, y, x] = comp[1]
    image[2, y, x] = comp[2]

def zero_one_tensor_attack(image, is_rgb, y, x, probability_1 = 0.5):
    rnd_number = np.random.uniform(0, 1)
    if is_rgb:
        image[:, y, x] = torch.tensor([[[1], [1], [1]]]) if rnd_number <= probability_1 else torch.tensor([[[0], [0], [0]]])
    else:
        image[:, y, x] = 1 if rnd_number <= probability_1 else 0

    return image

def additive_noise_tensor_attack(image, is_rgb, y, x, mean=0, std=1):
    noise = np.random.normal(mean, std)
    if is_rgb:
        # TODO check if RGB is normalized
        image[:, y, x] = np.clip(image[:, y, x] + noise, 0, 1)
    else:
        image[:, y, x] = np.clip(image[:, y, x] + noise, 0, 1)
    return image

def attack_tensor_image(image, attack='complementary', k=1):
    # edits image in place!
    channels, height, width = image.shape
    indeces = select_random_pixels(height, width)
    if attack == 'complementary':
        foo = complementary_tensor_attack
    elif attack == 'zero_one':
        foo = zero_one_tensor_attack
    elif attack == 'additive_noise':
        foo = additive_noise_tensor_attack
    else:
        exit(1, 'illegal function')

    is_rgb = channels == 3
    print("change", k)
    for y, x in indeces[:k]:
        foo(image, is_rgb, y, x)

    plt.imshow(image[0].numpy())
    plt.show()
    return image

"""
def explicit_pixel_attack_tensor(input_image, attack="complementery", pixel_list=[[30, 30], [32, 32]]):

    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    image = image.copy()
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    is_rgb = channels == 3
    if attack == "complementary":
        for idx in pixel_list:
            input_image[0][idx] = 1 - input_image[0][idx]
    elif attack == "zero_one":
        for idx in pixel_list:
            input_image = zero_one_tensor(input_image, is_rgb, idx[0], idx[1])
    elif attack == "additive_noise":
        for idx in pixel_list:
            input_image = additive_noise_tensor(input_image, is_rgb, idx[0], idx[1])
    else:
        exit(1, 'illegal function')

    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()
        foo = additive_noise
    else:
        exit(1, 'illegal function')

    print(input_image)
    return input_image
"""


if __name__ == "__main__":
    MONTE_CARLO = True
    DATA_AUGMENTATION = True

    ## load model
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    download = True
    num_workers = 2
    NUM_EPOCHS = 1
    BATCH_SIZE = 64
    lr = 0.001

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)


    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    attack_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Lambda(attack_tensor_image),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/resnet18_' + data_flag + '.pth')

    if DATA_AUGMENTATION: 
        dataset = load_mnist(data_flag, BATCH_SIZE, download, num_workers, data_transform, data_aug = True)
        dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform, data_aug = True)
    else: 
        dataset = load_mnist(data_flag, BATCH_SIZE, download, num_workers, data_transform)
        dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform)

    attack_loader = dataset_attack["test_loader"]
    test_loader = dataset["test_loader"]

    model = create_resnet(data_flag)

    if MONTE_CARLO:
        # Apply Monte-Carlo Dropout
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.3

    model.to(dev)
    model.load_state_dict(torch.load(PATH))

    print("Attack Evaluation:")
    evaluate(model, attack_loader, "test", data_flag, dev=dev)
    print("Normal Evaluation:")
    evaluate(model, test_loader, "test", data_flag, dev=dev)

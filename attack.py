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


def complementary(image, is_rgb, y, x):
    image[y, x] = complement(image[y, x]) if is_rgb else 1 - image[y, x]

def zero_one(image, is_rgb, y, x, probability_1 = 0.5):
    rnd_number = np.random.uniform(0, 1)
    if is_rgb:
        image[y, x] = (1, 1, 1) if rnd_number <= probability_1 else (0, 0, 0)
    else:
        image[y, x] = 1 if rnd_number <= probability_1 else 0
    return image

def additive_noise(image, is_rgb, y, x, mean=0, std=1):
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
        foo = complementary
    elif attack == 'zero_one':
        foo = zero_one
    elif attack == 'additive_noise':
        foo = additive_noise
    else:
        exit(1, 'illegal function')

    indeces = select_random_pixels(height, width, seed)

    is_rgb = channels == 3
    for y, x in indeces[:k]:
        foo(image, is_rgb, y, x)

    return image

def complementary_tensor(image, is_rgb, y, x):
    image[:, y, x] = complement(image[:, y, x]) if is_rgb else 1 - image[:, y, x]

def zero_one_tensor(image, is_rgb, y, x, probability_1 = 0.5):
    rnd_number = np.random.uniform(0, 1)
    if is_rgb:
        image[:, y, x] = (1, 1, 1) if rnd_number <= probability_1 else (0, 0, 0)
    else:
        image[:, y, x] = 1 if rnd_number <= probability_1 else 0

    return image

def additive_noise_tensor(image, is_rgb, y, x, mean=0, std=1):
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
        foo = complementary_tensor
    elif attack == 'zero_one':
        foo = zero_one_tensor
    elif attack == 'additive_noise':
        foo = additive_noise_tensor
    else:
        exit(1, 'illegal function')

    is_rgb = channels == 3
    for y, x in indeces[:k]:
        foo(image, is_rgb, y, x)

def attack_complementary_pixel(input_image, image_type="grayscale", k=1):
    """
    Replaces k-pixels in the image with their complementary color (1-value in the case of grayscale)
    """
    # 1,64,64 / 3,64,64
    plot = True
    if plot:
        plt.imshow(input_image[0])
        plt.show()

    ys = torch.randperm(input_image[0].size(0))
    xs = torch.randperm(input_image[0].size(1))
    for i,y in enumerate(ys[:k]):
        x = xs[i]
        if image_type == "grayscale":
            input_image[0][y,x] = 1 - input_image[0][y,x]
        else:
            input_image[0][y,x] = complement(input_image[0][y,x])


    if plot and image_type == "grayscale":
        plt.imshow(input_image[0])
        plt.show()
    elif plot:
        i = np.asarray(input_image).T
        plt.imshow(i)
        plt.show()
    return input_image

def attack_0_1_pixel(input_image, image_type="grayscale", k=1, probability_1=0.5):
    """
    Replaces k-pixels in the input image with randomly black or white (0,1 for grayscale)
    Selection of black or white is determined by probability_1, where probability for 0 is 1-probability_1
    """
    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    indeces = select_random_pixels(height, width, seed)
    ys = torch.randperm(input_image[0].size(0))
    xs = torch.randperm(input_image[0].size(1))
    for i,y in enumerate(ys[:k]):
        x = xs[i]
        rnd_number = np.random.uniform(0,1)
        if image_type == "grayscale":
            input_image[0][y,x] = 1 if rnd_number <= probability_1 else 0
        else:
            input_image[0][y,x] = (1,1,1) if rnd_number <= probability_1 else (0,0,0)


    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()
    return input_image

def attack_addative_noise_on_pixel(input_image, image_type="grayscale", k=1, mean=0, std=1):
    """
    Replaces k-pixels in the input image with randomly black or white (0,1 for grayscale)
    Selection of black or white is determined by probability_1, where probability for 0 is 1-probability_1
    """
    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    indeces = select_random_pixels(height, width, seed)
    ys = torch.randperm(input_image[0].size(0))
    xs = torch.randperm(input_image[0].size(1))
    for i,y in enumerate(ys[:k]):
        x = xs[i]
        noise = np.random.normal(mean, std)
        if image_type == "grayscale":
            input_image[0][y,x] = np.clip(input_image[0][y,x] + noise, 0, 1)
        else:
            #TODO check if RGB is normalized
            input_image[0][y,x] = np.clip(input_image[0][y,x] + noise, 0, 1)


    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()
    return input_image

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

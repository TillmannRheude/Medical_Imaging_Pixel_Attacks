import os

import torch
from medmnist import INFO
from models import create_resnet
import torchvision.transforms as transforms

from evaluate import evaluate

from utils import load_mnist

import matplotlib.pyplot as plt

def random_single_pixel_flip_attack(input_image, image_type="grayscale", k=1):
    """
    Flips a single random pixel in a binary image
    :param input_image:
    :return:
    """
    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()
    ys = torch.randperm(input_image[0].size(0))
    xs = torch.randperm(input_image[0].size(1))
    idx = [ys[:k], xs[:k]]
    input_image[0][idx] = 1 - input_image[0][idx]
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()
    return input_image

# attack

def explicit_pixel_attack(input_image, pixel_list=[[30, 30], [32, 32]], image_type="grayscale"):

    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    for idx in pixel_list:
        input_image[0][idx] = 1 - input_image[0][idx]

    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    return input_image





if __name__ == "__main__":
    ## load model
    data_flag = 'octmnist'
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
        transforms.Lambda(explicit_pixel_attack),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/', data_flag + '.pth')

    dataset = load_mnist(data_flag, BATCH_SIZE, download, num_workers, data_transform)
    dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform)

    attack_loader = dataset_attack["test_loader"]
    test_loader = dataset["test_loader"]

    model = create_resnet(data_flag)
    model.to(dev)
    model.load_state_dict(torch.load(PATH))

    print("Attack Evaluation:")
    evaluate(model, attack_loader, "test", data_flag, dev=dev)
    print("Normal Evaluation:")
    evaluate(model, test_loader, "test", data_flag, dev=dev)

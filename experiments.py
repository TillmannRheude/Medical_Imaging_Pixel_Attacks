import os
import torch
from medmnist import INFO
from models import create_resnet
import torchvision.transforms as transforms
from evaluate import evaluate
from utils import load_mnist
import matplotlib.pyplot as plt
import json
import numpy as np

from attack import attack_single_image


def random_pixels(input_image):

    with open('attacked_pixels.json', 'r') as f:
        inf = json.load(f)

    k = inf['k']
    attack = inf['attack_func']
    input_image = attack_single_image(input_image[0].numpy(), attack=attack, k=k)
    input_image = torch.tensor(np.asarray([input_image,]))
    return input_image


def explicit_pixel_attack(input_image, image_type="grayscale"):
    plot = False
    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    pixels = {}
    with open('attacked_pixels.json', 'r') as f:
        pixels = json.load(f)

    for idx in pixels['pixels']:
        input_image[0][idx[0]][idx[1]] = 1 - input_image[0][idx[0]][idx[1]]

    if plot:
        plt.imshow(input_image[0], cmap='gray')
        plt.show()

    return input_image


def create_attacked_dataset(type, data_flag='resnet18_octmnist'):
    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/', data_flag + '.pth')
    data_flag = data_flag.split("_")[-1]
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    download = True
    num_workers = 2
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    lr = 0.001
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    ## hear the different attacks will be performed on the dataset
    if type == "random":
        attack_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Lambda(random_pixels),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif type == "explicit":
        attack_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Lambda(explicit_pixel_attack),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


    dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform)

    attack_loader = dataset_attack["test_loader"]
    # load model
    model = create_resnet(data_flag)
    model.to(dev)
    model.load_state_dict(torch.load(PATH))

    return model, attack_loader, dev


def experiment_location_vs_error(size, data_flag):
    n = size
    map = []
    step = int(64 / n)
    for x in range(0, 64, step):
        map.append([])
        for y in range(0, 64, step):
            pixels = []
            for i in range(step):
                for ii in range(step):
                    pixels.append([x + i, y + ii])

            with open('attacked_pixels.json', 'w') as f:
                json.dump({'pixels': pixels}, f)

            model, loader, dev = create_attacked_dataset("explicit", data_flag=data_flag)
            acc = evaluate(model, loader, "test", data_flag.split("_")[-1], dev=dev)
            map[int(x / step)].append(acc)

    print(map)
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.show()


def experiment_count_vs_error(count, step, attack_func, data_flag):
    x_count = []
    y_error = []

    for k in range(64,count, step):
        with open('attacked_pixels.json', 'w') as f:
            json.dump({'k': k, 'attack_func': attack_func}, f)

        model, loader, dev = create_attacked_dataset("random", data_flag=data_flag)
        acc = evaluate(model, loader, "test", data_flag.split("_")[-1], dev=dev)
        x_count.append(k)
        y_error.append(acc)

    print(x_count)
    print(y_error)

    plt.plot(x_count, y_error, '.b-')
    plt.ylim([0, 1])
    plt.show()



if __name__ == "__main__":

    print("start experiment")
    #experiment_location_vs_error(16, "resnet18_octmnist")
    experiment_count_vs_error(70, 1, "zero_one", "resnet18_octmnist")

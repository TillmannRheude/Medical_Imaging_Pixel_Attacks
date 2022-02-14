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

from attack import attack_tensor_image, explicit_pixel_attack_tensor


def random_pixels(input_image):

    with open('attacked_pixels.json', 'r') as f:
        inf = json.load(f)

    k = inf['k']
    attack = inf['attack_func']
    input_image = attack_tensor_image(input_image, attack=attack, k=k)
    return input_image

def explicit_pixels(input_image):

    with open('attacked_pixels.json', 'r') as f:
        inf = json.load(f)

    pxls = inf['pixels']
    attack = inf['attack_func']
    input_image = explicit_pixel_attack_tensor(input_image, attack=attack, pixel_list=pxls)

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
    MONTE_CARLO = True
    DATA_AUGMENTATION = True

    model = data_flag
    data_flag = data_flag.split("_")[1]
    ## load model
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
            transforms.Lambda(explicit_pixels),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/' + model + '.pth')

    if DATA_AUGMENTATION:
        dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform, data_aug=True)
    else:
        dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform)

    attack_loader = dataset_attack["test_loader"]
    model = create_resnet(data_flag, dropout=False)
    model.to(dev)
    model.load_state_dict(torch.load(PATH))
    return model, attack_loader, dev



def experiment_location_vs_error(size, data_flag, attack_func="zero_one"):
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
                json.dump({'pixels': pixels, 'attack_func': attack_func}, f)

            model, loader, dev = create_attacked_dataset("explicit", data_flag=data_flag)
            acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
            map[int(x / step)].append(acc[0])

    print(map)
    flag = data_flag.split("_")
    model = flag[0]
    dataset = flag[1]
    train_type = ""
    PATH = os.path.join(os.path.abspath(os.getcwd()), 'experiment_results/heatmap_' + data_flag + "_" + attack_func + '.png')

    if len(flag) == 3:
        train_type = flag[2]
        plt.title("Model: " + model + " Dataset: " + dataset + " Attack: " + attack_func + " Type: " + train_type)
    else:
        plt.title("Model: " + model + " Dataset: " + dataset + " Attack: " + attack_func)


    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.savefig(PATH)
    plt.show()


def experiment_count_vs_error(count, step, attack_func, data_flag):
    x_count = []
    y_error = []

    for k in range(0,10, 1):
        print(k)
        with open('attacked_pixels.json', 'w') as f:
            json.dump({'k': k, 'attack_func': attack_func}, f)

        model, loader, dev = create_attacked_dataset("random", data_flag=data_flag)
        acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
        x_count.append(k)
        y_error.append(acc)

    for k in range(10,100, 10):
        print(k)
        with open('attacked_pixels.json', 'w') as f:
            json.dump({'k': k, 'attack_func': attack_func}, f)

        model, loader, dev = create_attacked_dataset("random", data_flag=data_flag)
        acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
        x_count.append(k)
        y_error.append(acc)

    for k in range(100, 1000, 100):
        print(k)
        with open('attacked_pixels.json', 'w') as f:
            json.dump({'k': k, 'attack_func': attack_func}, f)

        model, loader, dev = create_attacked_dataset("random", data_flag=data_flag)
        acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
        x_count.append(k)
        y_error.append(acc)

    for k in range(1000, 4096, 1000):
        with open('attacked_pixels.json', 'w') as f:
            json.dump({'k': k, 'attack_func': attack_func}, f)

        model, loader, dev = create_attacked_dataset("random", data_flag=data_flag)
        acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
        x_count.append(k)
        y_error.append(acc)



    print(x_count)
    print(y_error)

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'experiment_results/' + data_flag + "_" + attack_func +'.png')

    plt.plot(x_count, y_error, '.-', label=("ACC", "AUC"))
    plt.ylim([0, 1])
    plt.xscale("log")

    flag = data_flag.split("_")
    model = flag[0]
    dataset = flag[1]
    train_type = ""
    if len(flag) == 3:
        train_type = flag[2]
        plt.title("Model: " + model + " Dataset: " + dataset + " Attack: " + attack_func + " Type: " + train_type)
    else:
        plt.title("Model: " + model + " Dataset: " + dataset + " Attack: " + attack_func)

    plt.xlabel("Attacked Pixels")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(PATH)
    plt.show()




if __name__ == "__main__":

    print("start experiment")
    #experiment_location_vs_error(16, "resnet18_octmnist")
    #experiment_count_vs_error(500, 5, "zero_one", "resnet18_octmnist")
    #experiment_count_vs_error(500, 5, "complementary", "resnet18_octmnist")
    #experiment_count_vs_error(500, 5, "additive_noise", "resnet18_octmnist")

    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_octmnist_dataaug")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_octmnist_dataaug")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_octmnist_dataaug")
    #
    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_bloodmnist")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_bloodmnist")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_bloodmnist")
    #
    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_breastmnist")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_breastmnist")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_breastmnist")
    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_octmnist_mcdropout")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_octmnist_mcdropout")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_octmnist_mcdropout")

    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_retinamnist")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_retinamnist")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_retinamnist")

    # experiment_count_vs_error(500, 5, "zero_one", "resnet18_tissuemnist")
    # experiment_count_vs_error(500, 5, "complementary", "resnet18_tissuemnist")
    # experiment_count_vs_error(500, 5, "additive_noise", "resnet18_tissuemnist")


    # experiment_location_vs_error(16,"resnet18_octmnist")
    # experiment_location_vs_error(16, "resnet18_bloodmnist")
    # experiment_location_vs_error(16, "resnet18_breastmnist")
    experiment_location_vs_error(16,"resnet18_octmnist_dataaug")
    experiment_location_vs_error(16, "resnet18_retinamnist")



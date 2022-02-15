import os
import torch
from medmnist import INFO
from models import create_resnet
import torchvision.transforms as transforms
from evaluate import evaluate
from utils import load_mnist, get_mnist_dataset
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle

import torch.utils.data as data

def dump_raw(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_raw(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

from attack import attack_tensor_image, explicit_pixel_attack_tensor

seed = 1234






def create_attacked_dataset(attack_function, attack_dimension, data_flag='resnet18_octmnist'):
    s = data_flag.split("_")
    mc = len(s) == 3 and s[2] == 'mcdropout'
    model = data_flag
    data_flag = s[1]
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
    attack_transform = transforms.Compose([
        transforms.Resize(attack_dimension),
        transforms.ToTensor(),
        transforms.Lambda(attack_function),
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/' + model + '.pth')

    #dataset_attack = load_mnist(data_flag, BATCH_SIZE, download, num_workers, attack_transform)

    test_dataset = get_mnist_dataset(data_flag, test=True, download=download, data_transform=attack_transform, data_aug=False)
    attack_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=num_workers)

    model = create_resnet(data_flag, dropout=mc)
    model.to(dev)
    model.load_state_dict(torch.load(PATH))
    return model, attack_loader, dev

def experiment_location_vs_error(size, data_flag, attack_func):
    print('************************************************')
    print(f"Starting experiment: {data_flag}_{attack_func}")
    print('************************************************')
    map = np.zeros((size,size))
    num_pixels = size * size
    for y, row in enumerate(map):
        for x, value in enumerate(row):
            progress = (y) * size + x+1
            print(f'Pixel: {y},{x}  Progress: {progress}/{num_pixels}')
            def attack_function(image):
                return explicit_pixel_attack_tensor(image, attack_func, [[y,x]], seed=seed)

            model, loader, dev = create_attacked_dataset(attack_function, size, data_flag=data_flag)

            _, acc = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
            map[y,x] = acc

    raw = f'location_acc_experiments_raw/{data_flag}_{attack_func}.pth'
    dump_raw(raw, map)
    print(f'Dumped results to {raw}')


def experiment_count_vs_error( data_flag, attack_func, size=28):
    print('************************************************')
    print(f"Starting experiment: {data_flag}_{attack_func}")
    print('************************************************')

    results={}
    # 28*28 = 784
    num_pixels = list(range(0,10, 1)) + list(range(10,100, 10)) + list(range(100, 784, 100)) + [784]

    for i, k in enumerate(num_pixels):
        print(f'Attacking {k} pixels Progress: {i+1}/{len(num_pixels)}')
        def random_attack(image):
            return attack_tensor_image(image, attack_func, k, seed=seed)

        model, loader, dev = create_attacked_dataset(random_attack, size, data_flag=data_flag)
        metrics = evaluate(model, loader, "test", data_flag.split("_")[1], dev=dev)
        results[k] = metrics
    raw = f'count_error_experiments_raw/{data_flag}_{attack_func}.pth'
    dump_raw(raw, map)
    return results


def experiment_count_vs_error_old(count, step, attack_func, data_flag, mc=False):
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

    #experiment_location_vs_error(28, "resnet18_octmnist", 'zero_one')
    #experiment_location_vs_error(28, "resnet18_octmnist", 'complementary')
    experiment_location_vs_error(28, "resnet18_octmnist", 'additive_noise')


    experiment_location_vs_error(28, "resnet18_octmnist_dataaug", 'zero_one')
    experiment_location_vs_error(28, "resnet18_octmnist_mcdropout", 'zero_one')

    experiment_location_vs_error(28, "resnet18_octmnist_dataaug", 'complementary')
    experiment_location_vs_error(28, "resnet18_octmnist_mcdropout", 'complementary')

    experiment_location_vs_error(28, "resnet18_octmnist_dataaug", 'additive_noise')
    experiment_location_vs_error(28, "resnet18_octmnist_mcdropout", 'additive_noise')

    exit()


    experiment_count_vs_error("resnet18_octmnist_mcdropout", "complementary")
    exit()
    experiment_location_vs_error(16, "resnet18_octmnist_mcdropout", mc=True)

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
    experiment_count_vs_error(16, 5, "complementary", "resnet18_octmnist_mcdropout", mc=True)
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



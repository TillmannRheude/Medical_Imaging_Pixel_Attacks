import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from utils import load_mnist


def train(model, train_loader, dev, lr, NUM_EPOCHS, task="multi-label, binary-class"):
    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # train

    for nr_epoch in range(1, NUM_EPOCHS + 1):

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {nr_epoch}")

                optimizer.zero_grad()

                inputs, targets = inputs.to(dev), targets.to(dev)

                outputs = model(inputs)

                targets = torch.squeeze(targets)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())  # , accuracy=100. * accuracy

    model_flag = f'resnet18_{data_flag}'
    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/', model_flag + '.pth')
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":

    data_flag = 'bloodmnist'

    download = True
    num_workers = 2
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = load_mnist(data_flag, BATCH_SIZE, download, num_workers, data_transform)

    train_loader = dataset["train_loader"]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    from models import create_resnet
    model = create_resnet(data_flag)
    model.to(dev)

    train(model, train_loader, dev, lr, NUM_EPOCHS, task)


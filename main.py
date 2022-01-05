""" 
    Information

    Code inspired by: 
        https://colab.research.google.com/github/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb#scrollTo=Erbb5YvH4yHQ 
    Dataset by: 
        Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification". arXiv preprint arXiv:2110.14795, 2021.
        Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
""" 

################################################################ Imports ###############################################################
import medmnist
import torch 

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from time import sleep
from tqdm import tqdm
from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator

from utils import get_info_task_channels_classes, test


############################################################ Argument Parser ############################################################
# TODO 


############################################################ Hyperparameter #############################################################
data_flag = "tissuemnist" #"octmnist"
download = True 
NR_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

info, task, nr_channels, nr_classes = get_info_task_channels_classes(data_flag)

DataClass = getattr(medmnist, info["python_class"])


############################################################## Data Loading ############################################################
data_transform = transforms.Compose([
                                transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5], std = [0.5])
                                ])

train_dataset = DataClass(split = "train", transform = data_transform, download = download)
test_dataset = DataClass(split = "test", transform = data_transform, download = download)

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
validation_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


############################################################### Debug stuff ############################################################
#print("_" * 30)
#print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
#print("_" * 30)
#print(train_dataset)
#print("_" * 30)
# Show example picture
# train_dataset.montage(length = 1, save_folder = "testpictures")


############################################################### Build model ############################################################
model = models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.conv1 = nn.Conv2d(nr_channels, model.conv1.out_channels, kernel_size = model.conv1.kernel_size, stride = model.conv1.stride, padding = model.conv1.padding, bias=model.conv1.bias) # get the information with print(model)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features = nr_classes, bias = True)

model = model.to(device)

# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

for nr_epoch in range(1, NR_EPOCHS + 1):

    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            tepoch.set_description(f"Epoch {nr_epoch}")

            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            targets = torch.squeeze(targets)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item()) # , accuracy=100. * accuracy
            sleep(0.1)

print('==> Evaluating ...')
test('train', model, task, data_flag, validation_loader, test_loader)
test('test', model, task, data_flag, validation_loader, test_loader)
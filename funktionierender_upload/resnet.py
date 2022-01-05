from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from load_medmnist import load_mnist
from evaluate_network import test

import torchvision.models as models

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

data_flag = 'octmnist'
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
test_loader = dataset["test_loader"]
train_loader_at_eval = dataset["train_loader_at_eval"]

# define a simple CNN model
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(n_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                        stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
output_layer = nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
model.fc = output_layer

model = model.to(dev)

# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    model.train()
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(dev)
        targets = targets.to(dev)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

print('==> Evaluating ...')
test(model, train_loader_at_eval, "train", data_flag)
test(model, test_loader, "test", data_flag)
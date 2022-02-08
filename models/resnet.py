import torch.nn as nn
from medmnist import INFO
import torchvision.models as models

def create_model(data_flag, dropout=False):

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(n_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
    if dropout:
        output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
        )
    else:
        output_layer = nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)

    model.fc = output_layer

    return model


########## Imports #########
import medmnist
import torch 

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from medmnist import INFO, Evaluator
from utils import get_info_task_channels_classes
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from .random_pixels import AddRandomGaussianNoise, RandomGaussianBlur

def get_mnist_dataset(data_flag, test=False, download=True, data_transform=None, data_aug = False):

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # preprocessing
    if not data_transform:
        data_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    
    if data_aug: 
        data_transform.transforms.insert(1, RandomGaussianBlur())
        data_transform.transforms.insert(1, AddRandomGaussianNoise())

    DataClass = getattr(medmnist, info['python_class'])
    return DataClass(split='test' if test else 'train', transform=data_transform, download=download)


def load_mnist(data_flag="octmnist", BATCH_SIZE=128, download=True, num_workers=4, data_transform=None, data_aug = False):

    # load the data
    if data_aug:
        train_dataset = get_mnist_dataset(data_flag, download=download, data_transform=data_transform, data_aug = True)
        test_dataset = get_mnist_dataset(data_flag, test=True, download=download, data_transform=data_transform, data_aug = True)
    else: 
        train_dataset = get_mnist_dataset(data_flag, download=download, data_transform=data_transform)
        test_dataset = get_mnist_dataset(data_flag, test=True, download=download, data_transform=data_transform)


    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return {"train_dataset": train_dataset, "test_dataset": test_dataset,
            "train_loader": train_loader, "train_loader_at_eval": train_loader_at_eval, "test_loader": test_loader}


if __name__ == "__main__":
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    import matplotlib.pyplot as plt

    dict = load_mnist()
    train_dataset = dict["train_dataset"]
    test_dataset = dict["test_dataset"]

    print(train_dataset)
    print("===================")
    print(test_dataset)

    # montage
    plt.imshow(train_dataset.montage(length=1), cmap='Greys')
    plt.show()
    plt.imshow(train_dataset.montage(length=20), cmap='Greys')
    plt.show()

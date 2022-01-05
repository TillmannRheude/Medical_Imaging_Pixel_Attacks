import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

def load_mnist(data_flag="octmnist", BATCH_SIZE=128, download=True, num_workers=4, data_transform=None):
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    ###############################################################

    # preprocessing
    if not data_transform:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)


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
